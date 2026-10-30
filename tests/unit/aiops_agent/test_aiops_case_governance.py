"""案例脱敏、用途分级及撤权后阻断新检索回归。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from aiops_agent.application.case_redaction import CaseRedactor
from aiops_agent.application.case_source_validation import validate_case_sources
from aiops_agent.application.operations_knowledge import _diagnosis_case_markdown, OperationsKnowledgeService
from aiops_agent.contracts.knowledge import DiagnosisCaseProfile, KnowledgeSearchRequest
from aiops_agent.playbooks import canonical_hash
from platform_core.identity import uuid7


def profile(**updates):
    return DiagnosisCaseProfile.model_validate({
        "case_kind": "DIAGNOSTIC_REFERENCE",
        "problem_signature": {"normalized_symptoms": ["连接 10.0.0.88 失败"]},
        "environment": {"database_type": "POSTGRESQL", "username": "real_user"},
        "root_cause": {"summary": "核对 /home/customer/key 与 https://private.example/path?token=secret", "evidence_refs": ["artifact:raw#c1"]},
        "verification": {"result": "NOT_VERIFIED"},
        "source": {"investigation_record_id": "raw-record", "run_ids": ["raw-run"]},
        **updates,
    })


def test_redaction_precedes_index_and_keeps_lineage_and_original_profile_separate():
    original = profile()
    sanitized = CaseRedactor().profile(original)
    text = _diagnosis_case_markdown(title="10.0.0.88 调查", profile=sanitized).decode()
    wire = sanitized.model_dump_json()
    assert "10.0.0.88" not in text and "/home/customer" not in text and "private.example" not in text
    assert "real_user" not in wire
    assert sanitized.source == original.source
    assert sanitized.root_cause["evidence_refs"] == original.root_cause["evidence_refs"]
    assert sanitized.redaction.status == "REVIEW_REQUIRED"
    assert "10.0.0.88" in original.problem_signature["normalized_symptoms"][0]
    assert original.redaction is None


@pytest.mark.parametrize("kind", ["VERIFIED_RESOLUTION", "VERIFIED_OPTIMIZATION"])
def test_positive_case_cannot_be_created_from_manual_recovery_or_different_pdf_refs(kind):
    with pytest.raises(ValidationError):
        profile(case_kind=kind, actions=({"summary": "操作", "evidence_refs": ["receipt"]},),
            verification={"result": "RESOLVED", "before_refs": ["pdf-a"], "after_refs": ["pdf-b"]})


def test_negative_experience_is_traceable_and_is_not_a_recommendation():
    value = profile(case_kind="NEGATIVE_EXPERIENCE",
        actions=({"summary": "调整参数后回退", "evidence_refs": ["manual-receipt"]},),
        verification={"result": "ADVERSE", "method": "USER_REPORTED"})
    assert value.case_kind == "NEGATIVE_EXPERIENCE"
    assert value.verification["method"] == "USER_REPORTED"
    with pytest.raises(ValidationError):
        profile(case_kind="NEGATIVE_EXPERIENCE")


def source_uow(failure=None):
    artifact_id, run_id, target_id, agent_id, domain_id, version_id = (str(uuid7()) for _ in range(6))
    payload = {"schema_version": "DIAGNOSIS_CLOSURE.v1"}
    digest = canonical_hash(payload)
    source = SimpleNamespace(source_report_id="report", source_run_id=run_id,
        source_artifact_id=artifact_id, content_hash=digest)
    artifact = SimpleNamespace(ops_run_id=run_id, payload_json=payload, content_hash=digest)
    run = SimpleNamespace(ops_run_id=run_id, target_id=target_id, agent_id=agent_id,
        plan_snapshot_json={"capability_snapshot": {"source_snapshots": [{"source_id": "monitor"}]}})
    if failure == "hash":
        artifact.content_hash = "wrong"
    report = SimpleNamespace(is_current=failure != "report_corrected", status="READY", report_id="report")
    repository = SimpleNamespace(
        get_version_relations=AsyncMock(return_value={"sources": [source]}),
        list_published_candidates=AsyncMock(),
    )
    uow = SimpleNamespace(
        operations_knowledge=repository,
        inspections=SimpleNamespace(get_report_scoped=AsyncMock(return_value=report),
            list_report_successors_scoped=AsyncMock(return_value=[])),
        runs=SimpleNamespace(get_artifact=AsyncMock(return_value=artifact), get_run_scoped=AsyncMock(return_value=run)),
        agents=SimpleNamespace(
            get=AsyncMock(return_value=SimpleNamespace(current_version_id=version_id)),
            version_has_target=AsyncMock(return_value=failure != "target_revoked"),
            version_source_ids=AsyncMock(return_value=[] if failure == "source_revoked" else ["monitor"]),
        ),
    )
    return uow, domain_id, version_id, agent_id


@pytest.mark.parametrize("failure", [None, "hash", "report_corrected", "target_revoked", "source_revoked"])
def test_source_health_rechecks_versions_hashes_target_and_source_authorization(failure):
    uow, domain, version, agent = source_uow(failure)
    result = asyncio.run(validate_case_sources(uow=uow, domain_id=domain, version_id=version, agent_id=agent))
    assert result == (failure is None)


def test_case_search_filters_invalid_source_before_kc_reads_body():
    async def run():
        uow, domain, version, agent = source_uow("source_revoked")
        uow.operations_knowledge.list_published_candidates.return_value = [(
            SimpleNamespace(asset_id=str(uuid7()), asset_kind="DIAGNOSIS_CASE", display_name="case"),
            SimpleNamespace(asset_version_id=version, version_no=1, profile_json=profile().model_dump()),
            SimpleNamespace(collection_id=str(uuid7()), bundle_revision_id=str(uuid7())),
        )]
        class Uow:
            async def __aenter__(self): return uow
            async def __aexit__(self, *args): pass
        service = OperationsKnowledgeService.__new__(OperationsKnowledgeService)
        service._uow_factory = Uow
        service._kc = SimpleNamespace(discover=AsyncMock())
        kwargs = {"domain_id": domain, "agent_id": agent,
            "request": KnowledgeSearchRequest(query="连接失败", knowledge_base_ids=(uuid7(),)),
            "context": SimpleNamespace()}
        # 两平台的租户由现有应用边界负责，不在案例中新增共享通道。
        import inspect
        if "tenant_id" in inspect.signature(service.search).parameters:
            kwargs["tenant_id"] = uuid7()
        result = await service.search(**kwargs)
        assert not result["results"]
        service._kc.discover.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("relation,expected", [("SUPPORTS", True), ("REFUTES", False), ("SUPERSEDES", False)])
def test_later_explicit_revision_blocks_stale_case_but_not_unrelated_investigation(monkeypatch, relation, expected):
    from aiops_agent.application.case_source_validation import _case_claims_still_current
    from aiops_agent.contracts.investigation import InvestigationRecord
    from aiops_agent.contracts.diagnosis_state import ClaimRevision
    root_run, target, later_run_id, root_artifact, later_artifact_id = (str(uuid7()) for _ in range(5))
    old = SimpleNamespace(record_id="investigation-1", target_id=target,
        claims=(SimpleNamespace(claim_id="c1", revision_state="ACTIVE"),))
    data = {"frozen_record": "third-revision"}
    later = SimpleNamespace(record_id=old.record_id, target_id=target, run_id=later_run_id,
        parent=SimpleNamespace(artifact_id="middle-artifact"),
        claim_revisions=(ClaimRevision(prior_claim_id="c1", prior_artifact_id=root_artifact,
            current_claim_id="c1", relation=relation, reason="取得新的区分性证据"),))
    monkeypatch.setattr(InvestigationRecord, "model_validate", lambda *_args, **_kwargs: later)
    payload = {"investigation_record": data}
    uow = SimpleNamespace(
        turns=SimpleNamespace(
            get_run_link_by_ops_run_id=AsyncMock(return_value=SimpleNamespace(turn_id="t1")),
            get_turn=AsyncMock(return_value=SimpleNamespace(conversation_id="conversation-1", turn_no=1)),
            list_turns=AsyncMock(return_value=[SimpleNamespace(turn_id="t3")]),
            get_run_link=AsyncMock(return_value=SimpleNamespace(ops_run_id=later_run_id)),
        ),
        runs=SimpleNamespace(
            get_run_scoped=AsyncMock(return_value=SimpleNamespace(ops_run_id=later_run_id, target_id=target, final_artifact_id=later_artifact_id)),
            get_artifact=AsyncMock(return_value=SimpleNamespace(artifact_id=later_artifact_id, ops_run_id=later_run_id, payload_json=payload,
                schema_version="AIOPS_TURN_RESULT.v1", content_hash=canonical_hash(payload))),
        ),
    )
    kwargs = {"uow": uow, "domain_id": "domain", "run": SimpleNamespace(ops_run_id=root_run),
        "artifact": SimpleNamespace(artifact_id=root_artifact), "record": old}
    assert asyncio.run(_case_claims_still_current(**kwargs)) is expected
    later.record_id = "different-investigation"
    assert asyncio.run(_case_claims_still_current(**kwargs)) is True


def test_profile_verified_flags_do_not_replace_a_controlled_effect_artifact():
    from aiops_agent.application.case_source_validation import validate_case_effect
    checked = profile(
        case_kind="VERIFIED_OPTIMIZATION",
        actions=({"summary": "已登记动作", "evidence_refs": ["receipt"]},),
        verification={"controlled_verification_ref": "artifact:" + str(uuid7()),
            "comparability": "COMPARABLE", "guardrails_passed": True, "measured_effect": "IMPROVED"},
    )
    uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(return_value=None)))
    assert not asyncio.run(validate_case_effect(uow=uow, domain_id=uuid7(), profile=checked))
    uow.runs.get_artifact.assert_awaited_once()


def test_knowledge_rechecks_publication_after_remote_body_retrieval():
    async def run():
        domain, revision, version = (str(uuid7()) for _ in range(3))
        candidate = (
            SimpleNamespace(asset_id=str(uuid7()), asset_kind="MANUAL", display_name="已撤回手册"),
            SimpleNamespace(asset_version_id=version, version_no=1, profile_json={"scope": {}}),
            SimpleNamespace(collection_id=str(uuid7()), bundle_revision_id=revision),
        )
        repo = SimpleNamespace(list_published_candidates=AsyncMock(side_effect=[[candidate], []]))
        class Uow:
            async def __aenter__(self): return SimpleNamespace(operations_knowledge=repo)
            async def __aexit__(self, *args): pass
        service = OperationsKnowledgeService.__new__(OperationsKnowledgeService)
        service._uow_factory = Uow
        service._kc = SimpleNamespace(
            discover=AsyncMock(return_value={"candidates": [{"bundle_revision_id": revision}]}),
            retrieve_evidence=AsyncMock(return_value={"citations": [{"bundle_revision_id": revision, "quote": "不应泄露"}]}),
        )
        kwargs = {"domain_id": domain, "agent_id": str(uuid7()),
            "request": KnowledgeSearchRequest(query="方法", knowledge_base_ids=(uuid7(),)), "context": SimpleNamespace()}
        import inspect
        if "tenant_id" in inspect.signature(service.search).parameters:
            kwargs["tenant_id"] = uuid7()
        result = await service.search(**kwargs)
        assert result["results"] == []
        assert repo.list_published_candidates.await_count == 2
    asyncio.run(run())
