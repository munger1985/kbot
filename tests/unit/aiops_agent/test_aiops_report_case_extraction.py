"""报告提炼案例的生产服务回归，覆盖来源复核和人工处置边界。"""

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.configuration.common import sha256_json
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.operations_knowledge import OperationsKnowledgeService
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_report_lifecycle import fixture, generate


async def case_fixture(*, closure=False, result="RESOLVED", action_refs=True):
    f = fixture()
    f.run.agent_id = uuid7()
    f.uow.turns.get_run_link_by_ops_run_id = AsyncMock(return_value=None)
    f.uow.agents = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(current_version_id=uuid7())),
        version_has_target=AsyncMock(return_value=True),
        version_source_ids=AsyncMock(return_value=[]),
    )
    f.uow.targets = SimpleNamespace(get_scoped=AsyncMock(return_value=SimpleNamespace(
        db_type="POSTGRESQL", version_code="16", environment="TEST", capabilities_json={},
    )))
    f.state.versions, f.state.assets, f.state.relations = {}, {}, {}
    async def store(**kwargs):
        asset, version = kwargs["asset"], kwargs["version"]
        asset.row_version = version.row_version = 1
        f.state.assets[asset.asset_id] = asset
        f.state.versions[version.asset_version_id] = version
        f.state.relations[version.asset_version_id] = {key: kwargs[key] for key in ("sources", "indexes")}
        for index in kwargs["indexes"]:
            index.index_ref_id = index.index_ref_id or uuid7()
    f.uow.operations_knowledge = SimpleNamespace(
        find_by_source_hash=AsyncMock(return_value=None),
        find_case_by_artifact_id=AsyncMock(return_value=None),
        find_case_by_report_id=AsyncMock(return_value=None),
        add_asset_version=AsyncMock(side_effect=store),
        get_version=AsyncMock(side_effect=lambda **kw: f.state.versions.get(kw["asset_version_id"])),
        get_asset=AsyncMock(side_effect=lambda **kw: f.state.assets.get(kw["asset_id"])),
        get_version_relations=AsyncMock(side_effect=lambda **kw: f.state.relations.get(kw["asset_version_id"], {})),
        add_review=AsyncMock(),
    )
    f.uow.outbox.get_by_idempotency = AsyncMock(return_value=None)
    f.uow.inspections.get_report_scoped = AsyncMock(side_effect=lambda **kw: f.state.current)
    f.uow.inspections.list_report_sources = AsyncMock(side_effect=lambda **kw:
        f.uow.inspections.add_report_sources.await_args.args[0])
    if closure:
        payload = {"schema_version": "DIAGNOSIS_CLOSURE.v1",
            "conversation_id": str(f.conversation_id), "source_turn_ids": [str(f.turn_id)],
            "profile": {"problem_signature": {"normalized_symptoms": ["业务曾有等待"]},
                "root_cause": {"summary": "人工复盘，不代替数据库证据"},
                "actions": [{"summary": "调整调用频率",
                    "evidence_refs": [str(f.artifact.artifact_id)] if action_refs else []}],
                "verification": {"method": "USER_REPORTED", "result": result,
                    "direct_effect": "UNKNOWN", "measured_effect": "INCONCLUSIVE",
                    "comparability": "UNKNOWN", "symptom_recovered": True,
                    "guardrails_passed": None, "attribution_strength": "UNKNOWN",
                    "causal_limitations": ["人工反馈，未经受控对照验证"]}},
        }
        f.state.closure = SimpleNamespace(artifact_id=uuid7(), ops_run_id=f.run_id,
            ops_task_id=f.task_id, schema_version="DIAGNOSIS_CLOSURE.v1",
            payload_json=payload, content_hash=sha256_json(payload), security_level=1,
            provenance_json={"source_artifact_id": str(f.artifact.artifact_id)})
        f.state.artifacts[f.state.closure.artifact_id] = f.state.closure
    await generate(f, conversation=closure)
    f.uow.outbox.add.reset_mock()
    f.report = f.state.current
    f.content_artifact = f.state.artifacts[f.report.content_artifact_id]
    f.report.is_current = True
    f.service = OperationsKnowledgeService.__new__(OperationsKnowledgeService)
    f.service._uow_factory = lambda: f.manager
    f.service._plan_capture_reader = None
    f.service._structured_model = SimpleNamespace(generate_structured=AsyncMock(
        return_value=SimpleNamespace(output={"spans": []})))
    f.service._prompt_registry = SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace(
        ref=lambda: {"prompt_sha256": "prompt-hash"}, content="只识别敏感字面值")))
    catalog = SimpleNamespace(get_model=AsyncMock(return_value={
        "model_name": "fixture-model", "served_model_name": "fixture-model"}))
    f.service._model_catalog = f.service._llm_model_catalog = catalog
    f.service._collection = AsyncMock(return_value={
        "collection_id": str(uuid7()), "models": {"parser_llm": str(uuid7())}})
    f.service._kc = SimpleNamespace(
        ingest_user_file=AsyncMock(return_value=SimpleNamespace(payload={"items": [{
            "status": "ACCEPTED", "bundle_id": str(uuid7()), "bundle_revision_id": str(uuid7())}]})),
        review_user_intake=AsyncMock(),
    )
    f.context = SimpleNamespace(principal_id=f.actor, asserted_user_id=str(f.actor), client_id="fixture")
    return f


async def extract(f):
    kwargs = dict(domain_id=f.domain, report_id=f.report.report_id, context=f.context)
    if "tenant_id" in inspect.signature(f.service.extract_case).parameters:
        kwargs["tenant_id"] = uuid7()
    return await f.service.extract_case(**kwargs)


def invalidate(f, failure):
    if failure == "report_hash":
        f.report.content_hash = "f" * 64
    elif failure == "payload_hash":
        f.content_artifact.payload_json["summary"] = "被替换的内容"
    elif failure == "source_hash":
        f.artifact.content_hash = "f" * 64
    elif failure == "source_missing":
        f.state.artifacts.pop(f.artifact.artifact_id)
    elif failure == "target_revoked":
        f.uow.agents.version_has_target.return_value = False
    elif failure == "superseded":
        f.report.is_current = False


@pytest.mark.parametrize("failure", [
    "report_hash", "payload_hash", "source_hash", "source_missing", "target_revoked", "superseded",
])
def test_eligibility_and_extract_reject_invalid_source_before_model_or_index(failure):
    async def run():
        f = await case_fixture()
        invalidate(f, failure)
        result = await f.service.case_eligibility(domain_id=f.domain, report_id=f.report.report_id)
        assert result["eligible"] is False
        with pytest.raises(AIOpsApplicationError):
            await extract(f)
        f.service._structured_model.generate_structured.assert_not_awaited()
        f.service._kc.ingest_user_file.assert_not_awaited()
    asyncio.run(run())


def test_idempotent_case_return_does_not_bypass_current_source_validation():
    async def run():
        f = await case_fixture()
        result = await extract(f)
        pair = (f.state.assets[next(iter(f.state.assets))], f.state.versions[next(iter(f.state.versions))])
        f.uow.operations_knowledge.find_by_source_hash.return_value = pair
        invalidate(f, "target_revoked")
        with pytest.raises(AIOpsApplicationError):
            await extract(f)
        assert f.service._kc.ingest_user_file.await_count == 1
        assert result["replayed"] is False
    asyncio.run(run())


@pytest.mark.parametrize("result,action_refs,kind", [
    ("RESOLVED", True, "DIAGNOSTIC_REFERENCE"),
    ("ADVERSE", True, "NEGATIVE_EXPERIENCE"),
    ("ADVERSE", False, "DIAGNOSTIC_REFERENCE"),
])
def test_manual_outcome_keeps_provenance_and_never_becomes_verified(result, action_refs, kind):
    async def run():
        f = await case_fixture(closure=True, result=result, action_refs=action_refs)
        outcome = await extract(f)
        profile = outcome["version"]["profile"]
        assert profile["case_kind"] == kind
        assert profile["verification"] == f.state.closure.payload_json["profile"]["verification"]
        assert profile["actions"][0]["source_kind"] == "USER_PROVIDED"
        assert profile["actions"][0]["evidence_refs"] == (
            [str(f.artifact.artifact_id)] if action_refs else [])
        assert profile["actions"][0]["registration_ref"] == "artifact:" + str(f.state.closure.artifact_id)
        body = f.service._kc.ingest_user_file.await_args.kwargs["body"].decode()
        assert "人工" in body and "不代替" in body
        assert outcome["version"]["status"] == "PROCESSING"
        assert outcome["asset"]["current_version_id"] is None
    asyncio.run(run())


def test_valid_diagnostic_report_retains_exact_claim_lineage_and_requires_review():
    async def run():
        f = await case_fixture()
        result = await extract(f)
        profile = result["version"]["profile"]
        assert profile["case_kind"] == "DIAGNOSTIC_REFERENCE"
        assert profile["source"]["claim_refs"][0]["record_id"] == f.record.record_id
        assert profile["source"]["claim_refs"][0]["claim_id"] == "c1"
        assert not profile["actions"]
        assert result["version"]["status"] == "PROCESSING"
        assert f.uow.operations_knowledge.add_asset_version.await_count == 1
        assert f.service._kc.review_user_intake.await_count == 1
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["target_revoked", "superseded", "payload_hash", "report_hash", "source_hash"])
def test_source_revoked_during_semantic_scan_never_reaches_index(failure):
    async def run():
        f = await case_fixture()
        async def scan(**kw):
            invalidate(f, failure)
            return SimpleNamespace(output={"spans": []})
        f.service._structured_model.generate_structured.side_effect = scan
        with pytest.raises(AIOpsApplicationError):
            await extract(f)
        f.service._kc.ingest_user_file.assert_not_awaited()
        f.uow.operations_knowledge.add_asset_version.assert_not_awaited()
    asyncio.run(run())


def test_source_revoked_during_intake_is_tracked_and_withdrawn_without_approval():
    async def run():
        f = await case_fixture()
        response = f.service._kc.ingest_user_file.return_value
        async def ingest(**kw):
            invalidate(f, "target_revoked")
            return response
        f.service._kc.ingest_user_file.side_effect = ingest
        with pytest.raises(AIOpsApplicationError):
            await extract(f)
        f.service._kc.review_user_intake.assert_not_awaited()
        version = next(iter(f.state.versions.values()))
        assert version.status == "REVIEW_REQUIRED"
        assert f.uow.outbox.add.await_count == 1
    asyncio.run(run())


@pytest.mark.parametrize("point", ["source_list", "claim_body", "claim_id", "anchor_target"])
def test_report_cannot_substitute_another_claim_or_unregistered_source(point):
    async def run():
        f = await case_fixture()
        if point == "source_list":
            f.uow.inspections.list_report_sources.return_value = []
            f.uow.inspections.list_report_sources.side_effect = None
        elif point == "anchor_target":
            f.content_artifact.payload_json["target_id"] = str(uuid7())
        else:
            key = "summary" if point == "claim_body" else "claim_id"
            f.content_artifact.payload_json["facts"][0][key] = "不属于原调查的判断"
        f.content_artifact.content_hash = sha256_json(f.content_artifact.payload_json)
        f.report.content_hash = f.content_artifact.content_hash
        with pytest.raises(AIOpsApplicationError):
            await extract(f)
        f.service._kc.ingest_user_file.assert_not_awaited()
    asyncio.run(run())


def test_source_revoked_during_index_approval_is_persistently_withdrawn():
    async def run():
        f = await case_fixture()
        async def approve(**kw):
            invalidate(f, "target_revoked")
        f.service._kc.review_user_intake.side_effect = approve
        with pytest.raises(AIOpsApplicationError):
            await extract(f)
        version = next(iter(f.state.versions.values()))
        assert version.status == "REVIEW_REQUIRED"
        assert f.uow.outbox.add.await_count == 1
        f.uow.operations_knowledge.add_review.assert_awaited_once()
    asyncio.run(run())


def test_intake_approval_failure_keeps_recoverable_exact_revision():
    async def run():
        f = await case_fixture()
        f.service._kc.review_user_intake.side_effect = RuntimeError("模拟索引审批连接中断")
        with pytest.raises(RuntimeError, match="连接中断"):
            await extract(f)
        version = next(iter(f.state.versions.values()))
        index = f.state.relations[version.asset_version_id]["indexes"][0]
        assert version.status == "PROCESSING"
        assert index.index_status == "PENDING_APPROVAL"
        detail = {
            "asset": {"asset_kind": "DIAGNOSIS_CASE"},
            "version": {"status": "PROCESSING"},
            "indexes": [{"status": "PENDING_APPROVAL", "collection_id": str(index.collection_id),
                "bundle_id": str(index.bundle_id), "bundle_revision_id": str(index.bundle_revision_id)}],
        }
        f.service.get_version = AsyncMock(return_value=detail)
        f.service._kc.review_user_intake.side_effect = None
        f.service._kc.get_bundle_status = AsyncMock(side_effect=RuntimeError("审批已恢复，停止本测试的后续状态查询"))
        kwargs = dict(domain_id=f.domain, asset_version_id=version.asset_version_id, context=f.context)
        if "tenant_id" in inspect.signature(f.service.reconcile_version).parameters:
            kwargs["tenant_id"] = uuid7()
        with pytest.raises(RuntimeError, match="审批已恢复"):
            await f.service.reconcile_version(**kwargs)
        assert index.index_status == "PROCESSING"
        assert f.service._kc.review_user_intake.await_count == 2
        assert f.service._kc.ingest_user_file.await_count == 1
    asyncio.run(run())


def test_source_revocation_during_model_setup_is_checked_before_actual_model_call():
    async def run():
        f = await case_fixture()
        collection = f.service._collection.return_value
        async def lookup(**kw):
            invalidate(f, "target_revoked")
            return collection
        f.service._collection.side_effect = lookup
        with pytest.raises(AIOpsApplicationError):
            await extract(f)
        f.service._structured_model.generate_structured.assert_not_awaited()
        f.service._kc.ingest_user_file.assert_not_awaited()
    asyncio.run(run())


def test_valid_idempotent_replay_does_not_repeat_semantic_scan_or_index_intake():
    async def run():
        f = await case_fixture()
        first = await extract(f)
        pair = (next(iter(f.state.assets.values())), next(iter(f.state.versions.values())))
        f.uow.operations_knowledge.find_by_source_hash.return_value = pair
        replayed = await extract(f)
        assert replayed["replayed"] is True
        assert replayed["version"]["asset_version_id"] == first["version"]["asset_version_id"]
        assert f.service._structured_model.generate_structured.await_count == 1
        assert f.service._kc.ingest_user_file.await_count == 1
    asyncio.run(run())


def test_parallel_intake_converges_on_already_persisted_revision():
    async def run():
        f = await case_fixture()
        await extract(f)
        pair = (next(iter(f.state.assets.values())), next(iter(f.state.versions.values())))
        f.uow.operations_knowledge.find_by_source_hash.side_effect = [None, pair]
        replayed = await extract(f)
        assert replayed["replayed"] is True
        assert f.uow.operations_knowledge.add_asset_version.await_count == 1
        assert any(call.kwargs.get("lock") is True for call in f.uow.inspections.get_report_scoped.await_args_list)
    asyncio.run(run())


def test_closure_registration_must_be_in_frozen_report_for_both_preflight_and_extract():
    async def run():
        f = await case_fixture(closure=True)
        f.content_artifact.payload_json["scope"]["closure_artifact_id"] = str(uuid7())
        f.content_artifact.content_hash = sha256_json(f.content_artifact.payload_json)
        f.report.content_hash = f.content_artifact.content_hash
        eligible = await f.service.case_eligibility(domain_id=f.domain, report_id=f.report.report_id)
        assert eligible["eligible"] is False
        with pytest.raises(AIOpsApplicationError):
            await extract(f)
        f.service._kc.ingest_user_file.assert_not_awaited()
    asyncio.run(run())




def test_case_revision_preserves_original_manual_method_and_action_registration():
    from aiops_agent.application.operations_knowledge import _prepare_case_revision
    from aiops_agent.contracts.knowledge import DiagnosisCaseDraft, DiagnosisCaseProfile
    async def run():
        f = await case_fixture(closure=True)
        result = await extract(f)
        original = DiagnosisCaseProfile.model_validate(result["version"]["profile"])
        forged = original.model_copy(deep=True)
        forged.actions[0]["source_kind"] = "EXECUTOR"
        forged.actions[0]["registration_ref"] = "伪造来源"
        revised = _prepare_case_revision(command=DiagnosisCaseDraft(
            expected_asset_row_version=1, base_version_id=uuid7(), display_name="人工修订", profile=forged,
        ), base_profile=original)
        assert revised.actions == original.actions
        assert revised.verification["method"] == "USER_REPORTED"
    asyncio.run(run())


def test_supported_effect_report_runs_through_real_extraction_service_without_auto_publication():
    from aiops_agent.application.diagnosis.investigation_record import freeze_investigation
    from tests.unit.aiops_agent.test_aiops_controlled_effect import service_fixture
    from tests.unit.aiops_agent.test_aiops_investigation_record import draft, evidence
    from tests.unit.aiops_agent.test_aiops_turn_answer import _proposal_artifact

    async def run():
        f = await case_fixture()
        effects, uow, run, proposal, command, kwargs = await service_fixture()
        context = SimpleNamespace(run_id=run.ops_run_id, task_id=proposal.ops_task_id,
            target_id=run.target_id, plan_snapshot={"answer_context": {"question": "调查业务阻塞"}})
        fact = evidence()
        record = freeze_investigation(draft=draft(), evidence=(fact,), context=context)
        root = {"schema_version": "AIOPS_TURN_RESULT.v1", "status": "COMPLETED",
            "investigation_record": record.model_dump(mode="json"), "evidence": [fact.model_dump(mode="json")]}
        snapshot = _proposal_artifact(mode="AGENT_EXECUTE")["payload"]
        snapshot["proposal"].update(proposal_id=str(proposal.proposal_id), run_id=str(run.ops_run_id),
            target_id=str(run.target_id), proposal_hash="b" * 64, command_hash="c" * 64)
        run.final_artifact_id, proposal.snapshot_artifact_id = uuid7(), uuid7()
        for identity, body in ((run.final_artifact_id, root), (proposal.snapshot_artifact_id, snapshot)):
            await uow.runs.add_artifact(SimpleNamespace(
                artifact_id=identity, ops_run_id=run.ops_run_id, artifact_key=str(identity),
                schema_version=body["schema_version"], payload_json=body, content_hash=sha256_json(body),
                trust_level="SOURCE_VERIFIED", security_level=4, provenance_json={},
            ))
        target = uow.targets.get_scoped.return_value
        target.db_type, target.version_code, target.environment, target.capabilities_json = "POSTGRESQL", "16", "TEST", {}
        await effects.evaluate_effect_plan(command=command, **kwargs)
        published = uow.inspections.publish_report.await_args.args[0]
        published.is_current = True
        f.uow.inspections.get_report_scoped.side_effect = lambda **kw: published
        f.uow.inspections.list_report_sources.side_effect = lambda **kw: uow.inspections.add_report_sources.await_args.args[0]
        f.uow.runs, f.uow.targets, f.uow.agents, f.uow.changes = uow.runs, uow.targets, uow.agents, uow.changes
        f.uow.agents.version_source_ids = AsyncMock(return_value=[])
        f.report, f.domain = published, run.domain_id
        result = await extract(f)
        assert result["version"]["profile"]["case_kind"] == "VERIFIED_OPTIMIZATION"
        assert result["version"]["profile"]["root_cause"]["effective_level"] == "INCONCLUSIVE"
        assert result["version"]["profile"]["verification"]["method"] == "MATCHED_CONTROL_RATIO.v1"
        assert result["version"]["status"] == "PROCESSING"
        assert result["asset"]["current_version_id"] is None
        assert f.service._kc.ingest_user_file.await_count == 1
    asyncio.run(run())
