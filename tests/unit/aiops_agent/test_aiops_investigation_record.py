"""统一调查记录、回答、报告与案例入口的闭环回归。"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from aiops_agent.application.diagnosis.investigation_record import (
    freeze_investigation, investigation_report_source, render_investigation,
    validate_diagnosis,
)
from aiops_agent.application.operations_knowledge import OperationsKnowledgeService
from aiops_agent.application.reporting import normalize_report_source
from aiops_agent.contracts.investigation import DiagnosisAnswerDraft, InvestigationRecord
from aiops_agent.contracts.diagnosis_state import ClaimRevision, InvestigationGoal
from aiops_agent.contracts.turn_answer import AIOpsTurnResult, TurnAnswerBlock, TurnEvidenceFact
from platform_core.contracts.aiops import AnswerBlockType, SufficiencyStatus
from platform_core.identity import uuid7


def evidence(**overrides):
    return TurnEvidenceFact(
        evidence_ref="artifact:source#observation", artifact_id=str(uuid7()),
        source_id="database", step_id="a1", tool_id="db.session.blocking_chain",
        measurement_semantics="CURRENT_ACTIVITY", presentation_kind="TABLE",
        captured_at="2026-10-28T03:00:00+00:00",
        columns=({"name": "blocked_session"}, {"name": "blocker_session"}),
        rows=((88, 12),), row_count=1,
    ).model_copy(update=overrides)


def draft(*, goal_satisfied=True, **claim_overrides):
    claim = {
        "claim_id": "c1", "kind": "FACT", "subject": "当前目标会话",
        "statement": "会话 12 正在阻塞会话 88。",
        "level": "OBSERVED", "applicable_time": "2026-10-28T03:00:00Z 采样时点",
        "evidence_fields": ({"evidence_ref": "artifact:source#observation", "row_index": 0, "field": "blocker_session"},),
        "reasoning": "数据库当前阻塞链直接返回这项会话关系。",
        **claim_overrides,
    }
    return DiagnosisAnswerDraft(
        claims=(claim,), recommendations=({
            "recommendation_id": "r1", "conclusion_ids": ("c1",),
            "action": "核对阻塞者事务与业务归属，再评估是否需缓解。",
            "intent": "VERIFY", "preconditions": ("确认会话身份与当前事务仍相同。",),
            "risks": ("直接终止可能触发长时间回滚。",),
            "validation": ("重新采样同一阻塞关系与业务响应。",),
            "rollback_or_stop": "事务已提交或身份变化则停止。",
            "requires_approval": False,
        },), goal_satisfied=goal_satisfied,
    )


def context():
    return SimpleNamespace(
        run_id=uuid7(), task_id=uuid7(), target_id=uuid7(),
        plan_snapshot={"answer_context": {"question": "当前谁阻塞了业务？"}},
    )


def record():
    return freeze_investigation(draft=draft(), evidence=(evidence(),), context=context())


def test_record_is_strict_frozen_and_preserves_source_trust():
    value = record()
    assert value.revision == 1 and value.previous_revision is None
    with pytest.raises(ValidationError):
        value.target_id = "another"
    with pytest.raises(ValidationError):
        value.recommendations[0].requires_approval = True
    with pytest.raises(ValidationError):
        InvestigationRecord.model_validate({**value.model_dump(), "unknown": {}})
    with pytest.raises(ValueError):
        validate_diagnosis(draft(), (evidence(trust_level="USER_PROVIDED"),))
    with pytest.raises(ValueError):
        validate_diagnosis(draft(), (evidence(trust_level="MODEL_INFERENCE"),))


@pytest.mark.parametrize("field,row", [("absent", 0), ("blocker_session", 1)])
def test_nonexistent_row_and_field_are_rejected(field, row):
    claim = draft().claims[0].model_copy(update={
        "evidence_fields": (draft().claims[0].evidence_fields[0].model_copy(update={"field": field, "row_index": row}),),
    })
    with pytest.raises(ValueError):
        validate_diagnosis(draft().model_copy(update={"claims": (claim,)}), (evidence(),))


def test_causes_cannot_be_promoted_by_model_direct_label():
    with pytest.raises(ValueError):
        draft(kind="CAUSE")
    supported = draft(kind="CAUSE", level="SUPPORTED", limitations=("仅说明当前采样，尚未匹配业务故障窗口。",))
    value = freeze_investigation(draft=supported, evidence=(evidence(),), context=context())
    assert investigation_report_source(value)["root_cause"]["effective_level"] == "PROBABLE"


def test_record_report_projection_preserves_answers_actions_and_references():
    fact = evidence()
    value = freeze_investigation(draft=draft(), evidence=(fact,), context=context())
    analysis, solution, refs = render_investigation(value)
    result = AIOpsTurnResult(
        status="COMPLETED", sufficiency_status=SufficiencyStatus.ANSWERABLE,
        blocks=(
            TurnAnswerBlock(block_type=AnswerBlockType.ANALYSIS_MARKDOWN,
                schema_version="AIOPS_ANALYSIS_BLOCK.v1", payload={"markdown": analysis}, evidence_refs=refs),
            TurnAnswerBlock(block_type=AnswerBlockType.SOLUTION_MARKDOWN,
                schema_version="AIOPS_SOLUTION_BLOCK.v1", payload={"markdown": solution}, evidence_refs=refs),
        ), evidence=(fact,), investigation_record=value,
    )
    source = normalize_report_source(schema_version=result.schema_version, payload=result.model_dump(mode="json"), source_kind="CHAT")
    assert source["diagnosis_rationale"] == analysis
    assert source["solution"]["long_term_remediations"] == (solution,)
    assert source["facts"][0]["claim_id"] == "c1"
    assert source["investigation_record"]["record_id"] == value.record_id
    assert source["root_cause"]["effective_level"] == "INCONCLUSIVE"
    assert source["evidence_refs"] == refs


def test_unstructured_turn_keeps_analysis_and_solution_without_fabricating_claims():
    source = normalize_report_source(schema_version="AIOPS_TURN_RESULT.v1", source_kind="CHAT", payload={
        "status": "COMPLETED", "blocks": [
            {"block_type": "ANALYSIS_MARKDOWN", "payload": {"markdown": "原分析"}},
            {"block_type": "SOLUTION_MARKDOWN", "payload": {"markdown": "原方案"}},
        ],
    })
    assert source["diagnosis_rationale"] == "原分析"
    assert source["solution"]["long_term_remediations"] == ("原方案",)
    assert not source["facts"]
    missing = normalize_report_source(schema_version="AIOPS_TURN_RESULT.v1", source_kind="CHAT", payload={"status": "COMPLETED", "blocks": []})
    assert missing["status"] == "PARTIAL"
    assert missing["gaps"][0]["code"] == "REPORT_BODY_MISSING"


def test_manual_closure_cannot_confirm_root_cause():
    source = normalize_report_source(schema_version="DIAGNOSIS_CLOSURE.v1", source_kind="CHAT", payload={
        "profile": {"root_cause": {"summary": "人工认为已解决"}, "verification": {"result": "RESOLVED"}},
    })
    assert source["root_cause"]["effective_level"] == "INCONCLUSIVE"


@pytest.mark.parametrize("mode", ["valid", "target_mismatch", "run_mismatch", "latest_refuted", "latest_unstructured", "latest_hypothesis"])
def test_new_sql_turn_can_form_reference_candidate_without_formal_report(mode):
    async def run():
        fact = evidence()
        value = freeze_investigation(draft=draft(), evidence=(fact,), context=context())
        actor, domain, conversation_id = uuid7(), uuid7(), uuid7()
        turn_id, run_id, artifact_id = uuid7(), value.run_id, uuid7()
        target_id = value.target_id
        if mode == "target_mismatch":
            value = value.model_copy(update={"target_id": str(uuid7())})
        elif mode == "run_mismatch":
            value = value.model_copy(update={"run_id": str(uuid7())})
        elif mode == "latest_refuted":
            value = value.model_copy(update={
                "claims": (value.claims[0].model_copy(update={"revision_state": "REFUTED"}),),
                "recommendations": (),
                "goal_assessments": (),
                "stop_reason": "PARTIAL_ANSWER",
            })
        elif mode == "latest_hypothesis":
            value = freeze_investigation(
                draft=draft(level="HYPOTHESIS", goal_satisfied=False),
                evidence=(fact,), context=context(),
            ).model_copy(update={"target_id": target_id, "run_id": run_id})
        now = datetime.now(timezone.utc)
        artifact = SimpleNamespace(
            artifact_id=artifact_id, schema_version="AIOPS_TURN_RESULT.v1",
            payload_json={"status": "COMPLETED", "investigation_record": value.model_dump(mode="json"),
                "evidence": [fact.model_dump(mode="json")]},
            ops_task_id=uuid7(), artifact_key="answer", created_at=now,
        )
        if mode == "latest_unstructured":
            artifact.payload_json.pop("investigation_record")
        uow = SimpleNamespace(
            conversations=SimpleNamespace(get_conversation=AsyncMock(return_value=SimpleNamespace(created_by=actor, title="SQL 深入诊断"))),
            turns=SimpleNamespace(list_all_turns=AsyncMock(return_value=[
                SimpleNamespace(turn_id=uuid7(), status="COMPLETED"),
                SimpleNamespace(turn_id=turn_id, status="COMPLETED"),
            ]),
                get_run_link=AsyncMock(return_value=SimpleNamespace(ops_run_id=run_id))),
            runs=SimpleNamespace(
                get_run_scoped=AsyncMock(return_value=SimpleNamespace(status="COMPLETED", final_artifact_id=artifact_id, target_id=target_id, ops_run_id=run_id)),
                get_artifact=AsyncMock(return_value=artifact), list_artifacts=AsyncMock(return_value=[artifact]),
                get_artifact_by_key=AsyncMock(return_value=None)),
            targets=SimpleNamespace(get_scoped=AsyncMock(return_value=SimpleNamespace(db_type="POSTGRESQL", version_code="16", environment="TEST", capabilities_json={}))),
        )

        class Uow:
            async def __aenter__(self):
                return uow

            async def __aexit__(self, *args):
                pass

        service = OperationsKnowledgeService.__new__(OperationsKnowledgeService)
        service._uow_factory = Uow
        preview = await service._conversation_case_context(domain_id=domain, conversation_id=conversation_id, actor_id=actor)
        uow.turns.get_run_link.assert_awaited_once()
        if mode != "valid":
            assert preview["eligible"] is False
            assert preview["reasons"][0]["code"] == "INVESTIGATION_RECORD_MISSING"
            return
        assert preview["eligible"] is True
        assert preview["root_cause_grade"] == "INCONCLUSIVE"
        assert preview["included_turn_ids"] == [str(turn_id)]
        assert preview["_context"]["investigation_record"]["record_id"] == value.record_id

    asyncio.run(run())

def test_subgoals_cannot_be_hidden_by_global_completed_flag():
    ctx = context()
    ctx.plan_snapshot["answer_context"]["task_frame"] = {"success_criteria": ["列出阻塞链", "确认业务根因"]}
    value = freeze_investigation(draft=draft(), evidence=(evidence(),), context=ctx)
    assert value.stop_reason == "PARTIAL_ANSWER"
    assert {goal.goal_id for goal in value.goal_assessments} == {"g1", "g2"}
    partial = draft(goal_satisfied=False).model_copy(update={"goal_assessments": (
        InvestigationGoal(goal_id="g1", status="SATISFIED", claim_refs=("c1",)),
    )})
    with pytest.raises(ValueError):
        freeze_investigation(draft=DiagnosisAnswerDraft.model_validate(partial.model_dump()), evidence=(evidence(),), context=ctx)


def revision_fixture():
    from aiops_agent.contracts.investigation import InvestigationParent
    fact = evidence()
    ctx = context()
    old = freeze_investigation(draft=draft(), evidence=(fact,), context=ctx)
    parent_artifact = SimpleNamespace(artifact_id=uuid7(), content_hash="a" * 64)
    parent = InvestigationParent(
        record_id=old.record_id, revision=old.revision, target_id=old.target_id,
        run_id=old.run_id, artifact_id=str(parent_artifact.artifact_id),
        content_hash=parent_artifact.content_hash, claims=old.claims,
    )
    next_ctx = context()
    next_ctx.target_id = ctx.target_id
    next_ctx.plan_snapshot["answer_context"]["parent_investigation"] = parent.model_dump(mode="json")
    current = draft(statement="旧阻塞关系的会话映射有误，需按已补充身份重新判定。").model_copy(update={
        "continues_investigation": True,
        "claim_revisions": (ClaimRevision(prior_claim_id="c1", current_claim_id="c1", relation="REFUTES", reason="同一采样的会话身份重新核验。"),),
    })
    current = DiagnosisAnswerDraft.model_validate(current.model_dump())
    return fact, ctx, old, parent_artifact, next_ctx, current



def test_third_revision_can_refute_an_older_claim_not_repeated_in_middle_turn():
    from aiops_agent.application.diagnosis.revisions import active_historical_claims, merge_investigation_sources
    from aiops_agent.application.diagnosis.investigation_record import investigation_report_source
    from aiops_agent.contracts.investigation import InvestigationParent
    from aiops_agent.contracts.diagnosis_state import ClaimRevision
    fact, ctx, old, old_artifact, middle_ctx, current = revision_fixture()
    middle_draft = current.model_copy(update={"claim_revisions": ()})
    middle = freeze_investigation(draft=middle_draft, evidence=(fact,), context=middle_ctx)
    middle_artifact = SimpleNamespace(artifact_id=uuid7(), content_hash="middle-hash")
    parent = InvestigationParent(
        record_id=middle.record_id, revision=middle.revision, target_id=middle.target_id, run_id=middle.run_id,
        artifact_id=str(middle_artifact.artifact_id), content_hash=middle_artifact.content_hash,
        claims=middle.claims, historical_claims=active_historical_claims(middle),
    )
    last_ctx = context()
    last_ctx.target_id = middle_ctx.target_id
    last_ctx.run_id = uuid7()
    last_ctx.plan_snapshot["answer_context"]["parent_investigation"] = parent.model_dump(mode="json")
    change = ClaimRevision(prior_artifact_id=str(old_artifact.artifact_id), prior_claim_id="c1", current_claim_id="c1",
        relation="REFUTES", reason="原判断在第三轮才取得直接反证")
    last_draft = current.model_copy(update={"claim_revisions": (change,)})
    last = freeze_investigation(draft=last_draft, evidence=(fact,), context=last_ctx)
    last_artifact = SimpleNamespace(artifact_id=uuid7(), content_hash="last-hash")
    rows = [(SimpleNamespace(ops_run_id=value.run_id, target_id=value.target_id), artifact, investigation_report_source(value))
        for value, artifact in ((old, old_artifact), (middle, middle_artifact), (last, last_artifact))]
    projected, history = merge_investigation_sources(rows)
    assert projected[str(old_artifact.artifact_id)]["facts"] == ()
    assert projected[str(middle_artifact.artifact_id)]["facts"]
    assert history[0]["previous_artifact_id"] == str(old_artifact.artifact_id)
    assert not any(item.artifact_id == str(old_artifact.artifact_id) for item in active_historical_claims(last))


def test_revision_is_explicit_continuous_and_same_subject_and_phase():
    fact, ctx, old, artifact, next_ctx, current = revision_fixture()
    new = freeze_investigation(draft=current, evidence=(fact,), context=next_ctx)
    assert new.record_id == old.record_id and new.revision == 2
    assert new.previous_revision == str(artifact.artifact_id)
    for key, value in (("subject", "另一会话"), ("applicable_time", "下一次采样")):
        changed = current.model_copy(update={"claims": (current.claims[0].model_copy(update={key: value}),)})
        with pytest.raises(ValueError):
            freeze_investigation(draft=changed, evidence=(fact,), context=next_ctx)
    with pytest.raises(ValueError):
        freeze_investigation(draft=current, evidence=(fact,), context=ctx)
    independent = freeze_investigation(draft=draft(), evidence=(fact,), context=next_ctx)
    assert independent.revision == 1 and independent.record_id != old.record_id


def test_report_current_view_retires_refuted_claim_and_action_but_keeps_lineage():
    from aiops_agent.application.diagnosis.revisions import merge_investigation_sources
    fact, ctx, old, old_artifact, next_ctx, current = revision_fixture()
    new = freeze_investigation(draft=current, evidence=(fact,), context=next_ctx)
    new_artifact = SimpleNamespace(artifact_id=uuid7(), content_hash="b" * 64)
    rows = [
        (SimpleNamespace(ops_run_id=old.run_id, target_id=old.target_id), old_artifact, investigation_report_source(old)),
        (SimpleNamespace(ops_run_id=new.run_id, target_id=new.target_id), new_artifact, investigation_report_source(new)),
    ]
    sources, history = merge_investigation_sources(rows)
    assert not sources[str(old_artifact.artifact_id)]["facts"]
    assert not sources[str(old_artifact.artifact_id)]["solution"]["long_term_remediations"]
    assert sources[str(new_artifact.artifact_id)]["facts"][0]["summary"] == new.claims[0].statement
    assert history[0]["relation"] == "REFUTES"
    assert sources[str(old_artifact.artifact_id)]["investigation_record"] == old.model_dump(mode="json")
    old_artifact.content_hash = "wrong"
    with pytest.raises(ValueError):
        merge_investigation_sources(rows)


@pytest.mark.parametrize("failure", [None, "actor", "target_version", "source", "artifact_hash", "source_changed", "unbound", "record_target"])
def test_parent_loading_rechecks_authorization_and_immutable_provenance(failure):
    from aiops_agent.application.investigation.prior_investigation import load_parent_investigation
    from aiops_agent.playbooks import canonical_hash
    async def run_test():
        fact = evidence()
        ctx = context()
        old = freeze_investigation(draft=draft(), evidence=(fact,), context=ctx)
        tenant, domain, agent, version, actor, conversation = (uuid7() for _ in range(6))
        previous = SimpleNamespace(tenant_id=tenant, domain_id=domain, conversation_id=conversation, created_by=actor, status="COMPLETED", turn_id=uuid7())
        turn = SimpleNamespace(**{key: getattr(previous, key) for key in ("tenant_id", "domain_id", "conversation_id", "created_by")}, turn_no=2)
        shared = dict(tenant_id=tenant, domain_id=domain, agent_id=agent, agent_version_id=version, target_id=ctx.target_id, actor_id=actor)
        target = SimpleNamespace(row_version=2, db_type="POSTGRESQL")
        snapshot = {"target": {"row_version": 2, "db_type": "POSTGRESQL"}, "capability_snapshot": {"source_snapshots": []}}
        prior_run = SimpleNamespace(**shared, ops_run_id=old.run_id, plan_snapshot_json=snapshot, final_artifact_id=uuid7(), completed_at=datetime.now(timezone.utc))
        run = SimpleNamespace(**shared)
        if failure == "actor":
            turn.created_by = uuid7()
        elif failure == "target_version":
            target.row_version = 3
        elif failure == "source":
            snapshot["capability_snapshot"]["source_snapshots"] = [{"source_id": "revoked"}]
        elif failure == "record_target":
            old = old.model_copy(update={"target_id": "another"})
        payload = {"investigation_record": old.model_dump(mode="json"), "evidence": [fact.model_dump(mode="json")]}
        artifact = SimpleNamespace(artifact_id=prior_run.final_artifact_id, ops_run_id=prior_run.ops_run_id,
            payload_json=payload, content_hash=canonical_hash(payload), schema_version="AIOPS_TURN_RESULT.v1")
        if failure == "artifact_hash":
            artifact.content_hash = "wrong"
        uow = SimpleNamespace(
            turns=SimpleNamespace(get_previous_turn=AsyncMock(return_value=previous), get_run_link=AsyncMock(return_value=SimpleNamespace(ops_run_id=old.run_id))),
            runs=SimpleNamespace(get_run=AsyncMock(return_value=prior_run), get_artifact=AsyncMock(return_value=artifact)),
            agents=SimpleNamespace(version_has_target=AsyncMock(return_value=failure != "unbound")),
        )
        result = await load_parent_investigation(
            uow=uow, turn=turn, run=run, target=target,
            capabilities=SimpleNamespace(model_dump=lambda **kw: {"source_snapshots": []}),
            sources=[SimpleNamespace(updated_at=datetime.now(timezone.utc))] if failure == "source_changed" else [],
        )
        assert (result is not None) == (failure is None)
    asyncio.run(run_test())

@pytest.mark.parametrize("database_type,executor", [("ORACLE", "SQLPLUS"), ("POSTGRESQL", "PSQL"), ("MYSQL", "MYSQL")])
def test_three_database_remediation_runbook_is_environment_scoped_and_never_execution_authority(database_type, executor):
    ctx = context()
    ctx.plan_snapshot["capability_snapshot"] = {"database_type": database_type, "database_version": "待现场核验"}
    data = draft().model_dump()
    data["recommendations"][0].update({
        "intent": "REPAIR", "requires_approval": True, "draft_sql": "SELECT 1",
        "missing_parameters": ["业务窗口"],
        "parameters": [{"name": "blocker", "value": "12", "source": "OBSERVATION",
            "evidence_field": {"evidence_ref": "artifact:source#observation", "row_index": 0, "field": "blocker_session"}}],
    })
    value = freeze_investigation(draft=DiagnosisAnswerDraft.model_validate(data), evidence=(evidence(),), context=ctx)
    assert value.runbook.database_type == database_type
    assert value.runbook.status == "BLOCKED_BY_REQUIRED_INPUTS"
    command = value.runbook.phases[0].steps[0].commands[0]
    assert command.executor == executor and "待审批" in command.run_as
    assert command.node_scope == (str(ctx.target_id),)
    data["recommendations"][0]["parameters"][0]["value"] = "999"
    with pytest.raises(ValueError):
        freeze_investigation(draft=DiagnosisAnswerDraft.model_validate(data), evidence=(evidence(),), context=ctx)
