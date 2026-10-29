"""SQL 阶段诊断复用、关键补证、失败保留及来源边界；不替代真实模型验收。"""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from aiops_agent.adapters.model_serving import AIOpsModelError
from aiops_agent.application.diagnosis.deadline_result import deadline_result
from aiops_agent.application.diagnosis.evidence_projection import database_evidence_fact
from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle
from aiops_agent.application.diagnosis.sql_review import evidence_hash, reusable_checkpoint, review_context, scope_hash
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.sql_diagnosis import SqlDiagnosisDraft, SqlDiagnosisReview, SqlDiagnosticCheckpoint
from aiops_agent.contracts.tool_execution import DbaToolResult
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment, TurnEvidenceGap
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_sql_diagnosis import diagnosis
from tests.unit.aiops_agent.test_aiops_sql_plan_import import SUBJECTS, anchor
from tests.unit.aiops_agent.test_aiops_sql_plan_priority import frame
from tests.unit.aiops_agent.test_aiops_turn_answer import _context, _tool_artifact


def review(fact, *, needs_more=False):
    original = diagnosis(fact)
    return SqlDiagnosisReview.model_validate(original.model_copy(update={
        "summary": "已检查范围未见明显异常，暂不变更。", "verdict": "NO_OBVIOUS_ISSUE",
        "goal_satisfied": not needs_more,
        "next_evidence": ("核对该 SQL 会话的锁等待与监控时间关联，以区分等待与调用开销。",) if needs_more else (),
        "conclusions": (original.conclusions[0].model_copy(update={
            "statement": "已取得匹配主体计数，当前已检查范围未见明显异常。", "level": "OBSERVED",
        }),),
        "recommendations": (original.recommendations[0].model_copy(update={
            "intent": "OBSERVE", "action": "暂不变更，继续按相同业务口径观测。", "requires_approval": False,
            "preconditions": (), "risks": (), "rollback_or_stop": "",
        }),),
    }).model_dump(mode="python"))


def setup(engine="POSTGRESQL", *, needs_more=False):
    fact = anchor(engine)
    task = frame(engine, SUBJECTS[engine])
    context = replace(_context(task_frame_overrides=task), target_id="target")
    context.plan_snapshot["capability_snapshot"] = {"database_type": engine}
    task = context.plan_snapshot["answer_context"]["task_frame"]
    model = SimpleNamespace(generate_structured=AsyncMock(return_value=SimpleNamespace(output=review(fact, needs_more=needs_more), receipt=None)))
    prompts = SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace(content="按观测诊断", ref=lambda: {})))
    gap = TurnEvidenceGap(source_id="window", step_id="window", code="WINDOW_NOT_FULLY_COVERED", detail="未覆盖历史窗口", retryable=True)
    assessment = DbaSufficiencyAssessment(status="PARTIAL", evidence=(fact,), gaps=(gap,))
    return fact, task, context, model, prompts, assessment


@pytest.mark.parametrize("engine", ["ORACLE", "POSTGRESQL", "MYSQL"])
def test_review_and_both_answers_share_one_diagnosis_despite_unrelated_window_gap(engine):
    async def exercise():
        fact, task, context, model, prompts, assessment = setup(engine)
        handler = DbaEvidenceAssessmentHandler(model_client=model, prompts=prompts)
        assessed = await handler._review_sql(context, assessment, task)
        assert assessed.status == "ANSWERABLE"
        assert assessed.investigation.next_action == "ANSWER"
        assert assessed.sql_checkpoint.draft.verdict == "NO_OBVIOUS_ISSUE"
        assert not AIOpsRuntimeService._should_replan_investigation(assessment=assessed, task_frame=task,
            deterministic_replan=True, no_progress_count=0, has_deferred=True)
        answer_context = replace(context, input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": assessed.model_dump(mode="json")},))
        composer = DbaAnswerComposeHandler(model_client=model, prompts=prompts)
        result = await composer.execute(answer_context)
        streamed = [item async for item in composer.execute_stream(answer_context)]
        assert result.status == streamed[-1].status == "COMPLETED"
        assert result.blocks == streamed[-1].blocks
        assert "暂不变更" in result.blocks[0].payload["markdown"]
        solution = next(block.payload["markdown"] for block in result.blocks if block.block_type == "SOLUTION_MARKDOWN")
        assert "停止或回退" not in solution and "风险：" not in solution
        model.generate_structured.assert_awaited_once()
        call = model.generate_structured.await_args.kwargs
        assert call["purpose"] == "aiops.sql-diagnosis" and call["output_model"] is SqlDiagnosisReview
        assert "resource_capabilities" not in call["input_payload"]
        assert call["input_payload"]["evidence_bundle"]["observations"][0]["records"]
    asyncio.run(exercise())


def test_only_conclusion_changing_requests_replan_and_keep_all_source_options():
    async def exercise():
        _, task, context, model, prompts, assessment = setup(needs_more=True)
        assessed = await DbaEvidenceAssessmentHandler(model_client=model, prompts=prompts)._review_sql(context, assessment, task)
        assert assessed.status == "PARTIAL" and assessed.investigation.next_action == "REPLAN"
        assert any(gap.code == "SQL_DIAGNOSIS_KEY_EVIDENCE" for gap in assessed.gaps)
        assert AIOpsRuntimeService._should_replan_investigation(assessment=assessed, task_frame=task,
            deterministic_replan=False, no_progress_count=0)
        assert "数据库、监控、日志" in model.generate_structured.await_args.kwargs["input_payload"]["delivery_requirements"]["next_evidence"]
    asyncio.run(exercise())


@pytest.mark.parametrize("engine", ["ORACLE", "POSTGRESQL", "MYSQL"])
def test_full_assessment_invokes_sql_review_instead_of_generic_assessment(engine):
    async def exercise():
        entry, task, context, model, prompts, _ = setup(engine)
        context = replace(context, target_id=str(uuid7()))
        raw = _tool_artifact(semantics="CUMULATIVE_SINCE_LOAD", tool_id=entry.tool_id)
        data = raw["payload"]["tool_outcomes"][0]["observation"]
        data.update(db_type=engine, target_id=context.target_id, columns=[
            {**column, "logical_type": "STRING", "sensitivity": "PUBLIC"} for column in entry.columns
        ], rows=[list(entry.rows[0])], row_count=1)
        source = DbaToolResult.model_validate(raw["payload"])
        projected = database_evidence_fact(result=source, outcome=source.tool_outcomes[0], artifact_id=raw["artifact_id"])
        model.generate_structured.return_value.output = review(projected)
        context = replace(context, input_artifacts=(raw,))
        assessed = await DbaEvidenceAssessmentHandler(model_client=model, prompts=prompts).execute(context)
        assert assessed.sql_checkpoint is not None and assessed.status == "ANSWERABLE"
        model.generate_structured.assert_awaited_once()
        assert model.generate_structured.await_args.kwargs["purpose"] == "aiops.sql-diagnosis"
    asyncio.run(exercise())


def test_unchanged_evidence_does_not_repeat_diagnosis_and_problem_measures_survive():
    async def exercise():
        fact, task, context, model, prompts, assessment = setup()
        model.generate_structured.return_value.output = diagnosis(fact)
        handler = DbaEvidenceAssessmentHandler(model_client=model, prompts=prompts)
        assessed = await handler._review_sql(context, assessment, task)
        context.plan_snapshot["answer_context"]["sql_checkpoint"] = assessed.sql_checkpoint.model_dump(mode="json")
        reused = await handler._review_sql(context, assessment, task)
        assert reused.sql_checkpoint.draft.verdict == "PROBLEM_FOUND"
        model.generate_structured.assert_awaited_once()
        answer_context = replace(context, input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": reused.model_dump(mode="json")},))
        result = await DbaAnswerComposeHandler(model_client=model, prompts=prompts).execute(answer_context)
        assert result.status == "COMPLETED"
        assert result.investigation_record.runbook is not None
        assert "调用合并" in next(block.payload["markdown"] for block in result.blocks if block.block_type == "SOLUTION_MARKDOWN")
        model.generate_structured.assert_awaited_once()
    asyncio.run(exercise())


def test_failed_later_diagnosis_keeps_checkpoint_without_retry_or_false_completion():
    async def exercise():
        fact, task, context, model, prompts, assessment = setup(needs_more=True)
        handler = DbaEvidenceAssessmentHandler(model_client=model, prompts=prompts)
        previous = await handler._review_sql(context, assessment, task)
        context.plan_snapshot["answer_context"]["sql_checkpoint"] = previous.sql_checkpoint.model_dump(mode="json")
        extra = fact.model_copy(update={"evidence_ref": fact.evidence_ref + "-next", "captured_at": "2026-10-29T01:00:00Z"})
        model.generate_structured.side_effect = AIOpsModelError("MODEL_DEADLINE_EXCEEDED")
        assessed = await handler._review_sql(context, assessment.model_copy(update={"evidence": (fact, extra)}), task)
        assert assessed.sql_checkpoint.model_dump(mode="json") == previous.sql_checkpoint.model_dump(mode="json")
        assert assessed.sql_diagnosis_error == "MODEL_DEADLINE_EXCEEDED"
        assert assessed.investigation.next_action == "ANSWER"
        answer_context = replace(context, input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": assessed.model_dump(mode="json")},))
        result = await DbaAnswerComposeHandler(model_client=model, prompts=prompts).execute(answer_context)
        assert result.status == "PARTIAL" and "上一阶段已核验判断" in result.blocks[0].payload["markdown"]
        assert "暂不变更" in result.blocks[0].payload["markdown"]
        assert model.generate_structured.await_count == 2
    asyncio.run(exercise())


@pytest.mark.parametrize("change", ["subject", "engine", "time", "action", "required", "evidence", "field"])
def test_checkpoint_cannot_cross_task_identity_or_use_missing_evidence(change):
    fact, task, _, _, _, _ = setup()
    checkpoint = SqlDiagnosticCheckpoint(draft=review(fact), scope_hash=scope_hash(task, "POSTGRESQL"), evidence_hash=evidence_hash((fact,)))
    engine, facts = "POSTGRESQL", (fact,)
    if change == "subject":
        task = {**task, "subject_ref": {"query_id": "99"}}
    elif change == "engine":
        engine = "MYSQL"
    elif change == "time":
        task = {**task, "time_scope": "另一时间窗口"}
    elif change == "action":
        task = {**task, "action_intent": "CHANGE"}
    elif change == "required":
        task = {**task, "completion_requirements": [{"requirement_id": "new"}]}
    elif change == "evidence":
        facts = (fact.model_copy(update={"rows": (tuple([*fact.rows[0][:-1], 999]),)}),)
    else:
        bad = checkpoint.draft.conclusions[0].evidence_fields[0].model_copy(update={"field": "imaginary"})
        checkpoint = checkpoint.model_copy(update={"draft": checkpoint.draft.model_copy(update={
            "conclusions": (checkpoint.draft.conclusions[0].model_copy(update={"evidence_fields": (bad,)}),),
        })})
    assert reusable_checkpoint(checkpoint, task_frame=task, database_type=engine, evidence=facts) is None


def test_updated_investigation_progress_is_not_a_different_task_scope():
    fact, task, *_ = setup()
    checkpoint = SqlDiagnosticCheckpoint(draft=review(fact), scope_hash=scope_hash(task, "POSTGRESQL"), evidence_hash=evidence_hash((fact,)))
    progressed = {**task, "known_facts": ["已核验 SQL 主体"], "unknowns": ["补证后的新疑问"]}
    assert reusable_checkpoint(checkpoint, task_frame=progressed, database_type="POSTGRESQL", evidence=(fact,)) is not None


def test_changes_still_need_risk_preconditions_and_rollback_but_observation_does_not():
    fact, *_ = setup()
    good = review(fact)
    for field, empty in (("preconditions", ()), ("risks", ()), ("rollback_or_stop", "")):
        change = diagnosis(fact).model_dump(mode="python")
        change["recommendations"][0][field] = empty
        with pytest.raises(ValidationError):
            SqlDiagnosisDraft.model_validate(change)
    bad = good.model_dump(mode="python")
    bad["recommendations"] = diagnosis(fact).model_dump(mode="python")["recommendations"]
    with pytest.raises(ValidationError):
        SqlDiagnosisReview.model_validate(bad)


@pytest.mark.parametrize("evidence_request", ["", "   ", "x" * 2001])
def test_invalid_next_evidence_is_rejected_before_gap_projection(evidence_request):
    fact, *_ = setup()
    draft = review(fact, needs_more=True).model_dump(mode="python")
    draft["next_evidence"] = [evidence_request]
    with pytest.raises(ValidationError):
        SqlDiagnosisReview.model_validate(draft)


def test_hypothesis_or_background_alone_cannot_claim_problem_found():
    fact, *_ = setup()
    for updates in ({"level": "HYPOTHESIS"}, {"scope": "CONTEXT"}):
        draft = diagnosis(fact, goal=False).model_dump(mode="python")
        draft["conclusions"][0].update(updates)
        with pytest.raises(ValidationError):
            SqlDiagnosisReview.model_validate(draft)


@pytest.mark.parametrize("invalid", [None, "unvalidated", "scope", "hash", "field"])
def test_deadline_preserves_only_authorized_source_validated_checkpoint(invalid):
    raw = _tool_artifact(semantics="CUMULATIVE_SINCE_LOAD")
    for column in raw["payload"]["tool_outcomes"][0]["observation"]["columns"]:
        column["name"] = column["name"].lower()
    source = DbaToolResult.model_validate(raw["payload"])
    raw_id = uuid7()
    fact = database_evidence_fact(result=source, outcome=source.tool_outcomes[0], artifact_id=raw_id)
    task = frame(subject={"subject_type": "ORACLE_SQL_ID", "sql_id": "abc123"})
    draft = diagnosis(fact, field="elapsed_seconds").model_copy(update={"summary": "已核验阶段判断应保留"})
    checkpoint = SqlDiagnosticCheckpoint(draft=draft, scope_hash=scope_hash(task, "ORACLE"), evidence_hash=evidence_hash((fact,)))
    if invalid == "scope":
        checkpoint = checkpoint.model_copy(update={"scope_hash": "0" * 64})
    elif invalid == "hash":
        checkpoint = checkpoint.model_copy(update={"evidence_hash": "0" * 64})
    elif invalid == "field":
        bad = draft.conclusions[0].evidence_fields[0].model_copy(update={"field": "imaginary"})
        checkpoint = checkpoint.model_copy(update={"draft": draft.model_copy(update={
            "conclusions": (draft.conclusions[0].model_copy(update={"evidence_fields": (bad,)}),),
        })})
    assessed = DbaSufficiencyAssessment(status="PARTIAL", evidence=(fact,), sql_checkpoint=checkpoint)
    assessed_id = uuid7()
    artifacts = [SimpleNamespace(artifact_id=raw_id, trust_level="SOURCE_VERIFIED", schema_version="DBA_TOOL_RESULT.v1", payload_json=source.model_dump(mode="json")),
        SimpleNamespace(artifact_id=assessed_id, trust_level="SOURCE_VERIFIED", schema_version="DBA_SUFFICIENCY.v1", payload_json=assessed.model_dump(mode="json"))]
    result = deadline_result(artifacts=artifacts, task_frame=task, database_type="ORACLE", projected_ids={str(raw_id), str(assessed_id)},
        validated_assessment_ids=frozenset() if invalid == "unvalidated" else frozenset({str(assessed_id)}))
    assert result.status == "PARTIAL" and result.blocks[0].payload["budget_exhausted"]
    assert ("已核验阶段判断应保留" in result.blocks[0].payload["markdown"]) == (invalid is None)


def test_review_projection_preserves_records_and_physical_identity_without_duplicate_graph():
    fact, task, *_ = setup()
    bundle = build_sql_evidence_bundle(task_frame=task, database_type="POSTGRESQL", evidence=(fact,))
    projected = review_context(bundle)
    assert "object_graph" not in projected
    assert projected["subjects"] == bundle["subjects"]
    assert projected["observations"][0]["records"] == bundle["observations"][0]["records"]
    assert projected["observations"][0]["lineage"]["provenance"]["physical_identity"]


def test_review_keeps_quality_window_and_reset_scope_in_compact_source():
    source = {"observations": [{"lineage": {"artifact_id": "a1", "provenance": {
        "content_hash": "large-hash", "quality_flags": ["PARTIAL"], "window_start": "t1", "window_end": "t2",
        "native_lineage": {"relation": "GV$SQL", "reset_scope": "CURSOR_LIFETIME", "fields": [{"large": "duplicate"}]},
    }}}]}
    compact = review_context(source)["observations"][0]["lineage"]
    assert compact["provenance"]["quality_flags"] == ["PARTIAL"]
    assert compact["provenance"]["window_start"] == "t1" and compact["provenance"]["window_end"] == "t2"
    assert compact["native_statistics_scope"] == {"relation": "GV$SQL", "reset_scope": "CURSOR_LIFETIME"}
    assert "native_lineage" not in compact["provenance"]


def test_mixed_observation_and_repair_does_not_create_empty_rollback_runbook_steps():
    async def exercise():
        fact, task, context, model, prompts, assessment = setup()
        original = diagnosis(fact)
        model.generate_structured.return_value.output = original.model_copy(update={"recommendations": (
            *original.recommendations, review(fact).recommendations[0].model_copy(update={"recommendation_id": "r2"}),
        )})
        assessed = await DbaEvidenceAssessmentHandler(model_client=model, prompts=prompts)._review_sql(context, assessment, task)
        answer_context = replace(context, input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": assessed.model_dump(mode="json")},))
        result = await DbaAnswerComposeHandler(model_client=model, prompts=prompts).execute(answer_context)
        assert result.status == "COMPLETED"
        steps = result.investigation_record.runbook.phases[0].steps
        assert len(steps) == 1 and steps[0].step_id == "diagnosis.r1"
    asyncio.run(exercise())
