"""健康检查空结果、完成状态与DBA表达回归，不调用真实数据库或模型。"""

import asyncio
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.diagnosis.investigation_record import freeze_investigation, validate_diagnosis
from aiops_agent.application.diagnosis.model_evidence import evidence_model_payload
from aiops_agent.contracts.investigation import DiagnosisAnswerDraft, DiagnosisStreamResponse, InvestigationRecord
from aiops_agent.contracts.sql_diagnosis import EvidenceFieldReference
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment, TurnEvidenceGap
from aiops_agent.ports.model import ModelTextDelta
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
from tests.unit.aiops_agent.test_aiops_health_diagnosis_delivery import raw_fact
from tests.unit.aiops_agent.test_aiops_turn_answer import _AnswerModel, _TestPrompts, _context, _tool_artifact


def empty_draft(fact, *, field="$result.row_count", row_index=0):
    return DiagnosisAnswerDraft(
        claims=({
            "claim_id": "c1", "kind": "FACT", "subject": "本次检查范围",
            "statement": "本次查询未返回匹配记录。", "level": "OBSERVED",
            "applicable_time": fact.captured_at,
            "evidence_fields": ({"evidence_ref": fact.evidence_ref, "row_index": row_index, "field": field},),
            "reasoning": "只读查询已完整返回零行。",
        },),
        goal_satisfied=True,
    )


@pytest.mark.parametrize("field", ["$result.row_count", "$result.captured_at"])
def test_empty_result_is_citable_without_fabricating_a_data_row(field):
    _, fact = raw_fact("db.session.blocking_chain", empty=True)
    before = fact.model_dump_json()
    payload = evidence_model_payload(fact)
    assert payload["empty_result_reference"]["fields"][field] in (0, fact.captured_at)
    assert payload["rows"] == [] and payload["row_count"] == 0
    draft = empty_draft(fact, field=field)
    validate_diagnosis(draft, (fact,))
    context = _context(task_frame_overrides={"diagnostic_profile": "GENERAL"})
    record = freeze_investigation(draft=draft, evidence=(fact,), context=context)
    assert record.stop_reason == "GOAL_ANSWERED"
    assert record.observations[0].row_count == 0
    InvestigationRecord.model_validate(record.model_dump()).validate_observations((fact,))
    assert fact.model_dump_json() == before


@pytest.mark.parametrize("updates", [
    {"truncated": True},
    {"trust_level": "USER_PROVIDED"},
    {"trust_level": "MODEL_INFERENCE"},
    {"rows": (("actual", 1),), "row_count": 1},
    {"rows": (("inconsistent", 1),), "row_count": 0},
    {"rows": (), "row_count": 1},
    {"tool_id": "metric.query_range", "source_id": "monitoring.overview"},
    {"tool_id": "monitor.query_native_metric"},
])
def test_missing_or_incomplete_samples_are_not_successful_empty_results(updates):
    _, original = raw_fact("db.session.blocking_chain", empty=True)
    fact = original.model_copy(update=updates)
    assert "empty_result_reference" not in evidence_model_payload(fact)
    with pytest.raises(ValueError):
        validate_diagnosis(empty_draft(fact), (fact,))


@pytest.mark.parametrize("field,row_index", [
    ("$result.unknown", 0), ("$result.row_count", 1), ("SQL_ID", 0),
])
def test_empty_result_does_not_authorize_fake_fields_or_rows(field, row_index):
    _, fact = raw_fact("db.session.blocking_chain", empty=True)
    with pytest.raises(ValueError):
        validate_diagnosis(empty_draft(fact, field=field, row_index=row_index), (fact,))


def test_failed_query_keeps_its_gap_and_has_no_successful_empty_receipt():
    _, fact = raw_fact("db.session.blocking_chain", empty=True)
    gap = TurnEvidenceGap(source_id=fact.source_id, step_id=fact.step_id,
        evidence_ref=fact.evidence_ref, code="QUERY_FAILED", detail="只读查询未完成。")
    assessment = DbaSufficiencyAssessment(status="PARTIAL", evidence=(fact,), gaps=(gap,))
    payload = DbaAnswerComposeHandler._model_assessment_payload(assessment)
    assert "empty_result_reference" not in payload["evidence"][0]
    with pytest.raises(ValueError, match="查询缺口"):
        freeze_investigation(draft=empty_draft(fact), evidence=(fact,),
            context=_context(task_frame_overrides={"diagnostic_profile": "GENERAL"}), gaps=(gap,))


@pytest.mark.parametrize("missing_metric", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_health_check_without_problem_rows_completes_unless_there_is_a_real_gap(missing_metric, streaming):
    async def exercise():
        tools = ("db.instance.identity", "db.session.active", "db.session.blocking_chain",
            "db.storage.capacity", "db.transaction.long_running", "db.replication.status",
            "db.sql.problem_candidates")
        empty_tools = {"db.session.blocking_chain", "db.transaction.long_running", "db.sql.problem_candidates"}
        artifacts = tuple(_tool_artifact(semantics="CURRENT_ACTIVITY", tool_id=tool,
            row_count=0 if tool in empty_tools else 1) for tool in tools)
        context = _context(artifacts=artifacts, task_frame_overrides={
            "diagnostic_profile": "GENERAL", "success_criteria": ["完成本次健康检查并给出判断"],
            "subject_ref": {"conversation_starter": "database.health.overview"},
        })
        context = replace(context, plan_snapshot={**context.plan_snapshot,
            "planning_route": {"mode": "CONVERSATION_STARTER"},
            "monitoring": {"initial_gaps": [{"code": "SOURCE_NO_DATA", "detail": "CPU指标没有有效采样"}]} if missing_metric else {},
        })
        assessment = await DbaEvidenceAssessmentHandler(model_client=None, prompts=None).execute(context)
        assert assessment.status == ("PARTIAL" if missing_metric else "ANSWERABLE")
        assert sum(fact.row_count == 0 for fact in assessment.evidence) == 3
        markdown = "本次检查未发现阻塞、长事务或匹配的问题SQL，无需立即处置。"

        async def compose(**kwargs):
            payload = kwargs["input_payload"]
            assert payload["diagnostic_completion_policy"]["basis"] == "FROZEN_TASK_SCOPE"
            assert "仅在用户目标要求" in payload["diagnostic_completion_policy"]["historical_baseline"]
            assert "历史基线" not in "".join(gap["detail"] for gap in payload["sufficiency"]["gaps"])
            facts = [fact for fact in payload["sufficiency"]["evidence"] if fact["row_count"] == 0]
            assert len(facts) == 3 and all(fact["rows"] == [] for fact in facts)
            claims = tuple({
                "claim_id": f"c{index}", "kind": "FACT", "subject": fact["tool_id"],
                "statement": "本次检查未返回匹配记录。", "level": "OBSERVED",
                "applicable_time": fact["captured_at"],
                "evidence_fields": ({"evidence_ref": fact["evidence_ref"], "row_index": 0, "field": "$result.row_count"},),
                "reasoning": "数据库只读查询完整返回零行。",
            } for index, fact in enumerate(facts, 1))
            output = DiagnosisAnswerDraft(claims=claims, goal_satisfied=True,
                goal_assessments=({"goal_id": "g1", "status": "SATISFIED",
                    "claim_refs": tuple(claim["claim_id"] for claim in claims)},))
            if streaming:
                assert kwargs["output_model"] is DiagnosisStreamResponse
                await kwargs["on_text_delta"](ModelTextDelta(delta=markdown))
                output = DiagnosisStreamResponse(**output.model_dump(), answer_markdown=markdown)
            baseline = await _AnswerModel(evidence_refs=()).generate_structured(**kwargs)
            return SimpleNamespace(output=output, receipt=baseline.receipt)

        model = SimpleNamespace(generate_structured=AsyncMock(side_effect=compose))
        final = replace(context, input_artifacts=({
            "artifact_id": "assessment", "schema_version": "DBA_SUFFICIENCY.v1",
            "payload": assessment.model_dump(mode="json"),
        },))
        deltas = []
        async def on_delta(value):
            deltas.append(value)
        result = await DbaAnswerComposeHandler(model_client=model, prompts=_TestPrompts()).execute(
            final, **({"on_text_delta": on_delta} if streaming else {}))
        assert result.status == ("PARTIAL" if missing_metric else "COMPLETED")
        assert result.investigation_record.stop_reason == "GOAL_ANSWERED"
        assert len(result.evidence_gaps) == int(missing_metric)
        assert all(not fact.rows for fact in result.evidence if fact.tool_id in empty_tools)
        if streaming:
            assert result.answer_streamed
            assert all(not delta.reset for delta in deltas)
            assert "".join(delta.delta for delta in deltas) == markdown
            assert next(block.payload["markdown"] for block in result.blocks
                if block.payload.get("raw_model_output")) == markdown
    asyncio.run(exercise())


def test_frozen_result_reference_is_revalidated_when_evidence_is_replayed():
    _, fact = raw_fact("db.session.blocking_chain", empty=True)
    record = freeze_investigation(draft=empty_draft(fact), evidence=(fact,),
        context=_context(task_frame_overrides={"diagnostic_profile": "GENERAL"}))
    payload = record.model_dump()
    payload["claims"][0]["evidence_fields"][0]["field"] = "$result.missing"
    with pytest.raises(ValueError, match="空结果引用"):
        InvestigationRecord.model_validate(payload).validate_observations((fact,))


def test_result_metadata_is_not_a_numeric_business_measurement():
    _, fact = raw_fact("db.session.blocking_chain", empty=True)
    ref = EvidenceFieldReference(evidence_ref=fact.evidence_ref, row_index=0, field="$result.row_count")
    assert ref.observed_value(fact) == 0
    assert "SQL_ID" not in evidence_model_payload(fact)["empty_result_reference"]["fields"]


def test_health_prompts_use_dba_judgment_without_inventing_completion_requirements():
    root = Path(__file__).resolve().parents[3]
    catalog = tomllib.loads((root / "packages/platform_core/src/platform_core/resources/prompts.toml").read_text())
    prompts = {item["prompt_key"]: item for item in catalog["prompts"]}
    for name in ("answer_compose", "answer_stream", "diagnosis_answer_compose", "investigation_assessor"):
        prompt = prompts[f"aiops_agent.{name}"]
        assert prompt["version"] == "1.0.0" and prompt["active"] is True
        assert "历史基线" in prompt["content"]
        assert "零行" in prompt["content"]
    diagnosis = prompts["aiops_agent.diagnosis_answer_compose"]["content"]
    assert "正式DBA值班记录" in diagnosis
    assert "diagnostic_completion_policy" in diagnosis
    assert "$result.row_count" in diagnosis
    assert "无异常检查可以满足goal_catalog" in diagnosis
