"""健康诊断的大输入、阶段预算和超时事实交付回归，不依赖线上模型或数据库。"""

import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from aiops_agent.adapters.model_serving import AIOpsModelError, AIOpsStructuredModelClient
from aiops_agent.application.diagnosis.deadline_result import deadline_result
from aiops_agent.application.diagnosis.evidence_projection import database_evidence_fact
from aiops_agent.application.diagnosis.model_evidence import compact_native_values, evidence_model_payload
from aiops_agent.application.diagnosis.observed_result import observed_facts_markdown, observed_tables
from aiops_agent.config import AIOpsDiagnosisConfig
from aiops_agent.contracts.tool_execution import DbaToolResult
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment, TurnEvidenceGap
from aiops_agent.ports.model import ModelTextDelta
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
from platform_core.contracts.aiops import InvestigationAssessment
from tests.unit.aiops_agent.test_aiops_model_serving import _SequencedSession
from tests.unit.aiops_agent.test_aiops_native_answer_stream import Session, generate, stream_response
from tests.unit.aiops_agent.test_aiops_series_calculation import monitoring
from tests.unit.aiops_agent.test_aiops_turn_answer import _context, _tool_artifact, _TestPrompts
from tests.unit.aiops_agent.test_aiops_turn_answer import _AnswerModel


def assessment_output(**updates):
    return {
        "round_no": 1, "sufficiency_status": "ANSWERABLE", "next_action": "ANSWER",
        "progress_made": True, "reason": "本轮固定观测已返回。", **updates,
    }


@pytest.mark.parametrize("status,action", [("ANSWERABLE", "ANSWER"), ("PARTIAL", "REPLAN"), ("UNSAFE", "STOP_UNSAFE")])
def test_non_clarifying_assessment_does_not_require_a_null_placeholder(status, action):
    value = InvestigationAssessment.model_validate(assessment_output(sufficiency_status=status, next_action=action))
    assert value.clarification_question is None
    assert "clarification_question" not in InvestigationAssessment.model_json_schema()["required"]


@pytest.mark.parametrize("updates", [
    {"sufficiency_status": "NEEDS_CLARIFICATION", "next_action": "ASK_USER"},
    {"sufficiency_status": "NEEDS_CLARIFICATION", "next_action": "ASK_USER", "clarification_question": ""},
    {"sufficiency_status": "ANSWERABLE", "next_action": "ASK_USER", "clarification_question": "选择哪个范围？"},
    {"sufficiency_status": "NEEDS_CLARIFICATION", "next_action": "ANSWER", "clarification_question": "选择哪个范围？"},
])
def test_real_clarification_still_requires_question_and_matching_status(updates):
    with pytest.raises(ValidationError):
        InvestigationAssessment.model_validate(assessment_output(**updates))


def test_missing_optional_question_is_accepted_on_first_model_response():
    async def exercise():
        session = _SequencedSession((json.dumps(assessment_output(), ensure_ascii=False),))
        client = AIOpsStructuredModelClient(base_url="http://model-serving", audience="model-serving",
            caller_service="aiops-agent", timeout_seconds=30, session=session)
        prompt = "只评估已有证据。"
        with patch("aiops_agent.adapters.model_serving.build_internal_auth_headers", return_value={}):
            result = await client.generate_structured(
                purpose="aiops.investigation-assessment", output_model=InvestigationAssessment,
                model_snapshot={"model_id": "model", "tenant_id": "019c03b5-4b88-7ab2-8c19-7b6ea34f2a11", "technical_name": "test", "revision": "1"},
                prompt_ref={"content": prompt, "prompt_id": "test", "prompt_version": "1.0.0", "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()},
                input_payload={}, deadline=None, idempotency_key="first-assessment",
            )
        assert result.output.next_action == "ANSWER"
        assert len(session.payloads) == 1
    asyncio.run(exercise())


def test_assessment_sends_summaries_not_chart_or_sample_copies_and_keeps_raw_evidence():
    async def exercise():
        artifact = monitoring(tuple(range(2048)), days=1 / 288)
        context = _context(artifacts=(artifact,), task_frame_overrides={"diagnostic_profile": "GENERAL"})
        model = SimpleNamespace(generate_structured=AsyncMock(side_effect=AIOpsModelError("MODEL_DEADLINE_EXCEEDED")))
        before = datetime.now(UTC)
        result = await DbaEvidenceAssessmentHandler(model_client=model, prompts=_TestPrompts()).execute(context)
        request = model.generate_structured.await_args.kwargs
        fact, = result.evidence
        projected, = request["input_payload"]["deterministic_sufficiency"]["evidence"]
        assert "samples" not in projected["series_computations"][0]
        assert "visualizations" not in projected
        assert projected["rows"] == fact.model_dump(mode="json")["rows"]
        assert projected["provenance"] == fact.model_dump(mode="json")["provenance"]
        assert projected["series_computations"][0]["summary"] == fact.series_computations[0].summary
        assert len(fact.series_computations[0].samples) == 2048
        assert len(json.dumps(projected, ensure_ascii=False, separators=(",", ":"))) < len(fact.model_dump_json()) / 2
        assert before + timedelta(seconds=14) < request["deadline"] < datetime.now(UTC) + timedelta(seconds=16)
        assert result.status in {"ANSWERABLE", "PARTIAL"}
    asyncio.run(exercise())


@pytest.mark.parametrize("remaining,cap", [(80, 7), (25, 60)])
def test_assessment_uses_shorter_of_stage_budget_and_remaining_work(remaining, cap):
    async def exercise():
        config = AIOpsDiagnosisConfig(evidence_assessment_budget_seconds=cap)
        context = replace(_context(artifacts=(_tool_artifact(semantics="CURRENT_ACTIVITY"),)),
            deadline_at=(datetime.now(UTC) + timedelta(seconds=remaining)).isoformat())
        model = SimpleNamespace(generate_structured=AsyncMock(side_effect=AIOpsModelError("MODEL_DEADLINE_EXCEEDED")))
        before = datetime.now(UTC)
        await DbaEvidenceAssessmentHandler(model_client=model, prompts=_TestPrompts(), diagnosis_config=config).execute(context)
        actual = model.generate_structured.await_args.kwargs["deadline"]
        work = datetime.fromisoformat(context.deadline_at) - timedelta(seconds=config.answer_reserve_seconds)
        assert actual <= work
        assert actual <= datetime.now(UTC) + timedelta(seconds=cap)
        if remaining == 25:
            assert actual == work
        else:
            assert before + timedelta(seconds=cap) <= actual
    asyncio.run(exercise())


@pytest.mark.parametrize("missing_metric", [False, True])
def test_fixed_health_starter_uses_one_model_call_for_verified_final_answer(missing_metric):
    async def exercise():
        tools = ("db.instance.identity", "db.session.active", "db.session.blocking_chain",
                 "db.storage.capacity", "db.transaction.long_running", "db.replication.status")
        artifacts = tuple(_tool_artifact(semantics="CURRENT_ACTIVITY", tool_id=tool, row_count=0 if index == 2 else 1)
                          for index, tool in enumerate(tools))
        context = _context(artifacts=artifacts, task_frame_overrides={"diagnostic_profile": "GENERAL"})
        context = replace(context, deadline_at=(datetime.now(UTC) + timedelta(seconds=85)).isoformat(),
                          plan_snapshot={**context.plan_snapshot, "planning_route": {"mode": "CONVERSATION_STARTER"},
                            "monitoring": {"initial_gaps": [{"code": "SOURCE_UNAVAILABLE", "detail": "监控源不可用"}]} if missing_metric else {}})
        model = _AnswerModel(evidence_refs=())
        model.generate_structured = AsyncMock(side_effect=model.generate_structured)
        assessed = await DbaEvidenceAssessmentHandler(model_client=model, prompts=_TestPrompts()).execute(context)
        model.generate_structured.assert_not_awaited()
        # 身份探测只用于来源校验，不冒充第六项健康结论。
        assert len(assessed.evidence) == 5
        assert assessed.investigation.next_action == "ANSWER"
        assert assessed.status == ("PARTIAL" if missing_metric else "ANSWERABLE")
        assert bool(assessed.gaps) == missing_metric
        final_context = replace(context, input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1",
            "artifact_id": "assessment", "payload": assessed.model_dump(mode="json")},))
        result = await DbaAnswerComposeHandler(model_client=model, prompts=_TestPrompts()).execute(final_context)
        model.generate_structured.assert_awaited_once()
        request = model.generate_structured.await_args.kwargs
        assert request["purpose"] == "aiops.dba-diagnosis-answer-compose"
        assert (request["deadline"] - datetime.now(UTC)).total_seconds() > 80
        assert len(request["input_payload"]["sufficiency"]["evidence"]) == 5
        assert result.investigation_record is not None
        assert result.investigation_record.claims[0].level == "SUPPORTED"
        assert result.evidence_gaps == assessed.gaps
        assert result.status == ("PARTIAL" if missing_metric else "COMPLETED")
    asyncio.run(exercise())


def test_many_monitor_series_share_metadata_without_losing_rows_or_sources():
    artifact = monitoring(("9007199254740993.000000001", "9007199254740993.000000004"))
    from copy import deepcopy
    from aiops_agent.contracts.evidence import ObservationSet
    base = artifact["payload"]["observations"][0]
    artifact["payload"]["observations"] = []
    for index in range(32):
        observation = deepcopy(base)
        observation["metric_code"] = f"health.metric.{index}"
        artifact["payload"]["observations"].append(observation)
    fact = DbaEvidenceAssessmentHandler._monitoring_fact(artifact_id=artifact["artifact_id"],
        result=ObservationSet.model_validate(artifact["payload"]))
    before = fact.model_dump_json()
    projected = evidence_model_payload(fact)
    originals = [item.model_dump(mode="json", exclude={"samples"}) for item in fact.series_computations]
    assert len(originals) == 32
    for original, compact in zip(originals, projected["series_computations"], strict=True):
        assert {**projected["series_computation_defaults"], **compact} == original
        assert compact["input_ref"] == original["input_ref"]
        assert compact["row_index"] == original["row_index"]
        assert compact["summary"] == original["summary"]
    assert len(json.dumps(projected["series_computations"]) + json.dumps(projected["series_computation_defaults"])) < len(json.dumps(originals)) * 0.8
    assert projected["rows"] == fact.model_dump(mode="json")["rows"]
    assert projected["provenance"] == fact.model_dump(mode="json")["provenance"]
    assert fact.model_dump_json() == before


def test_native_metadata_compaction_is_lossless_and_does_not_merge_per_row_quality():
    metadata = {"unit": "s", "kind": "ADDITIVE_COUNTER", "projection": "IDENTITY", "aggregation": "NONE",
        "native_dimensions": [], "comparable_window_verified": False}
    raw = {"rows": [{"row_index": index, "values": {"duration": {
        **metadata, "value": "9007199254740993.000000001" if index else None,
        "quality": "MISSING" if not index else "OBSERVED_VALUE_CONVERSION", "native_origin_ids": [f"origin:{index}"],
    }}} for index in range(32)], "measurement_count": 32, "truncated": False,
        "verification_scope": "MAPPED_FIELDS_ONLY", "comparable_window_verified": False}
    compact = compact_native_values(raw)
    for original, projected in zip(raw["rows"], compact["rows"], strict=True):
        assert {**compact["fields"]["duration"], **projected["values"]["duration"]} == original["values"]["duration"]
    assert compact["truncated"] is False and compact["comparable_window_verified"] is False
    assert len(json.dumps(compact)) < len(json.dumps(raw)) * 0.7
    raw["rows"][1]["values"]["duration"]["unit"] = "ms"
    with pytest.raises(ValueError, match="共享语义不一致"):
        compact_native_values(raw)


def raw_fact(tool_id="db.storage.capacity", *, empty=False):
    raw = _tool_artifact(semantics="CURRENT_ACTIVITY", tool_id=tool_id, row_count=0 if empty else 1)
    source = DbaToolResult.model_validate(raw["payload"])
    fact = database_evidence_fact(result=source, outcome=source.tool_outcomes[0], artifact_id=raw["artifact_id"])
    return SimpleNamespace(artifact_id=raw["artifact_id"], schema_version=raw["schema_version"], trust_level="SOURCE_VERIFIED", payload_json=raw["payload"]), fact


def test_health_deadline_keeps_all_six_observations_with_times_values_and_empty_scope():
    tool_ids = ("db.instance.identity", "db.session.active", "db.session.blocking_chain",
        "db.storage.capacity", "db.transaction.long_running", "db.replication.status")
    pairs = [raw_fact(tool, empty=index == 2) for index, tool in enumerate(tool_ids)]
    artifacts = [item[0] for item in pairs]
    result = deadline_result(artifacts=artifacts, task_frame={"diagnostic_profile": "GENERAL"},
        database_type="ORACLE", projected_ids={item.artifact_id for item in artifacts})
    assert result.status == "PARTIAL" and len(result.evidence) == 6
    body = next(item.payload["markdown"] for item in result.blocks if item.block_type == "ANALYSIS_MARKDOWN")
    assert "不等同于健康结论" in body and "不能据此判定数据库整体正常" in body
    for _, fact in pairs:
        assert fact.tool_id in body and fact.captured_at in body
    tables = [block for block in result.blocks if block.block_type == "TABLE"]
    assert len(tables) == 5
    assert tables[0].payload["rows"] == [["abc123", 120.5]]
    assert {ref for table in tables for ref in table.evidence_refs} <= {fact.evidence_ref for fact in result.evidence}
    assert "RUN_DEADLINE_EXCEEDED" in {gap.code for gap in result.evidence_gaps}


def test_fallback_preview_preserves_precision_truncation_history_gaps_and_original_rows():
    _, fact = raw_fact()
    fact = fact.model_copy(update={"rows": tuple(("sql", "9007199254740993.000000001") for _ in range(32)),
        "row_count": 32, "truncated": True, "warnings": ("覆盖不完整",), "followup_run_id": fact.artifact_id})
    before = fact.model_dump_json()
    table, = observed_tables((fact,))
    assert len(table.payload["rows"]) == 10 and table.payload["row_count"] == 32
    assert table.payload["truncated"] is True
    assert table.payload["rows"][0][1] == "9007199254740993.000000001"
    gap = TurnEvidenceGap(source_id="monitor", code="SOURCE_UNAVAILABLE", detail="监控源不可用")
    body = observed_facts_markdown((fact,), (gap,))
    assert all(text in body for text in ("结果已截断", "不代表当前状态", "覆盖不完整", "SOURCE_UNAVAILABLE"))
    assert fact.model_dump_json() == before
    assert observed_tables((fact.model_copy(update={"trust_level": "USER_PROVIDED"}),)) == ()


@pytest.mark.parametrize("tool", ["db.oracle.awr.report", "db.oracle.sql_monitor.report"])
def test_raw_html_does_not_reenter_assessment_or_fallback_tables(tool):
    _, fact = raw_fact(tool)
    fact = fact.model_copy(update={"rows": (("<script>report</script>",),), "columns": ({"name": "output", "logical_type": "STRING"},)})
    assert evidence_model_payload(fact)["rows"] == []
    assert observed_tables((fact,)) == ()
    assert "<script>" not in observed_facts_markdown((fact,), ())
    assert fact.rows == (("<script>report</script>",),)


def test_answer_model_timeout_resets_draft_and_finishes_partial_before_worker_deadline():
    async def exercise():
        _, fact = raw_fact()
        assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(fact,))
        context = replace(_context(task_frame_overrides={"diagnostic_profile": "GENERAL"}),
            deadline_at=(datetime.now(UTC) + timedelta(seconds=50)).isoformat(),
            input_artifacts=({"artifact_id": "assessment", "schema_version": "DBA_SUFFICIENCY.v1", "payload": assessment.model_dump(mode="json")},))
        deltas = []
        async def on_delta(item):
            deltas.append(item)
        async def timeout(**kwargs):
            assert kwargs["text_field"] == "answer_markdown"
            assert kwargs["deadline"] == datetime.fromisoformat(context.deadline_at) - timedelta(seconds=3)
            await kwargs["on_text_delta"](ModelTextDelta(delta="未核验草稿"))
            raise AIOpsModelError("MODEL_DEADLINE_EXCEEDED")
        model = SimpleNamespace(generate_structured=AsyncMock(side_effect=timeout))
        result = await DbaAnswerComposeHandler(model_client=model, prompts=_TestPrompts()).execute(context, on_text_delta=on_delta)
        assert deltas[-1].reset and result.status == "PARTIAL" and not result.answer_streamed
        assert result.investigation_record.stop_reason != "GOAL_ANSWERED"
        assert not any(block.payload.get("raw_model_output") for block in result.blocks)
        body = "\n".join(block.payload.get("markdown", "") for block in result.blocks)
        assert fact.captured_at in body and fact.tool_id in body and "不等同于健康结论" in body
        assert any(block.payload.get("rows") == [["abc123", 120.5]] for block in result.blocks)
        assert "执行预算" in body and "这不是证据校验失败" in body
        assert "诊断生成未通过证据校验或模型服务不可用" not in body
        assert any(gap.source_id == "diagnosis.answer" and gap.code == "MODEL_DEADLINE_EXCEEDED" for gap in result.evidence_gaps)
        model.generate_structured.assert_awaited_once()
    asyncio.run(exercise())


@pytest.mark.parametrize("error,message,code", [
    (AIOpsModelError("MODEL_CONFIGURATION_REQUIRED"), "尚未配置诊断模型", "MODEL_CONFIGURATION_REQUIRED"),
    (AIOpsModelError("MODEL_SERVICE_UNAVAILABLE"), "服务暂不可用", "MODEL_SERVICE_UNAVAILABLE"),
    (AIOpsModelError("MODEL_STREAM_INTERRUPTED"), "响应中断", "MODEL_STREAM_INTERRUPTED"),
    (AIOpsModelError("MODEL_OUTPUT_INVALID"), "结构化回答合同", "MODEL_OUTPUT_INVALID"),
    (ValueError("不应泄漏的内部错误详情"), "证据引用校验", "DIAGNOSIS_VALIDATION_FAILED"),
])
def test_answer_failure_explains_actual_stage_without_exposing_internal_details(error, message, code):
    async def exercise():
        _, fact = raw_fact()
        assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(fact,))
        context = replace(_context(task_frame_overrides={"diagnostic_profile": "GENERAL"}),
            input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "artifact_id": "assessment",
                              "payload": assessment.model_dump(mode="json")},))
        result = await DbaAnswerComposeHandler(model_client=SimpleNamespace(
            generate_structured=AsyncMock(side_effect=error)), prompts=_TestPrompts()).execute(context)
        body = "\n".join(block.payload.get("markdown", "") for block in result.blocks)
        assert message in body and code in body
        assert "不应泄漏" not in body
        assert result.status == "PARTIAL" and result.evidence == assessment.evidence
        assert result.evidence_gaps[-1].code == code
    asyncio.run(exercise())


def test_finalize_reserve_is_configurable_and_must_fit_answer_reserve():
    with pytest.raises(ValidationError, match="落盘预留"):
        AIOpsDiagnosisConfig(answer_reserve_seconds=5, answer_finalize_reserve_seconds=5)
    handler = DbaAnswerComposeHandler(model_client=None, prompts=None,
        diagnosis_config=AIOpsDiagnosisConfig(answer_finalize_reserve_seconds=7))
    deadline = datetime.now(UTC)
    assert handler._answer_deadline(deadline.isoformat()) == deadline - timedelta(seconds=7)
    assert handler._answer_deadline(None) is None


def test_streaming_prompt_schema_starts_with_body_even_when_contract_inherits_it_last():
    from aiops_agent.contracts.investigation import DiagnosisStreamResponse

    async def exercise():
        payload = {
            "answer_markdown": "当前证据不足以作健康判断。",
            "claims": [{
                "claim_id": "c1", "kind": "INTERPRETATION", "subject": "当前目标",
                "statement": "当前证据不足以作健康判断。", "level": "UNKNOWN",
                "applicable_time": "本次采集时点", "reasoning": "仍有明确缺口。",
                "limitations": ["不能推断数据库整体健康。"],
            }], "goal_satisfied": False,
        }
        original_schema = DiagnosisStreamResponse.model_json_schema()
        assert list(original_schema["properties"])[-1] == "answer_markdown"
        session = Session(stream_response(payload))
        result = await generate(session, AsyncMock(), output_model=DiagnosisStreamResponse)
        prompt = session.payloads[0]["messages"][0]["content"]
        schema, _ = json.JSONDecoder().raw_decode(prompt.split("返回对象必须严格满足以下 JSON Schema：\n", 1)[1])
        assert next(iter(schema["properties"])) == "answer_markdown"
        assert schema["required"][0] == "answer_markdown"
        assert schema == original_schema | {"required": schema["required"]}
        assert set(schema["required"]) == set(original_schema["required"])
        assert result.output.answer_markdown == payload["answer_markdown"]
        assert DiagnosisStreamResponse.model_json_schema() == original_schema
    asyncio.run(exercise())
