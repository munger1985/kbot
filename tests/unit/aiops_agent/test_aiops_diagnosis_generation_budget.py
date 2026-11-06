"""完整诊断预算、模型判断视图及无正文流统计回归。"""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from loguru import logger
from pydantic import ValidationError

from aiops_agent.adapters.model_serving import AIOpsModelError, AIOpsStructuredModelClient
from aiops_agent.application.diagnosis.model_evidence import evidence_model_payload
from aiops_agent.application.turns import ConversationTurnService
from aiops_agent.config import AIOpsDiagnosisConfig
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment
from aiops_agent.domain.evidence.physical_identity import physical_identity_model_context
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler
from platform_core.contracts.aiops import (
    ConversationCreate, ConversationSourceContext, ConversationStarterSelection,
    TrustedContextHint, TurnCreate,
)
from aiops_agent.application.conversation_starters import ConversationStarterCatalog
from tests.unit.aiops_agent.test_aiops_native_answer_stream import Response, Session, frame, generate, stream_response
from tests.unit.aiops_agent.test_aiops_series_calculation import fact_from, monitoring
from tests.unit.aiops_agent.test_aiops_turn_application import _Uow
from tests.unit.aiops_agent.test_aiops_turn_answer import _context


@pytest.mark.parametrize("starter,expected", [("database.health.overview", 240), (None, 90)])
def test_only_verified_diagnostic_selection_receives_diagnostic_budget(starter, expected):
    async def exercise():
        uow = _Uow()
        uow.target.connectivity_status = "CONNECTED"
        service = ConversationTurnService(uow_factory=lambda: uow)
        before = datetime.now(UTC)
        await service.start(domain_id=7, actor_id="dba@example.com",
            trace_id="budget-test", conversation_create=ConversationCreate(
                agent_id=uow.agent.agent_id, source=ConversationSourceContext(),
                context_hint=TrustedContextHint(source_kind="RESOURCE_PAGE", resource_ids=(uow.resource.resource_id,))),
            first_turn=TurnCreate(idempotency_key="health-budget",
                starter=ConversationStarterSelection(starter_id=starter, catalog_version=ConversationStarterCatalog().version)
                if starter else None,
                content=() if starter else ({"content_type": "TEXT", "text": "查看当前情况"},)))
        deadline = datetime.fromisoformat(uow.outbox.rows[-1].payload_json["execution_context"]["deadline_at"])
        assert before + timedelta(seconds=expected) <= deadline <= datetime.now(UTC) + timedelta(seconds=expected)
        # 已冻结期限必须原样保留；不能在接收重试时重新获得完整诊断预算。
        await service._create_turn(uow=uow, conversation=uow.conversation_rows[0], version=uow.version,
            command=TurnCreate(idempotency_key="health-existing-deadline", content=({"content_type": "TEXT", "text": "继续"},),
                starter=ConversationStarterSelection(starter_id=starter, catalog_version=ConversationStarterCatalog().version)
                if starter else None),
            actor_id="dba@example.com", trace_id="existing-budget", source_run=None,
            focused_target_id=uow.target.target_id, execution_context={"deadline_at": deadline.isoformat()})
        assert uow.outbox.rows[-1].payload_json["execution_context"]["deadline_at"] == deadline.isoformat()
        before = datetime.now(UTC)
        await service._create_turn(uow=uow, conversation=uow.conversation_rows[0], version=uow.version,
            command=TurnCreate(idempotency_key="metadata-is-not-selection", content=({"content_type": "TEXT", "text": "继续"},)),
            actor_id="dba@example.com", trace_id="metadata-budget", source_run=None,
            execution_context={"conversation_starter": {"execution_mode": "DIAGNOSTIC"}})
        metadata_deadline = datetime.fromisoformat(uow.outbox.rows[-1].payload_json["execution_context"]["deadline_at"])
        assert before + timedelta(seconds=90) <= metadata_deadline <= datetime.now(UTC) + timedelta(seconds=90)
    asyncio.run(exercise())


def test_diagnostic_budget_is_configurable_and_must_leave_answer_time():
    config = AIOpsDiagnosisConfig(diagnostic_turn_budget_seconds=360)
    service = ConversationTurnService(uow_factory=None, diagnosis_config=config)
    assert service._diagnostic_turn_budget_seconds == 360 and service._turn_budget_seconds == 90
    with pytest.raises(ValidationError, match="诊断总预算"):
        AIOpsDiagnosisConfig(diagnostic_turn_budget_seconds=30, answer_reserve_seconds=30)


def test_many_high_precision_metrics_have_one_value_copy_and_keep_complete_replay():
    values = tuple(f"9007199254740993.{index:018d}" for index in range(120))
    artifact = monitoring(values, days=1 / 2880)
    base = artifact["payload"]["observations"][0]
    artifact["payload"]["observations"] = [{**deepcopy(base), "metric_code": f"metric.{index}"} for index in range(32)]
    fact = fact_from(artifact)
    before = fact.model_dump_json()
    projected = evidence_model_payload(fact)
    assert len(projected["rows"]) == 32 and len(projected["series_computations"]) == 32
    assert projected["rows"] == fact.model_dump(mode="json")["rows"]
    previous = fact.model_dump(mode="json", exclude={"visualizations", "series_computations"})
    previous["series_computations"] = [item.model_dump(mode="json", exclude={"samples"}) for item in fact.series_computations]
    size = lambda item: len(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
    assert size(projected) < size(previous) * 0.6
    assert all(set(item["summary"]) == {"elapsed_days"} for item in projected["series_computations"])
    assert all(len(item.samples) == 120 for item in fact.series_computations)
    assert fact.model_dump_json() == before


def test_projection_keeps_nonduplicated_summary_and_row_quality():
    fact = fact_from(monitoring((1, 2, 3)))
    trace = fact.series_computations[0]
    altered = trace.model_copy(update={"summary": {**trace.summary, "change": 999, "extra": 42}, "quality_flags": ("PARTIAL_COVERAGE",)})
    projected = evidence_model_payload(fact.model_copy(update={"series_computations": (altered,)}))
    item, = projected["series_computations"]
    assert item["summary"] == {"change": 999, "extra": 42, "elapsed_days": 2}
    assert item["quality_flags"] == ["PARTIAL_COVERAGE"]
    assert item["input_ref"] == trace.input_ref


def test_shared_defaults_never_override_fields_explicitly_read_from_another_row():
    artifact = monitoring(tuple(range(10)))
    base = artifact["payload"]["observations"][0]
    artifact["payload"]["observations"] = [deepcopy(base), {**deepcopy(base), "metric_code": "other.metric"}]
    fact = fact_from(artifact)
    traces = tuple(item.model_copy(update={"forecast_horizon_days": None}) for item in fact.series_computations)
    horizon_index = next(index for index, column in enumerate(fact.columns) if column["name"] == "forecast_horizon_days")
    rows = [list(row) for row in fact.rows]
    rows[0][horizon_index] = 7
    changed = fact.model_copy(update={"rows": tuple(tuple(row) for row in rows), "series_computations": traces})
    projected = evidence_model_payload(changed)
    assert "forecast_horizon_days" not in projected["series_computation_defaults"]
    assert projected["series_computations"][0]["forecast_horizon_days"] is None
    assert "forecast_horizon_days" not in projected["series_computations"][1]


def test_identity_sample_view_preserves_comparison_without_implying_continuity():
    from tests.unit.aiops_agent.test_aiops_physical_identity_graph import monitoring as identity_monitoring, direct, NOW
    points = [(NOW - timedelta(seconds=119 - index), str(int((NOW - timedelta(seconds=120 - index)).timestamp()))) for index in range(120)]
    monitor, _, result, _ = identity_monitoring("ORACLE", points=points)
    database, _ = direct("ORACLE", result.target_id)
    before = monitor.model_dump_json()
    payload = DbaAnswerComposeHandler._model_assessment_payload(DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(database, monitor)))
    witness, = payload["evidence"][1]["source_identity_witnesses"]
    assert "sample_times" not in witness and witness["sample_time_extent"]["sample_count"] == 120
    graph = payload["physical_identity_graph"]
    assert all("observed_times" not in node for node in graph["nodes"])
    assert any(node["observed_time_extent"]["sample_count"] == 120 for node in graph["nodes"])
    assert all(not edge["window_continuity_verified"] and not edge["exact_runtime_instance_verified"] for edge in graph["edges"])
    assert graph == physical_identity_model_context((database, monitor))["physical_identity_graph"]
    assert monitor.model_dump_json() == before


@pytest.mark.parametrize("valid", [True, False])
def test_actual_stream_adapter_preserves_text_and_separates_metric_validation(valid):
    import hashlib
    from types import SimpleNamespace
    from tests.unit.aiops_agent.test_aiops_series_calculation import draft
    from platform_core.identity import uuid7

    async def exercise():
        artifact = monitoring(tuple(range(120)), days=1 / 2880)
        base = artifact["payload"]["observations"][0]
        artifact["payload"]["observations"] = [{**deepcopy(base), "metric_code": f"metric.{index}"} for index in range(32)]
        fact = fact_from(artifact)
        payload = draft(fact).model_dump(mode="json")
        payload["answer_markdown"] = "本轮已取得监控观测；该时点数值不能证明数据库整体健康。\n"
        if not valid:
            payload["claims"][0]["evidence_fields"][0]["field"] = "nonexistent"
        session = Session(stream_response(payload))
        client = AIOpsStructuredModelClient(base_url="http://model", audience="model", caller_service="aiops", timeout_seconds=300, session=session)
        prompt_text = "依据已有证据返回诊断。"
        prompts = SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace(content=prompt_text, ref=lambda: {
            "prompt_id": "test", "prompt_version": "1.0.0", "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest()})))
        context = _context(task_frame_overrides={"diagnostic_profile": "GENERAL"})
        context.plan_snapshot["answer_context"]["model"].update(model_id=str(uuid7()), tenant_id=str(uuid7()))
        context = replace(context, target_id=str(artifact["payload"]["target_id"]),
            deadline_at=(datetime.now(UTC) + timedelta(seconds=235)).isoformat(),
            input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(fact,)).model_dump(mode="json")},))
        deltas = []
        async def on_delta(delta):
            deltas.append(delta)
        with patch("aiops_agent.adapters.model_serving.build_internal_auth_headers", return_value={}):
            result = await DbaAnswerComposeHandler(model_client=client, prompts=prompts).execute(context, on_text_delta=on_delta)
        assert len(session.payloads) == 1
        assert 220 < session.payloads[0]["timeout"] <= 232
        assert "max_tokens" not in session.payloads[0]
        assert result.evidence == (fact,)
        assert len(result.evidence[0].series_computations) == 32
        if valid:
            assert result.status == "COMPLETED" and result.answer_streamed
            assert result.investigation_record.claims[0].level == "OBSERVED"
            assert any(block.payload.get("markdown") == payload["answer_markdown"] for block in result.blocks)
            assert not deltas[-1].reset
        else:
            assert result.status == "PARTIAL" and result.answer_streamed and not deltas[-1].reset
            block = next(block for block in result.blocks if block.payload.get("raw_model_output"))
            assert block.payload["markdown"] == payload["answer_markdown"]
            assert block.payload["validation_status"] == "UNVERIFIED"
            assert result.evidence_gaps[-1].code == "DIAGNOSIS_VALIDATION_FAILED"
    asyncio.run(exercise())


@pytest.mark.parametrize("stage", ["waiting", "reasoning", "content", "answer", "complete", "cancelled"])
def test_stream_progress_is_logged_on_success_timeout_and_cancellation_without_content(stage):
    async def exercise():
        secret_reasoning = "不应写入日志的推理文本"
        secret_body = "不应写入日志的诊断正文"
        class ProgressResponse(Response):
            async def iter_any(self):
                if stage != "waiting":
                    yield frame(reasoning=secret_reasoning)
                if stage in {"content", "cancelled"}:
                    yield frame('{"answer_markdown":')
                elif stage in {"answer", "complete"}:
                    yield frame(json.dumps({"answer_markdown": secret_body + "\n", "verified_count": 1}, ensure_ascii=False))
                if stage == "complete":
                    yield frame(finish="stop")
                    yield b"data: [DONE]\n\n"
                elif stage == "cancelled":
                    raise asyncio.CancelledError
                else:
                    raise TimeoutError
        logs = []
        sink = logger.add(lambda message: logs.append(str(message)), format="{message}")
        try:
            operation = generate(Session(ProgressResponse([])), AsyncMock())
            if stage == "complete":
                assert (await operation).output.verified_count == 1
            elif stage == "cancelled":
                with pytest.raises(asyncio.CancelledError):
                    await operation
            else:
                with pytest.raises(AIOpsModelError):
                    await operation
        finally:
            logger.remove(sink)
        combined = "\n".join(logs)
        assert secret_reasoning not in combined and secret_body not in combined
        record, = [item for item in logs if "模型流阶段统计" in item]
        assert f"stream_ended={stage == 'complete'}" in record
        assert f"reasoning_chars={0 if stage == 'waiting' else len(secret_reasoning)}" in record
        assert ("first_content_ms=None" in record) == (stage in {"waiting", "reasoning"})
        assert ("first_answer_ms=None" in record) == (stage not in {"answer", "complete"})
    asyncio.run(exercise())
