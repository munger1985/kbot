"""健康诊断的可读数值与精确证据分离，不靠正文正则改写已交付回复。"""

import asyncio
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.display_dictionary import format_metric_value
from aiops_agent.application.diagnosis.number_presentation import NUMBER_PRESENTATION_POLICY, monitoring_display_values
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment
from aiops_agent.domain.evidence.series_calculation import replay_series_computation
from aiops_agent.ports.model import ModelTextDelta
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler
from tests.unit.aiops_agent.test_aiops_series_calculation import draft, fact_from, monitoring
from tests.unit.aiops_agent.test_aiops_turn_answer import _context, _TestPrompts


@pytest.mark.parametrize("value,unit,expected", [
    (0.15438596491228068, "transactions_per_second", "0.15 个事务/秒"),
    ("0.16982456", "transactions_per_second", "0.17 个事务/秒"),
    (Decimal("0.6210526315789473"), "transactions_per_second", "0.62 个事务/秒"),
    (Decimal("0.68134503"), "transactions_per_second", "0.68 个事务/秒"),
    (Decimal("1.005"), "milliseconds", "1.01 毫秒"),
    (87.123456, "percent", "87.12%"),
    (Decimal("1234.5678"), "queries_per_second", "1,234.57 次查询/秒"),
    (60 * 1024 ** 3, "bytes", "60.00 GiB"),
    (Decimal("1572864.12345"), "bytes_per_second", "1.50 MiB/秒"),
    (Decimal("0.00456789"), "waits_per_second", "0.00457 次等待/秒"),
    (Decimal("-0.0000000001234567"), "errors_per_second", "-1.23E-10 次错误/秒"),
    (Decimal("-0.000"), "count", "0"),
    ("9007199254740993", "count", "9,007,199,254,740,993"),
    (2.125, "sessions", "2.13 个会话"),
    (None, "percent", "无有效值"),
    (float("nan"), "percent", "无有效值"),
    (float("inf"), "percent", "无有效值"),
    ("abc123", "count", "abc123"),
])
def test_metric_numbers_are_readable_without_float_precision_loss(value, unit, expected):
    assert format_metric_value(value, unit) == expected


def throughput_fact():
    artifact = monitoring(("0.18526315508771932", "0.15438596491228068"), days=1 / 24)
    observation = artifact["payload"]["observations"][0]
    observation.update(metric_code="db.transaction.throughput", unit="transactions_per_second", value_kind="GAUGE")
    observation["series"][0]["dimensions"]["command"] = "commit"
    rollback = deepcopy(observation["series"][0])
    rollback["dimensions"]["command"] = "rollback"
    for point, value in zip(rollback["points"], ("0.7416374284210527", "0.6210526315789473"), strict=True):
        point["value"] = value
    observation["series"].append(rollback)
    return fact_from(artifact)


def test_display_values_keep_hour_window_rate_units_dimensions_and_exact_evidence():
    fact = throughput_fact()
    before = fact.model_dump_json()
    projected = DbaAnswerComposeHandler._compose_evidence_payload(fact)
    values = projected["display_values"]
    assert values[0]["values"]["latest"] == "0.15 个事务/秒"
    assert values[0]["values"]["average"] == "0.17 个事务/秒"
    assert values[1]["values"]["latest"] == "0.62 个事务/秒"
    assert values[1]["values"]["average"] == "0.68 个事务/秒"
    assert [item["row_index"] for item in values] == [0, 1]
    assert projected["rows"] == fact.model_dump(mode="json")["rows"]
    assert projected["provenance"] == fact.model_dump(mode="json")["provenance"]
    for trace in fact.series_computations:
        assert (trace.window_end - trace.window_start).total_seconds() == 3600
        assert replay_series_computation(trace) == trace.summary
    assert fact.model_dump_json() == before


@pytest.mark.parametrize("unit", ["UNKNOWN", "UNSPECIFIED", "", "unix_seconds", "state"])
def test_unknown_units_and_identity_or_state_values_are_not_guessed_or_rounded(unit):
    fact = throughput_fact()
    index = next(index for index, column in enumerate(fact.columns) if column["name"] == "unit")
    rows = [list(row) for row in fact.rows]
    for row in rows:
        row[index] = unit
    assert monitoring_display_values(fact.model_copy(update={"rows": tuple(tuple(row) for row in rows)})) == []


def test_final_diagnosis_receives_readable_values_and_keeps_same_streamed_reply():
    async def exercise():
        fact = throughput_fact()
        assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(fact,))
        context = replace(_context(task_frame_overrides={"objectives": ["DIAGNOSE"], "diagnostic_profile": "GENERAL"}),
            input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": assessment.model_dump(mode="json")},))
        native = ""
        async def generate(**kwargs):
            nonlocal native
            payload = kwargs["input_payload"]
            assert payload["number_presentation"] == NUMBER_PRESENTATION_POLICY
            assert "原始精确值" in payload["number_presentation"]
            display = payload["sufficiency"]["evidence"][0]["display_values"]
            native = (f"同一小时采样窗口内，commit 最新约 {display[0]['values']['latest']}，"
                f"窗口均值约 {display[0]['values']['average']}；rollback 最新约 {display[1]['values']['latest']}，"
                f"窗口均值约 {display[1]['values']['average']}。")
            await kwargs["on_text_delta"](ModelTextDelta(delta=native))
            return SimpleNamespace(output=kwargs["output_model"].model_validate({**draft(fact).model_dump(), "answer_markdown": native}),
                receipt=SimpleNamespace(model_dump=lambda **_: {}))
        model = SimpleNamespace(generate_structured=AsyncMock(side_effect=generate))
        items = [item async for item in DbaAnswerComposeHandler(model_client=model, prompts=_TestPrompts()).execute_stream(context)]
        result = items[-1]
        assert result.investigation_record and result.evidence == (fact,)
        block = next(block for block in result.blocks if block.payload.get("raw_model_output"))
        assert block.payload["markdown"] == native
        assert "".join(item.payload["delta"] for item in items[:-1] if item.event_type == "answer.delta") == native
        assert "latest=" not in native and "0.15438596491228068" not in native
        model.generate_structured.assert_awaited_once()
    asyncio.run(exercise())
