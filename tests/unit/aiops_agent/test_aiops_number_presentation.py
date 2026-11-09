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
from aiops_agent.application.target_identity import target_answer_presentation, target_identity_snapshot
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


def state_fact(values=(1, 1), *, role="AVAILABILITY", metric_code="postgresql.availability"):
    artifact = monitoring(values, value_kind="STATE", days=1 / 24)
    observation = artifact["payload"]["observations"][0]
    observation.update(metric_code=metric_code, semantic_role=role, unit="state")
    return fact_from(artifact)


@pytest.mark.parametrize("role,metric_code,value,expected", [
    ("AVAILABILITY", "postgresql.availability", 1, "可用"),
    ("AVAILABILITY", "mysql.availability", "1.0", "可用"),
    ("AVAILABILITY", "db.availability", 0, "不可用"),
    ("BACKUP_LAST_RUN_STATUS", "backup.last_run.status", 1, "成功"),
    ("BACKUP_LAST_RUN_STATUS", "backup.last_run.status", 0, "失败"),
    ("AVAILABILITY", "postgresql.availability", 2, "未知状态"),
    ("AVAILABILITY", "postgresql.availability", "-1", "未知状态"),
])
def test_state_translation_uses_semantic_role_not_the_raw_boolean_code(role, metric_code, value, expected):
    fact = state_fact((value, value), role=role, metric_code=metric_code)
    before = fact.model_dump_json()
    result = DbaAnswerComposeHandler._compose_evidence_payload(fact)
    item = result["display_values"][0]
    assert item["values"]["latest"] == expected
    assert item["field_labels"]["latest"] == "最新采样状态"
    assert item["observed_states"] == [expected]
    assert "average" not in item["values"] and "sample_p95" not in item["values"]
    assert fact.model_dump_json() == before
    assert result["rows"] == fact.model_dump(mode="json")["rows"]


def test_mixed_availability_samples_do_not_turn_latest_up_into_continuous_health():
    fact = state_fact((0, 1))
    item = monitoring_display_values(fact)[0]
    assert item["values"]["latest"] == "可用"
    assert item["values"]["minimum"] == "不可用"
    assert item["observed_states"] == ["不可用", "可用"]
    assert "整个窗口连续健康" in item["state_policy"]
    assert "average" not in item["values"]


def test_unknown_state_semantics_are_not_assumed_to_be_healthy():
    assert monitoring_display_values(state_fact(role="CUSTOM_STATE")) == []


@pytest.mark.parametrize("value,expected", [(None, "未取得有效状态"), ("nan", "未取得有效状态"),
    ("inf", "未取得有效状态"), ("unexpected", "未知状态")])
def test_missing_and_invalid_state_are_not_translated_to_normal(value, expected):
    fact = state_fact()
    names = tuple(column["name"] for column in fact.columns)
    rows = [list(row) for row in fact.rows]
    rows[0][names.index("latest")] = value
    projected = monitoring_display_values(fact.model_copy(update={"rows": tuple(tuple(row) for row in rows)}))
    assert projected[0]["values"]["latest"] == expected


def test_target_name_is_frozen_without_exposing_connection_configuration():
    target = SimpleNamespace(target_id=_context().target_id,
        display_name="MES PostgreSQL 主库", row_version=5, db_type="POSTGRESQL",
        endpoint_json={"host": "private"}, password="secret")
    frozen = target_identity_snapshot(target)
    assert set(frozen) == {"target_id", "display_name", "row_version", "db_type"}
    target.display_name = "已经改名"
    context = replace(_context(), target_id=target.target_id, plan_snapshot={"target": frozen})
    display = target_answer_presentation(context)
    assert display["display_name"] == "MES PostgreSQL 主库"
    assert display["target_id"] == target.target_id
    assert "SQL ID" in display["policy"] and "不输出 target UUID" in display["policy"]
    assert "private" not in str(display) and "secret" not in str(display)


@pytest.mark.parametrize("target_snapshot", [{}, {"display_name": "未绑定名称"},
    {"target_id": "another-target", "display_name": "其他数据库"}])
def test_missing_or_mismatched_target_name_is_not_invented_or_replaced_by_uuid(target_snapshot):
    context = replace(_context(), plan_snapshot={"target": target_snapshot})
    assert target_answer_presentation(context)["display_name"] == "所选数据库"


def test_health_diagnosis_receives_target_name_and_state_labels_without_rewriting_stream():
    async def exercise():
        fact = state_fact()
        assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(fact,))
        context = _context(task_frame_overrides={"objectives": ["DIAGNOSE"], "diagnostic_profile": "GENERAL"})
        snapshot = deepcopy(context.plan_snapshot)
        snapshot["target"] = {"target_id": context.target_id, "display_name": "MES PostgreSQL 主库"}
        context = replace(context, plan_snapshot=snapshot,
            input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": assessment.model_dump(mode="json")},))
        native = ""
        async def generate(**kwargs):
            nonlocal native
            payload = kwargs["input_payload"]
            name = payload["target_presentation"]["display_name"]
            status = payload["sufficiency"]["evidence"][0]["display_values"][0]["values"]["latest"]
            native = f"{name}当前{status}；本轮采集未发现需要处置的性能问题。"
            await kwargs["on_text_delta"](ModelTextDelta(delta=native))
            return SimpleNamespace(output=kwargs["output_model"].model_validate({
                **draft(fact).model_dump(), "answer_markdown": native}), receipt=SimpleNamespace(model_dump=lambda **_: {}))
        model = SimpleNamespace(generate_structured=AsyncMock(side_effect=generate))
        items = [item async for item in DbaAnswerComposeHandler(model_client=model, prompts=_TestPrompts()).execute_stream(context)]
        result = items[-1]
        block = next(block for block in result.blocks if block.payload.get("raw_model_output"))
        assert block.payload["markdown"] == native and context.target_id not in native
        assert "当前可用" in native
        assert "".join(item.payload["delta"] for item in items[:-1] if item.event_type == "answer.delta") == native
        assert result.evidence == (fact,)
        model.generate_structured.assert_awaited_once()
    asyncio.run(exercise())
