"""监控精度、采样语义、冻结重放与原始来源复核的端到端本地回归。"""

import asyncio
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.diagnosis.investigation_record import freeze_investigation, validate_diagnosis
from aiops_agent.application.diagnosis.monitoring_calculations import validate_monitoring_trace_sources
from aiops_agent.application.reporting import normalize_report_source
from aiops_agent.contracts.evidence import MetricPoint, ObservationSet
from aiops_agent.contracts.investigation import DiagnosisAnswerDraft, InvestigationClaim
from aiops_agent.domain.evidence.series_calculation import replay_series_computation, summarize_series
from aiops_agent.domain.evidence.trends import summarize_numeric_trend
from aiops_agent.playbooks import canonical_hash
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_turn_answer import _monitoring_artifact


NOW = datetime(2026, 10, 28, tzinfo=UTC)


def monitoring(values=(10, 13), *, value_kind="GAUGE", days=1):
    artifact = _monitoring_artifact()
    observation = artifact["payload"]["observations"][0]
    observation.update(value_kind=value_kind, window_start=NOW.isoformat(),
        window_end=(NOW + timedelta(days=days * (len(values) - 1))).isoformat(),
        expected_points=len(values), actual_points=len(values))
    observation["series"][0]["points"] = [
        {"observed_at": (NOW + timedelta(days=index * days)).isoformat(), "value": value}
        for index, value in enumerate(values)
    ]
    return artifact


def fact_from(artifact, *, horizon=None):
    return DbaEvidenceAssessmentHandler._monitoring_fact(artifact_id=artifact["artifact_id"],
        result=ObservationSet.model_validate(artifact["payload"]), forecast_horizon_days=horizon)


def columns(fact):
    return dict(zip((item["name"] for item in fact.columns), fact.rows[0]))


def draft(fact):
    return DiagnosisAnswerDraft(claims=(InvestigationClaim(
        claim_id="c1", kind="FACT", subject="本轮采样", statement="最新采样值已经取得",
        level="OBSERVED", applicable_time="本轮窗口",
        reasoning="直接读取本轮同源采样值，不判断根因。",
        evidence_fields=({"evidence_ref": fact.evidence_ref, "row_index": 0, "field": "latest"},),
    ),), goal_satisfied=True)


def test_metric_point_preserves_large_integer_and_serializes_exact_wire():
    point = MetricPoint(observed_at=NOW, value=900719925474099312345)
    assert point.value == 900719925474099312345
    assert point.model_dump(mode="json")["value"] == "900719925474099312345"


def test_high_precision_changes_survive_worker_freeze_model_input_and_report():
    base = "1234567890" * 6
    artifact = monitoring((base + "001", base + "004"))
    fact = fact_from(artifact)
    assert columns(fact)["change"] == 3
    assert columns(fact)["change_per_day"] == 3
    assert columns(fact)["latest"] == base + "004"
    assert replay_series_computation(fact.series_computations[0])["change"] == 3
    model_input = DbaAnswerComposeHandler._compose_evidence_payload(fact)
    assert "samples" not in model_input["series_computations"][0]
    assert model_input["series_computations"][0]["summary"]["change"] == 3
    assert len(fact.series_computations[0].samples) == 2
    context = SimpleNamespace(run_id=uuid7(), task_id=uuid7(), target_id=artifact["payload"]["target_id"],
        plan_snapshot={"answer_context": {"question": "分析容量变化"}})
    record = freeze_investigation(draft=draft(fact), evidence=(fact,), context=context)
    assert record.observations[0].series_computations == fact.series_computations
    source = normalize_report_source(schema_version="AIOPS_TURN_RESULT.v1", source_kind="CHAT", payload={
        "status": "COMPLETED", "investigation_record": record.model_dump(mode="json"),
        "evidence": [fact.model_dump(mode="json")],
    })
    assert source["status"] == "READY"


@pytest.mark.parametrize("kind", ["COUNTER", "STATE", "UNKNOWN"])
def test_counter_and_unknown_semantics_are_not_gauge_trends_or_forecasts(kind):
    fact = fact_from(monitoring(tuple(range(10)), value_kind=kind), horizon=30)
    row = columns(fact)
    assert row["latest"] == 9
    assert row["change"] is None
    assert row["forecast_value"] is None
    assert row["forecast_confidence"] is None
    assert any("TREND_SEMANTICS_UNSUPPORTED" in item for item in fact.warnings)


@pytest.mark.parametrize("change", [{"truncated": True}, {"coverage_ratio": 0.99}, {"unit": "UNKNOWN"}])
def test_incomplete_or_unknown_unit_never_produces_forecast(change):
    artifact = monitoring(tuple(range(10)))
    artifact["payload"]["observations"][0].update(change)
    fact = fact_from(artifact, horizon=30)
    assert columns(fact)["forecast_value"] is None
    assert fact.series_computations[0].quality_flags


def test_duplicate_conflicting_timestamps_do_not_pick_arbitrary_latest():
    artifact = monitoring()
    points = artifact["payload"]["observations"][0]["series"][0]["points"]
    points.append({**points[-1], "value": 14})
    fact = fact_from(artifact)
    assert columns(fact)["latest"] is None
    assert not fact.series_computations
    assert any("SERIES_COMPUTATION_REJECTED" in item for item in fact.warnings)


def test_out_of_order_points_use_time_not_array_position_and_deduplicate_exact_duplicates():
    artifact = monitoring()
    points = artifact["payload"]["observations"][0]["series"][0]["points"]
    points[:] = [points[-1], points[0], points[0]]
    fact = fact_from(artifact)
    assert columns(fact)["latest"] == 13
    assert columns(fact)["sample_count"] == 2
    assert columns(fact)["change"] == 3


def test_tiny_decimal_changes_remain_exact_and_p95_is_sample_distribution():
    tiny = summarize_numeric_trend(((NOW, "0.000000000000000001"), (NOW + timedelta(days=1), "0.000000000000000004")))
    assert Decimal(str(tiny["change"])) == Decimal("0.000000000000000003")
    summary, _ = summarize_series(tuple((NOW + timedelta(seconds=i), i) for i in range(20)), value_kind="GAUGE")
    assert summary["sample_p95"] == 18
    assert summary["sample_count"] == 20
    assert summarize_series(((NOW, 1),), value_kind="GAUGE")[0]["sample_p95"] is None


@pytest.mark.parametrize("samples", [
    ((NOW.replace(tzinfo=None), 1), (NOW, 2)),
    ((NOW, "NaN"), (NOW + timedelta(seconds=1), 2)),
    ((NOW, True), (NOW + timedelta(seconds=1), 2)),
    tuple((NOW + timedelta(seconds=i), i) for i in range(4097)),
])
def test_bad_samples_or_exceeded_budget_return_no_derived_values(samples):
    assert summarize_numeric_trend(samples) is None


def test_slope_budget_is_bounded_and_original_endpoints_are_retained():
    result = summarize_numeric_trend(tuple((NOW + timedelta(seconds=i), i) for i in range(4096)))
    assert result["sample_count"] == 4096
    assert result["representative_sample_count"] == 128
    assert result["first"] == 0 and result["latest"] == 4095
    assert result["trend_slope_per_day"] == 86400


def test_mutated_summary_or_row_cannot_be_republished_as_valid_investigation():
    fact = fact_from(monitoring())
    trace = fact.series_computations[0]
    forged = trace.model_copy(update={"summary": {**trace.summary, "change": 999}})
    with pytest.raises(ValueError, match="重放"):
        replay_series_computation(forged)
    changed = list(fact.rows[0])
    changed[[item["name"] for item in fact.columns].index("latest")] = 999
    with pytest.raises(ValueError, match="不一致"):
        validate_diagnosis(draft(fact), (fact.model_copy(update={"rows": (tuple(changed),)}),))


def test_capacity_forecast_never_joins_different_units_or_sources():
    for key, changed in [("unit", "kilobytes"), ("source_id", str(uuid7())), ("source_version", 2)]:
        artifact = monitoring(tuple(range(10, 20)))
        used = artifact["payload"]["observations"][0]
        used.update(metric_code="storage.used", semantic_role="STORAGE_USED", unit="bytes")
        maximum = deepcopy(used)
        maximum.update(metric_code="storage.max", semantic_role="STORAGE_MAX")
        maximum[key] = changed
        for point in maximum["series"][0]["points"]:
            point["value"] = 100
        artifact["payload"]["observations"].append(maximum)
        row = columns(fact_from(artifact, horizon=30))
        assert row["forecast_value"] is not None
        assert row["forecast_utilization_percent"] is None
        assert row["estimated_days_to_limit"] is None


@pytest.mark.parametrize("tamper", ["none", "source_value", "source_hash", "source_run", "trace_value", "trace_scope", "missing"])
def test_report_and_case_recheck_raw_source_not_only_consistent_calculation(tamper):
    artifact = monitoring()
    fact = fact_from(artifact)
    run = SimpleNamespace(target_id=artifact["payload"]["target_id"], ops_run_id=uuid7())
    stored = SimpleNamespace(payload_json=deepcopy(artifact["payload"]), ops_run_id=run.ops_run_id,
        schema_version="OBSERVATION_SET.v1", content_hash=canonical_hash(artifact["payload"]))
    raw_fact = fact.model_dump(mode="json")
    if tamper == "source_value":
        stored.payload_json["observations"][0]["series"][0]["points"][0]["value"] = 5
        stored.content_hash = canonical_hash(stored.payload_json)
    elif tamper == "source_hash":
        stored.content_hash = "bad"
    elif tamper == "source_run":
        stored.ops_run_id = uuid7()
    elif tamper == "trace_value":
        raw_fact["series_computations"][0]["samples"][0][1] = "999"
    elif tamper == "trace_scope":
        raw_fact["series_computations"][0]["unit"] = "other"
    uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(return_value=None if tamper == "missing" else stored)))
    assert asyncio.run(validate_monitoring_trace_sources(uow=uow, run=run, payload={"evidence": [raw_fact]})) == (tamper == "none")
