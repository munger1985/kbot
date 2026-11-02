"""七天参考比较的口径、零基线、缺测及与健康结论隔离的回归。"""

import asyncio
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.inspections.daily_baseline import (
    daily_baseline_comparison, inspection_reference_window, observation_coverage,
)
from aiops_agent.application.inspections.assessment import monitoring_checks
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.application.reporting import report_presentation, resolve_system_template, render_pdf
from tests.unit.aiops_agent.test_aiops_professional_inspection_reports import full_assessment, PROFILE


END = datetime(2026, 11, 2, 2, tzinfo=UTC)
PERIOD_START = END - timedelta(days=1)


def metric(**updates):
    return {
        "kind": "inspection_monitoring_metric", "metric_code": "runtime.cpu.utilization",
        "semantic_role": "CPU_UTILIZATION", "value_kind": "GAUGE", "unit": "percent",
        "dimensions": "instance=sample", "latest": 10, "average": 5, "maximum": 10,
        "window_start": (END - timedelta(days=7)).isoformat(), "window_end": END.isoformat(),
        "coverage_ratio": 1, "sample_count": 168, "evidence_ref": "frozen-sample",
        "daily_points": [{"date": (END - timedelta(days=day)).date().isoformat(), "value": 5} for day in range(7, 0, -1)],
        **updates,
    }


def sampled_metric():
    start = PERIOD_START - timedelta(days=7)
    return metric(
        latest=99, average=999, window_start=start.isoformat(),
        sample_points=[{"observed_at": (start + timedelta(hours=hour)).isoformat(),
                        "value": 10 if hour < 168 else 20 if hour < 192 else 99}
                       for hour in range(193)],
    )


def compare(fact):
    return daily_baseline_comparison(
        fact, observation_start=PERIOD_START.isoformat(), observation_end=END.isoformat(), timezone="Asia/Shanghai",
    )


def test_frozen_seven_day_summary_explicitly_compares_latest_sample():
    result = compare(metric())
    assert result["baseline_status"] == "AVAILABLE"
    assert result["comparison_basis"] == "LATEST_SAMPLE"
    assert result["comparison_label"] == "最新样本"
    assert result["baseline_average"] == 5
    assert result["deviation"] == 5
    assert result["deviation_percent"] == 100
    assert "+5.00 个百分点" in result["change_display"]


@pytest.mark.parametrize("value", [0, 10])
def test_zero_baseline_keeps_absolute_change_without_infinite_relative_rate(value):
    result = compare(metric(average=0, latest=value))
    assert result["baseline_status"] == "AVAILABLE"
    assert result["baseline_average"] == 0
    assert result["deviation"] == value
    assert result["deviation_percent"] is None
    assert "基线为零" in result["change_display"]


def test_observation_mean_uses_disjoint_previous_seven_days_and_half_open_window():
    fact = sampled_metric()
    original = deepcopy(fact)
    result = compare(fact)
    assert result["comparison_basis"] == "OBSERVATION_MEAN"
    assert result["comparison_value"] == 20
    assert result["baseline_average"] == 10
    assert result["deviation_percent"] == 100
    assert all(point["value"] == 10 for point in result["trend_points"])
    assert fact == original


@pytest.mark.parametrize("kind,unit", [("STATE", "state"), ("STATE", "percent"), ("COUNTER", "count")])
def test_state_and_raw_counter_are_not_ordinary_mean_comparisons(kind, unit):
    result = compare(metric(value_kind=kind, unit=unit))
    assert result["baseline_applicable"] is False
    assert result["baseline_status"] == "INSUFFICIENT"


@pytest.mark.parametrize("updates", [
    {"window_start": (END - timedelta(days=1)).isoformat()},
    {"window_start": (END - timedelta(days=30)).isoformat()},
    {"window_start": "2026-10-26T02:00:00"}, {"window_start": None},
    {"coverage_ratio": .79}, {"coverage_ratio": 1.1}, {"coverage_ratio": None},
    {"latest": float("nan")}, {"latest": float("inf")}, {"latest": True},
    {"average": float("nan")}, {"average": float("inf")},
    {"average": 10**1000}, {"latest": 10**1000},
    {"series_quality_flags": ["COVERAGE_UNKNOWN"]}, {"series_quality_flags": ["TRUNCATED"]},
])
def test_invalid_or_non_seven_day_summary_does_not_fabricate_comparison(updates):
    assert compare(metric(**updates))["baseline_status"] == "INSUFFICIENT"


def test_missing_day_is_not_hidden_by_overall_eighty_percent_coverage():
    fact = sampled_metric()
    missing = PERIOD_START - timedelta(days=3)
    fact["sample_points"] = [point for point in fact["sample_points"]
                             if not missing <= datetime.fromisoformat(point["observed_at"]) < missing + timedelta(days=1)]
    result = compare(fact)
    assert result["baseline_status"] == "INSUFFICIENT"
    assert "缺测" in result["baseline_reason"]


def test_duplicate_samples_are_deduplicated_but_conflicting_values_are_rejected():
    fact = sampled_metric()
    fact["sample_points"].append(dict(fact["sample_points"][0]))
    assert compare(fact)["baseline_average"] == 10
    fact["sample_points"][-1]["value"] = 500
    result = compare(fact)
    assert result["baseline_status"] == "INSUFFICIENT"
    assert "冲突" in result["baseline_reason"]


def test_missing_old_history_does_not_invalidate_complete_current_checks():
    fact = sampled_metric()
    fact["latest"], fact["coverage_ratio"] = 10, .125
    fact["sample_points"] = [point for point in fact["sample_points"] if datetime.fromisoformat(point["observed_at"]) >= PERIOD_START]
    fact["observation_coverage_ratio"] = observation_coverage(
        fact, observation_start=PERIOD_START, observation_end=END,
    )
    assert fact["observation_coverage_ratio"] == 1
    policy, _, _ = full_assessment()
    checks = monitoring_checks([fact], profile=PROFILE, policy=policy)
    assert checks[0]["check_status"] == "CLEAR"
    assert compare(fact)["baseline_status"] == "INSUFFICIENT"


def test_large_relative_shift_does_not_override_frozen_normal_health_or_input():
    policy, checks, assessment = full_assessment()
    payload = {"report_type": "INSPECTION_DAILY", "period_start": PERIOD_START.isoformat(), "period_end": END.isoformat(),
               "scope": {"inspection_assessment": assessment, "report_policy": policy},
               "facts": [*checks, metric(average=.01)], "gaps": [], "evidence_refs": []}
    original = deepcopy(payload)
    presentation = report_presentation(payload=payload, template=resolve_system_template("system:inspection.daily"))
    daily = presentation["daily_inspection"]
    assert daily["metrics"][0]["deviation_percent"] == 99900
    assert daily["conclusion"]["health_status"] == "NORMAL"
    assert payload == original
    assert render_pdf(presentation).startswith(b"%PDF-")


def test_inspection_monitoring_reads_baseline_but_preserves_report_window():
    async def run():
        inspection = {"schedule_type": "DAILY", "period_start": PERIOD_START.isoformat(), "period_end": END.isoformat()}
        frozen = deepcopy(inspection)
        builder = SimpleNamespace(build=AsyncMock(return_value={"bindings": [], "initial_gaps": []}))
        uow = SimpleNamespace(targets=SimpleNamespace(get_scoped=AsyncMock(return_value=SimpleNamespace())),
                              runs=SimpleNamespace(database_now=AsyncMock(return_value=END + timedelta(seconds=45))))
        manager = AsyncMock()
        manager.__aenter__.return_value = uow
        service = SimpleNamespace(_monitoring_snapshot_builder=builder, _uow_factory=lambda: manager)
        context = SimpleNamespace(inspection=inspection, target_id="sample", domain_id="sample", source_ids=())
        await TurnPlanningService._prepare_monitoring(service, context, requested_window_seconds=86400)
        kwargs = builder.build.await_args.kwargs
        assert kwargs["window_start"] == PERIOD_START - timedelta(days=7)
        assert kwargs["window_end"] == END
        assert inspection == frozen
    asyncio.run(run())


def test_baseline_read_respects_source_query_window_limit_and_excludes_weekly():
    inspection = {"schedule_type": "DAILY", "period_start": PERIOD_START.isoformat(), "period_end": END.isoformat()}
    assert inspection_reference_window(inspection, maximum_seconds=86400) is None
    assert inspection_reference_window({**inspection, "schedule_type": "WEEKLY"}, maximum_seconds=60 * 86400) is None


def test_tiny_baseline_preserves_absolute_change_without_serializing_infinity():
    result = compare(metric(average=1e-310))
    assert result["baseline_status"] == "AVAILABLE"
    assert result["deviation_percent"] is None
    assert "超出可表示范围" in result["change_display"]
