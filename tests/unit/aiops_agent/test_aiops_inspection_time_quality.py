"""日巡检当前快照、历史参考、空指标覆盖及真实备份来源的回归。"""

import asyncio
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.adapters.diagnostic_sources.catalog import load_metric_catalog
from aiops_agent.application.inspections.assessment import monitoring_checks, assess_inspection
from aiops_agent.application.inspections.backup_observations import backup_policy_result
from aiops_agent.application.inspections.monitoring_observations import classify_monitoring_facts, monitoring_quality
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.application.reporting import report_presentation, resolve_system_template, render_pdf
from aiops_agent.contracts.evidence import MetricObservation, MetricSeries, MetricPoint, ObservationGap
from aiops_agent.contracts.turn_answer import AIOpsTurnResult
from aiops_agent.ports.diagnostic_source import MetricsEvidenceResult
from aiops_agent.workers.evidence_handlers import EvidenceObserveHandler
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from platform_core.identity import uuid7


NOW = datetime(2026, 11, 2, 10, 30, tzinfo=UTC)
CURRENT = {"start": (NOW - timedelta(minutes=15)).isoformat(), "end": NOW.isoformat()}
REFERENCE = {"start": (NOW - timedelta(days=8)).isoformat(), "end": NOW.isoformat()}
MONITORING = {"window": CURRENT, "current_window": CURRENT, "reference_window": REFERENCE,
              "snapshot_max_age_seconds": 300}
POLICY = {"version": "test", "checks": [], "required_metric_roles": {"CPU_UTILIZATION": "CPU"},
          "minimum_monitoring_coverage_ratio": .8}
PROFILE = {"metric_thresholds": {"runtime.cpu.utilization": {"warning_value": 70, "critical_value": 85}}}


def metric(**updates):
    return {"kind": "inspection_monitoring_metric", "metric_code": "runtime.cpu.utilization",
            "semantic_role": "CPU_UTILIZATION", "unit": "percent", "value_kind": "GAUGE",
            "latest": 10, "average": 10, "maximum": 10, "dimensions": "-", "evidence_ref": "metric",
            "window_start": CURRENT["start"], "window_end": CURRENT["end"], "coverage_ratio": .06,
            "sample_points": [{"observed_at": NOW.isoformat(), "value": 10}],
            **updates}


def checked(facts):
    classified = classify_monitoring_facts(facts, monitoring=MONITORING)
    return monitoring_checks(classified, profile=PROFILE, policy=POLICY)


def test_just_connected_fresh_snapshot_can_be_normal_without_history():
    result = assess_inspection(checked([metric()]), policy=POLICY)
    assert result["health_status"] == "NORMAL"
    assert "当前快照" in result["checks"][0]["basis"]


def test_old_history_cannot_replace_missing_current_samples_or_override_current_health():
    old = metric(window_start=REFERENCE["start"], latest=99)
    assert checked([old])[0]["check_status"] == "MISSING"
    result = checked([old, metric()])
    assert len(result) == 1
    assert result[0]["check_status"] == "CLEAR"


@pytest.mark.parametrize("updates", [
    {"sample_points": []},
    {"sample_points": [{"observed_at": (NOW - timedelta(minutes=6)).isoformat(), "value": 10}]},
    {"sample_points": [{"observed_at": (NOW + timedelta(seconds=1)).isoformat(), "value": 10}]},
    {"sample_points": [{"observed_at": NOW.isoformat(), "value": None}]},
    {"series_quality_flags": ["TRUNCATED"]},
    {"series_quality_flags": ["COVERAGE_UNKNOWN"]},
])
def test_invalid_or_stale_snapshot_cannot_be_normal(updates):
    assert checked([metric(**updates)])[0]["check_status"] == "MISSING"


def test_missing_metrics_are_counted_and_reference_gaps_do_not_lower_current_sampling():
    facts = classify_monitoring_facts([metric(coverage_ratio=1)], monitoring=MONITORING)
    quality = monitoring_quality(facts, monitoring={"bindings": [{"metrics": [
        {"metric_code": "runtime.cpu.utilization"}, {"metric_code": "host.memory.utilization"},
    ]}]})
    assert quality == {"coverage_percent": 50, "metric_count": 2, "low_coverage_metric_count": 1}
    assert monitoring_quality(facts, gaps=[{"step_id": "runtime.cpu.utilization", "code": "REFERENCE_SOURCE_NO_DATA"}])["coverage_percent"] == 100
    assert monitoring_quality(facts, gaps=[{"step_id": "runtime.cpu.utilization", "code": "SOURCE_NO_DATA"}])["coverage_percent"] == 0


def test_requested_unsupported_metrics_are_not_removed_from_sampling_denominator():
    facts = classify_monitoring_facts([metric(coverage_ratio=1)], monitoring=MONITORING)
    quality = monitoring_quality(facts, monitoring={"bindings": [{
        "metrics": [{"metric_code": "runtime.cpu.utilization"}],
        "unsupported_metrics": ["host.memory.utilization"],
    }]})
    assert quality == {"coverage_percent": 50, "metric_count": 2, "low_coverage_metric_count": 1}


def test_old_report_facts_are_not_mutated():
    original = metric()
    frozen = deepcopy(original)
    classify_monitoring_facts([original], monitoring=MONITORING)
    assert original == frozen
    assert classify_monitoring_facts([original], monitoring={}) == (original,)


def backup_metric(source="MYSQL_LOGICAL_DUMP", hours=1, **updates):
    return {**metric(), "metric_code": "backup.last_success.timestamp",
            "semantic_role": "BACKUP_LAST_SUCCESS_TIMESTAMP", "unit": "unix_seconds",
            "value_kind": "STATE", "latest": (NOW - timedelta(hours=hours)).timestamp(),
            "metric_dimensions": {"backup_source_type": source}, **updates}


def backup_result(facts, sources=("MYSQL_LOGICAL_DUMP",), db_type="MYSQL"):
    return backup_policy_result(
        policy={"enabled": True, "source_types": list(sources), "success_max_age_hours": 26},
        database_type=db_type,
        facts=classify_monitoring_facts(facts, monitoring=MONITORING),
        evidence_by_tool={}, reference_time=NOW.isoformat(),
    )


@pytest.mark.parametrize("source,db_type", [
    ("MYSQL_LOGICAL_DUMP", "MYSQL"), ("MYSQL_XTRABACKUP", "MYSQL"),
    ("POSTGRESQL_PGBACKREST", "POSTGRESQL"), ("ORACLE_RMAN", "ORACLE"),
])
def test_real_backup_success_is_supported_for_each_engine(source, db_type):
    fact, gaps = backup_result([backup_metric(source)], sources=(source,), db_type=db_type)
    assert fact["check_status"] == "COMPLIANT"
    assert not gaps
    assert fact["evidence_refs"] == ["metric"]


@pytest.mark.parametrize("facts", [
    [backup_metric(hours=-1)], [backup_metric(latest=float("nan"))],
    [backup_metric(evidence_ref="")],
    [backup_metric(sample_points=[])],
])
def test_future_invalid_or_untraceable_backup_remains_a_technical_limitation(facts):
    fact, gaps = backup_result(facts)
    assert fact["check_status"] == "NOT_OBSERVABLE"
    assert gaps[0]["code"] == "BACKUP_VERIFICATION_FAILED"
    assert "核验未完成" in fact["summary"]


@pytest.mark.parametrize("facts", [[], [backup_metric("MYSQL_XTRABACKUP")], [backup_metric(latest=0)]])
def test_missing_backup_success_is_a_completed_policy_risk(facts):
    fact, gaps = backup_result(facts)
    assert fact["check_status"] == "FINDING"
    assert not gaps
    result = assess_inspection([fact], policy={})
    assert result["completion_status"] == "COMPLETE"
    assert result["health_status"] == "ATTENTION"
    assert result["completed_required_count"] == result["required_check_count"] == 1
    assert not result["limitations"]
    assert "未提供有效备份成功记录" in result["findings"][0]["summary"]


def test_backup_overdue_is_real_finding_not_missing_data():
    fact, gaps = backup_result([backup_metric(hours=27)])
    assert fact["check_status"] == "BREACHED"
    assert not gaps
    assessment = assess_inspection([fact], policy={})
    assert assessment["health_status"] == "ABNORMAL"
    assert assessment["completion_status"] == "COMPLETE"


def test_latest_failed_job_cannot_be_hidden_by_recent_success():
    status = backup_metric(metric_code="backup.last_run.status", semantic_role="BACKUP_LAST_RUN_STATUS", unit="state", latest=0)
    fact, gaps = backup_result([backup_metric(), status])
    assert fact["check_status"] == "BREACHED"
    assert not gaps
    assert assess_inspection([fact], policy={})["completion_status"] == "COMPLETE"


def test_one_success_cannot_cover_another_required_source():
    fact, _ = backup_result([backup_metric()], sources=("MYSQL_LOGICAL_DUMP", "MYSQL_XTRABACKUP"))
    assert fact["check_status"] == "FINDING"
    assert fact["missing_success_source_types"] == ["MYSQL_XTRABACKUP"]


def test_known_backup_failure_survives_missing_another_source():
    status = backup_metric(metric_code="backup.last_run.status", semantic_role="BACKUP_LAST_RUN_STATUS", unit="state", latest=0)
    fact, gaps = backup_result([backup_metric(), status], sources=("MYSQL_LOGICAL_DUMP", "MYSQL_XTRABACKUP"))
    assert not gaps
    assessment = assess_inspection([fact], policy={})
    assert assessment["health_status"] == "ABNORMAL"
    assert assessment["completion_status"] == "COMPLETE"


def test_binlog_success_without_backup_job_records_is_a_policy_risk():
    source = AIOpsTurnResult.model_validate({
        "status": "COMPLETED", "sufficiency_status": "ANSWERABLE",
        "blocks": [{"block_type": "MARKDOWN", "schema_version": "AIOPS_MARKDOWN_BLOCK.v1",
                    "payload": {"markdown": "检查完成。"}, "evidence_refs": []}],
    })
    facts, gaps, _ = AIOpsRuntimeService._inspection_report_projection(
        inspection={"evidence_steps": [], "period_end": NOW.isoformat(),
                    "backup_policy": {"enabled": True, "success_max_age_hours": 26, "source_types": ["MYSQL_LOGICAL_DUMP"]}},
        source=source, database_type="MYSQL", action_tool_ids={}, monitoring=MONITORING,
    )
    backup = next(f for f in facts if f.get("kind") == "backup_policy_check")
    assert backup["check_status"] == "FINDING"
    assert not any(g["step_id"] == "inspection.backup-policy" for g in gaps)


@pytest.mark.parametrize("history_failure", [False, True])
@pytest.mark.parametrize("metric_code", ["runtime.cpu.utilization", "backup.last_success.timestamp"])
def test_real_observation_pipeline_separates_current_from_reference(history_failure, metric_code):
    async def run():
        target, binding, source = str(uuid7()), str(uuid7()), str(uuid7())
        definition = load_metric_catalog().get(metric_code)
        calls = []

        async def query(request):
            calls.append(request)
            if request.window_start != datetime.fromisoformat(CURRENT["start"]):
                if history_failure:
                    raise RuntimeError("历史来源不可用")
                return MetricsEvidenceResult(gaps=(ObservationGap(metric_code=definition.metric_code,
                    code="SOURCE_NO_DATA", detail="历史暂无样本"),))
            observation = MetricObservation(
                metric_code=definition.metric_code, semantic_role=definition.semantic_role,
                semantic_version="1.0.0", unit=definition.unit, value_kind=definition.value_kind,
                window_start=request.window_start, window_end=request.window_end,
                requested_step_seconds=60, effective_step_seconds=60,
                source_id=source, source_type="PROMETHEUS", source_version=1,
                target_id=target, binding_id=binding, external_target_fingerprint="sample",
                series=(MetricSeries(
                    dimensions={"backup_source_type": "MYSQL_LOGICAL_DUMP"} if metric_code.startswith("backup.") else {},
                    points=(MetricPoint(observed_at=NOW, value=(NOW - timedelta(hours=1)).timestamp() if metric_code.startswith("backup.") else 10),),
                ),),
                expected_points=16, actual_points=1, coverage_ratio=1/16,
            )
            return MetricsEvidenceResult(observations=(observation,))

        adapter = SimpleNamespace(query_metrics=AsyncMock(side_effect=query))
        snapshot = {"binding_id": str(binding), "effective_capabilities": ["metric.query_range"], "unsupported_metrics": [],
                    "source_locator_key": "database", "source_locator": {}, "metrics": [definition.model_dump(mode="json")],
                    "source": {"source_id": str(source), "source_type": "PROMETHEUS", "adapter_id": "prometheus",
                               "adapter_version": "1.0.0", "config_version": 1, "endpoint": "http://example.test",
                               "declared_capabilities": {"metric.query_range": {}}, "config": {}}}
        monitoring = {**MONITORING, "bindings": [snapshot], "max_response_bytes": 65536}
        context = SimpleNamespace(task_key=f"observe:{binding}", target_id=target, trace_id="test",
                                  plan_snapshot={"monitoring": monitoring})
        result = await EvidenceObserveHandler(
            diagnostic_source_registry=SimpleNamespace(create=lambda *args, **kwargs: adapter),
            secret_store=object(),
        ).execute(context)
        assert len(calls) == 2
        assert calls[0].window_end == NOW
        assert calls[1].window_start == NOW - timedelta(days=8)
        assert all(request.max_response_bytes == 32768 for request in calls)
        assert result.gaps and all(gap.code.startswith("REFERENCE_") for gap in result.gaps)
        evidence = DbaEvidenceAssessmentHandler._monitoring_fact(artifact_id=str(uuid7()), result=result)
        frozen = AIOpsRuntimeService._inspection_monitoring_facts(SimpleNamespace(evidence=(evidence,)))
        if metric_code.startswith("backup."):
            published, _, _ = AIOpsRuntimeService._inspection_report_projection(
                inspection={"evidence_steps": [], "period_end": NOW.isoformat(),
                            "backup_policy": {"enabled": True, "success_max_age_hours": 26,
                                              "source_types": ["MYSQL_LOGICAL_DUMP"]}},
                source=SimpleNamespace(evidence=(evidence,), evidence_gaps=()),
                database_type="MYSQL", action_tool_ids={}, monitoring=monitoring,
            )
            fact = next(item for item in published if item.get("kind") == "backup_policy_check")
            assert fact["check_status"] == "COMPLIANT"
        else:
            assert checked(frozen)[0]["check_status"] == "CLEAR"
    asyncio.run(run())


def test_daily_presentation_keeps_fresh_current_value_with_missing_history_and_exports_pdf():
    fresh = classify_monitoring_facts([metric()], monitoring=MONITORING)[0]
    assessment = assess_inspection(checked([metric()]), policy=POLICY)
    payload = {"title": "日常巡检报告", "report_type": "INSPECTION_DAILY",
               "period_start": (NOW - timedelta(days=1)).isoformat(), "period_end": NOW.isoformat(),
               "scope": {"inspection_assessment": assessment}, "facts": [fresh], "gaps": []}
    original = deepcopy(payload)
    result = report_presentation(payload=payload, template=resolve_system_template("system:inspection.daily"))
    daily = result["daily_inspection"]
    assert daily["metrics"][0]["current"] == 10
    assert daily["metrics"][0]["baseline_status"] == "INSUFFICIENT"
    assert daily["conclusion"]["health_status"] == "NORMAL"
    assert render_pdf(result).startswith(b"%PDF-")
    assert payload == original
