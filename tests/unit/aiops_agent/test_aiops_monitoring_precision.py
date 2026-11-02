"""真实 Prometheus 毫秒舍入、冻结窗口、判定与报告出版的回归。"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.adapters.diagnostic_sources.catalog import load_metric_catalog
from aiops_agent.adapters.diagnostic_sources.prometheus import PrometheusAdapter
from aiops_agent.application.inspections.assessment import assess_inspection, monitoring_checks
from aiops_agent.application.inspections.monitoring_observations import classify_monitoring_facts
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.application.monitoring_snapshot import MonitoringSnapshotBuilder
from aiops_agent.application.reporting import render_pdf, report_presentation, resolve_system_template
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.ports.diagnostic_source import DiagnosticSourceContext, MetricsEvidenceRequest
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from platform_core.identity import uuid7


class PrometheusResponse:
    status = 200

    def __init__(self, payload):
        self.content = self
        self.raw = json.dumps(payload).encode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def iter_chunked(self, size):
        for start in range(0, len(self.raw), size):
            yield self.raw[start:start + size]


class MillisecondPrometheus:
    """只替代 HTTP 来源，真实保留 Prometheus 毫秒时间栅格。"""

    def __init__(self, value, defect=None):
        self.value, self.defect, self.params = value, defect, None

    def get(self, url, **kwargs):
        self.params = kwargs["params"]
        start = int(Decimal(str(self.params["start"])) * 1000)
        end = int(Decimal(str(self.params["end"])) * 1000)
        step = self.params["step"] * 1000
        points = [[stamp / 1000, str(self.value)] for stamp in range(start, end + 1, step)]
        if self.defect == "outside":
            points.insert(0, [(start - 1) / 1000, str(self.value)])
        elif self.defect == "future":
            points.append([(end + 1000) / 1000, str(self.value)])
        elif self.defect == "invalid":
            points[0][1] = "NaN"
        elif self.defect == "stale":
            points = [point for point in points if point[0] <= end / 1000 - 360]
        return PrometheusResponse({"status": "success", "data": {"result": [
            {"metric": {"target_key": "mysql-test"}, "values": points},
        ]}})


async def pipeline(*, microsecond, metric_code, defect=None):
    now = datetime(2026, 11, 2, 12, 0, 14, microsecond, tzinfo=UTC)
    target_id, binding_id, source_id = (str(uuid7()) for _ in range(3))
    uow = SimpleNamespace(
        targets=SimpleNamespace(get_scoped=AsyncMock(return_value=SimpleNamespace(target_id=target_id)),
                                list_source_bindings=AsyncMock(return_value=[])),
        runs=SimpleNamespace(database_now=AsyncMock(return_value=now)),
    )
    manager = AsyncMock()
    manager.__aenter__.return_value = uow
    service = SimpleNamespace(
        _uow_factory=lambda: manager,
        _monitoring_snapshot_builder=MonitoringSnapshotBuilder(
            metric_catalog=load_metric_catalog(), default_window_seconds=3600, max_response_bytes=65536),
    )
    inspection = {"schedule_type": "DAILY", "period_start": (now - timedelta(days=1)).isoformat(),
                  "period_end": now.isoformat()}
    snapshot = await TurnPlanningService._prepare_monitoring(
        service, SimpleNamespace(inspection=inspection, target_id=target_id, domain_id=target_id, source_ids=()),
        requested_window_seconds=86400,
    )
    window = snapshot["current_window"]
    request = MetricsEvidenceRequest(
        target_id=target_id, binding_id=binding_id, source_locator_key="mysql-test",
        metric_definitions=(load_metric_catalog().get(metric_code),),
        window_start=window["start"], window_end=window["end"],
        requested_step_seconds=60, max_response_bytes=65536, trace_id="precision-test",
    )
    session = MillisecondPrometheus(1 if metric_code == "mysql.availability" else .46, defect)
    adapter = PrometheusAdapter(
        context=DiagnosticSourceContext(source_id=source_id, source_type="PROMETHEUS", adapter_id="prometheus",
            adapter_version="1.0.0", config_version=1, endpoint="http://example.test",
            declared_capabilities={"metric.query_range": {}}),
        session=session, request_timeout_seconds=10, webhook_replay_seconds=300,
    )
    result = await adapter.query_metrics(request)
    from aiops_agent.contracts.evidence import ObservationSet
    raw = ObservationSet(target_id=target_id, binding_id=binding_id, source_id=source_id,
                         observations=result.observations, gaps=result.gaps, collected_at=now)
    artifact_id = str(uuid7())
    evidence = DbaEvidenceAssessmentHandler._monitoring_fact(artifact_id=artifact_id, result=raw)
    frozen = AIOpsRuntimeService._inspection_monitoring_facts(SimpleNamespace(evidence=(evidence,)))
    facts = classify_monitoring_facts(frozen, monitoring=snapshot)
    policy = {"checks": [], "required_metric_roles": {"CPU_UTILIZATION": "CPU"} if metric_code != "mysql.availability" else {},
              "minimum_monitoring_coverage_ratio": .8}
    checks = monitoring_checks(facts, profile={"metric_thresholds": {
        "runtime.cpu.utilization": {"warning_value": 70, "critical_value": 85},
    }}, policy=policy)
    assessment = assess_inspection(checks, policy=policy)
    from aiops_agent.application.diagnosis.monitoring_calculations import validate_monitoring_trace_sources
    from aiops_agent.playbooks import canonical_hash
    source_payload = raw.model_dump(mode="json")
    assert await validate_monitoring_trace_sources(
        uow=SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(return_value=SimpleNamespace(
            ops_run_id=target_id, schema_version="OBSERVATION_SET.v1",
            payload_json=source_payload, content_hash=canonical_hash(source_payload))))),
        run=SimpleNamespace(target_id=target_id, ops_run_id=target_id),
        payload={"evidence": [evidence.model_dump(mode="json")]},
    )
    return now, snapshot, session, facts, checks, assessment


@pytest.mark.parametrize("microsecond", [540001, 540186, 540499, 540500, 540999, 999999])
@pytest.mark.parametrize("metric_code", ["mysql.availability", "runtime.cpu.utilization"])
def test_prometheus_rounding_does_not_reject_fresh_normal_snapshot(microsecond, metric_code):
    now, snapshot, session, facts, checks, assessment = asyncio.run(
        pipeline(microsecond=microsecond, metric_code=metric_code))
    assert assessment["health_status"] == "NORMAL"
    assert checks[0]["check_status"] == "CLEAR"
    assert not facts[0]["series_quality_flags"]
    left, right = (datetime.fromisoformat(snapshot["window"][key]) for key in ("start", "end"))
    assert left.microsecond % 1000 == right.microsecond % 1000 == 0
    assert now - timedelta(minutes=15) <= left < right <= now
    for key, value in (("start", left), ("end", right)):
        assert Decimal(str(session.params[key])) * 1000 == Decimal(str(value.timestamp())) * 1000
    payload = {"report_type": "INSPECTION_DAILY", "title": "数据库日常巡检报告",
               "period_start": (now - timedelta(days=1)).isoformat(), "period_end": now.isoformat(),
               "scope": {"inspection_assessment": assessment}, "facts": [*facts, *checks], "gaps": []}
    presentation = report_presentation(payload=payload, template=resolve_system_template("system:inspection.daily"))
    assert presentation["daily_inspection"]["conclusion"]["health_status"] == "NORMAL"
    assert render_pdf(presentation).startswith(b"%PDF-")


@pytest.mark.parametrize("defect", ["outside", "future", "invalid", "stale"])
def test_real_invalid_or_stale_source_samples_are_still_rejected(defect):
    *_, checks, assessment = asyncio.run(pipeline(
        microsecond=540186, metric_code="mysql.availability", defect=defect))
    assert checks[0]["check_status"] == "MISSING"
    assert assessment["health_status"] == "INCOMPLETE"
    assert "MySQL 可用性：MySQL 可用性" not in assessment["text"]
    assert any(term in assessment["limitations"][0] for term in ("窗口", "无效", "超过", "过期"))

@pytest.mark.parametrize("microsecond", [0, 1, 999, 1000, 540186, 999999])
def test_window_alignment_is_idempotent_and_never_expands_scope(microsecond):
    from aiops_agent.domain.evidence.metric_time import canonical_metric_window
    from aiops_agent.monitoring.view import resolve_monitoring_window, resolve_monitoring_baseline_window
    end = datetime(2026, 11, 2, 12, 0, 14, microsecond, tzinfo=UTC)
    start = end - timedelta(minutes=15)
    left, right = canonical_metric_window(start, end)
    assert start <= left < right <= end
    assert canonical_metric_window(left, right) == (left, right)
    assert (resolve_monitoring_window("15m", now=end).start,
            resolve_monitoring_window("15m", now=end).end) == (left, right)
    baseline = resolve_monitoring_baseline_window("1w", now=end)
    assert end - timedelta(days=7) <= baseline.start < baseline.end <= end


def test_window_alignment_handles_timezone_day_boundary_and_rejects_empty_interval():
    from datetime import timezone
    from aiops_agent.domain.evidence.metric_time import canonical_metric_window, prometheus_timestamp
    start = datetime(2026, 11, 1, 23, 59, 59, 999999, tzinfo=UTC)
    end = start + timedelta(seconds=1)
    left, right = canonical_metric_window(start, end)
    assert left == datetime(2026, 11, 2, tzinfo=UTC)
    assert canonical_metric_window(start.astimezone(timezone(timedelta(hours=8))), end) == (left, right)
    with pytest.raises(ValueError):
        canonical_metric_window(start, start + timedelta(microseconds=1))
    with pytest.raises(ValueError):
        canonical_metric_window(start.replace(tzinfo=None), end)
    assert prometheus_timestamp(datetime(1970, 1, 1, tzinfo=UTC) - timedelta(milliseconds=1)) == "-0.001000"


def test_backup_unverified_does_not_hide_successfully_completed_runtime_checks():
    from tests.unit.aiops_agent.test_aiops_professional_inspection_reports import full_assessment
    from aiops_agent.application.inspections.backup_observations import backup_policy_result
    policy, checks, _ = full_assessment()
    backup, gaps = backup_policy_result(
        policy={"enabled": True, "source_types": ["MYSQL_LOGICAL_DUMP"], "success_max_age_hours": 26},
        database_type="MYSQL", facts=[], evidence_by_tool={}, reference_time="2026-11-02T12:00:00Z",
    )
    result = assess_inspection([*checks, backup], policy=policy, gaps=gaps)
    assert result["health_status"] == "INCOMPLETE"
    assert result["completed_required_count"] == result["required_check_count"] - 1
    assert len(result["limitations"]) == 1
    assert "备份成功状态未核验" in result["limitations"][0]
    assert f"已完成的 {result['completed_required_count']} 项必要检查未发现异常" in result["text"]
    assert "。；" not in result["text"]
