"""OEM 关键指标面板与 7 天、30 天基线的真实应用链路回归。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.orm import Session

from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.monitoring_oem_overview import baseline_series, overview_panels
from aiops_agent.entities import ResourceSourceBindingEntity
from platform_core.contracts.aiops.monitoring_native import MonitoringNativeMetric, MonitoringNativeQuery
from tests.unit.aiops_agent.test_aiops_oem_native_metrics import adapter, collection, metadata, request, series, NOW
from tests.unit.aiops_agent.test_monitoring_oem_native import native_lab
from tests.unit.aiops_agent.test_business_system_publication_lifecycle import lab


def metric(**changes):
    return MonitoringNativeMetric(name="CPUUsageSummary/OverAllUsagePct", display_name="CPU",
        group="CPUUsageSummary", native_value_type="NUMBER", unit="%", dimensions=(),
        query_supported=True, latest_supported=True).model_copy(update=changes)


def test_profiles_are_small_curated_and_never_render_full_directory():
    panels = overview_panels("HOST", [metric(), metric(name="BootEnvironmentStatus/ActiveOnReboot")])
    assert len(panels) == 8
    cpu = next(panel for panel in panels if panel.panel_id == "oem.host.cpu")
    assert cpu.metric.name == "CPUUsageSummary/OverAllUsagePct" and cpu.baseline_supported
    assert all("BootEnvironment" not in (panel.metric.name if panel.metric else "") for panel in panels)
    assert all(panel.unavailable_reason for panel in panels if panel.metric is None)
    database = overview_panels("DATABASE", [])
    assert len(database) == 8 and all(panel.metric is None for panel in database)


@pytest.mark.parametrize("changes", [{"unit": "ratio"}, {"native_value_type": "STRING"}, {"dimensions": ("cpu",)}])
def test_overview_refuses_unreviewed_units_types_and_dimensions(changes):
    cpu = next(panel for panel in overview_panels("HOST", [metric(**changes)]) if panel.panel_id == "oem.host.cpu")
    assert cpu.metric is None and "不符" in cpu.unavailable_reason


def test_adapter_summary_uses_every_sample_beyond_presentation_limit_without_filling_gaps():
    datapoints = [[(NOW-timedelta(seconds=1300-index)).isoformat(), "1.25"] for index in range(1200)]
    value, _ = adapter({"targetName":"mes-db","targetType":"oracle_database"}, metadata(),
        collection([series(datapoints=datapoints)]))
    result = asyncio.run(value.query_native_metric(request(window_start=NOW-timedelta(hours=1)),
        metric_name="Filesystems/size", limit=1000, _paginate=True, _summary_only=True))
    assert not result["truncated"] and result["series"][0]["sample_count"] == 1200
    assert result["series"][0]["total"] == "1500.00"
    assert baseline_series(result)[0].average == "1.25"


def test_baseline_never_averages_different_dimensions_or_truncated_data():
    result = {"truncated":False, "series":[
        {"dimensions":{"mountPoint":"/"},"total":"30.5","sample_count":2},
        {"dimensions":{"mountPoint":"/data"},"total":"150","sample_count":3}]}
    assert [(row.average, row.sample_count) for row in baseline_series(result)] == [("15.25",2),("50",3)]
    assert baseline_series({**result, "truncated":True}) == ()


def baseline_lab(lab, *, truncated=False, fail=False, revoke=False):
    service, instance_id, calls = native_lab(lab)
    actual = service._diagnostic_source_registry.create(None)
    original = actual.query_native_metric
    recorded = []
    async def query(native_request, **options):
        if not options.get("_summary_only"):
            return await original(native_request, **options)
        recorded.append(native_request)
        if fail:
            raise DiagnosticSourceAdapterError("SOURCE_UNREACHABLE", "测试历史来源不可用")
        if revoke:
            with Session(lab.engine) as session:
                session.query(ResourceSourceBindingEntity).filter_by(resource_id=instance_id).one().status = "DISABLED"
                session.commit()
        return {"unit":"MB", "truncated":truncated, "series":[
            {"dimensions":{"mountPoint":"/"},"total":"27.375","sample_count":3}]}
    actual.query_native_metric = query
    return service, instance_id, recorded


@pytest.mark.parametrize("period,days", [("1w",7),("1m",30)])
def test_public_baseline_uses_real_historical_window_and_preserves_current_samples(lab,period,days):
    service, instance_id, recorded = baseline_lab(lab)
    result = asyncio.run(service.native.samples(scope=lab.scope,source_id=lab.source_id,instance_id=instance_id,
        query=MonitoringNativeQuery(metric_name="Filesystems/size",baseline=period)))
    assert len(recorded) == 1
    assert abs((recorded[0].window_end-recorded[0].window_start-timedelta(days=days)).total_seconds()) < 0.002
    assert abs((recorded[0].window_end-datetime.now(UTC)).total_seconds()) < 10
    assert result.points[0].value == "90071992547409931"
    assert result.baseline.series[0].average == "9.125" and result.baseline.series[0].sample_count == 3
    assert result.baseline.aggregation == "SAMPLE_MEAN" and "覆盖率未知" in result.baseline.note


@pytest.mark.parametrize("truncated,fail", [(True,False),(False,True)])
def test_baseline_failure_or_truncation_does_not_destroy_current_samples(lab,truncated,fail):
    service,instance_id,_ = baseline_lab(lab,truncated=truncated,fail=fail)
    result = asyncio.run(service.native.samples(scope=lab.scope,source_id=lab.source_id,instance_id=instance_id,
        query=MonitoringNativeQuery(metric_name="Filesystems/size",baseline="1w")))
    assert result.points and result.baseline.series == () and result.baseline.unavailable_reason


def test_binding_revoked_during_baseline_discards_the_whole_result(lab):
    service,instance_id,_ = baseline_lab(lab,revoke=True)
    with pytest.raises(AIOpsApplicationError):
        asyncio.run(service.native.samples(scope=lab.scope,source_id=lab.source_id,instance_id=instance_id,
            query=MonitoringNativeQuery(metric_name="Filesystems/size",baseline="1m")))


def test_overview_does_not_compute_status_code_average():
    state = metric(name="Response/Status",unit="n/a",group="Response")
    panel = next(panel for panel in overview_panels("HOST", [state]) if panel.panel_id == "oem.host.status")
    assert panel.mode == "LATEST" and not panel.baseline_supported


def test_cached_baseline_is_reused_but_binding_revocation_still_blocks_every_read(lab):
    service,instance_id,recorded = baseline_lab(lab)
    query = MonitoringNativeQuery(metric_name="Filesystems/size",baseline="1w")
    first = asyncio.run(service.native.samples(scope=lab.scope,source_id=lab.source_id,instance_id=instance_id,query=query))
    actual = service._diagnostic_source_registry.create(None)
    previous = actual.query_native_metric
    async def replay(request, **options):
        if options.get("_summary_only"):
            return await previous(request, **options)
        return SimpleNamespace(unit="MB", sample_dimensions=tuple(point.dimensions for point in first.points),
            points=tuple((point.observed_at,point.value) for point in first.points),captured_at=first.captured_at,
            truncated=False,response_sha256=first.response_sha256)
    actual.query_native_metric = replay
    second = asyncio.run(service.native.samples(scope=lab.scope,source_id=lab.source_id,instance_id=instance_id,query=query))
    assert second.baseline == first.baseline and len(recorded) == 1
    entries = next(iter(service.native._catalog_cache.values()))[2]
    service.native._catalog = AsyncMock(return_value=entries)
    with Session(lab.engine) as session:
        binding = session.query(ResourceSourceBindingEntity).filter_by(resource_id=instance_id).one()
        binding.row_version += 1
        session.commit()
    asyncio.run(service.native.samples(scope=lab.scope,source_id=lab.source_id,instance_id=instance_id,query=query))
    assert len(recorded) == 2
    with Session(lab.engine) as session:
        session.query(ResourceSourceBindingEntity).filter_by(resource_id=instance_id).one().status = "DISABLED"
        session.commit()
    with pytest.raises(AIOpsApplicationError):
        asyncio.run(service.native.samples(scope=lab.scope,source_id=lab.source_id,instance_id=instance_id,query=query))
    assert len(recorded) == 2
