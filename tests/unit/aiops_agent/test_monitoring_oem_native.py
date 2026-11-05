"""OEM 原生监控的目录全量、采样、受控分页及资源授权回归。"""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.orm import Session

from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.adapters.diagnostic_sources.oem_catalog import catalog_entries, read_pages
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.entities import DiagnosticSourceEntity, ResourceSourceBindingEntity
from platform_core.contracts.aiops.monitoring_native import MonitoringNativeQuery
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_oem_native_metrics import adapter, collection, metadata, request, series, GUID
from tests.unit.aiops_agent.test_monitoring_business_system_catalog import service_for
from tests.unit.aiops_agent.test_resource_monitoring import configure_oem_host
from tests.unit.aiops_agent.test_business_system_publication_lifecycle import lab, publish


def test_catalog_rejects_encoded_metric_name_beyond_public_query_budget():
    with pytest.raises(DiagnosticSourceAdapterError) as caught:
        catalog_entries([metadata(name="指标组" * 10)])
    assert caught.value.code == "SOURCE_RESPONSE_INVALID"


def test_full_catalog_contains_more_than_diagnostic_limit_and_reads_safe_next_page():
    groups = [metadata(name=f"Group{index}") for index in range(64)]
    value, calls = adapter({"targetName": "host", "targetType": "host"},
        collection(groups, links={"next": {"href": f"/em/api/targets/{GUID}/metricGroups?page=abc%2B%3D"}}),
        collection([metadata(name="Extra")]))
    entries, truncated, digest = asyncio.run(value.discover_monitoring_metrics(request(True, scope="HOST")))
    assert len(entries) == 130 and not truncated and len(digest) == 64
    assert calls[-1][1]["params"] == {"page": "abc+="}
    assert all(not options["allow_redirects"] for _, options in calls)


@pytest.mark.parametrize("href", ["https://evil.invalid/em/api/metricTimeSeries?page=x", "/em/api/targets/other?page=x",
    "/em/api/metricTimeSeries?page=a&page=b", "/em/api/metricTimeSeries?page=x&targetId=other", "/em/api/metricTimeSeries?page=%0A"])
def test_native_pagination_rejects_external_or_changed_query(href):
    value, calls = adapter(collection([], links={"next": {"href": href}}))
    with pytest.raises(DiagnosticSourceAdapterError, match="OEM"):
        asyncio.run(read_pages(value, "/metricTimeSeries", params={}, max_bytes=262144))
    assert len(calls) == 1


def test_native_series_pagination_preserves_dimensions_and_numeric_json_precision():
    value, calls = adapter({"targetName": "mes-db", "targetType": "oracle_database"}, metadata(),
        collection([series("/")], links={"next": {"href": "/em/api/metricTimeSeries?page=next"}}),
        collection([series("/data", datapoints=[["2026-10-28T06:50:00.000Z", 1.25]])]))
    result = asyncio.run(value.query_native_metric(request(), metric_name="Filesystems/size", limit=1000, _paginate=True))
    assert tuple(point[1] for point in result.points) == ("90071992547409931", "1.25")
    assert result.sample_dimensions == ({"mountPoint": "/"}, {"mountPoint": "/data"})
    assert not result.truncated and calls[-1][1]["params"] == {"page": "next"}
    assert value._json(b'{"value":0.12345678901234567890123}')["value"] == "0.12345678901234567890123"


def test_pagination_accepts_only_unchanged_frozen_filters_with_token():
    value, calls = adapter(collection([], links={"next":{"href":"/em/api/metricTimeSeries?targetId="+GUID+"&limit=10&page=next"}}), collection([]))
    result = asyncio.run(read_pages(value,"/metricTimeSeries",params={"targetId":GUID,"limit":10},max_bytes=262144))
    assert not result[1] and calls[-1][1]["params"] == {"targetId":GUID,"limit":10,"page":"next"}


@pytest.mark.parametrize("native_type,value", [("STRING", "OPEN & READ WRITE"), ("STRING", None), ("NUMBER", 1.25)])
def test_latest_snapshot_preserves_text_null_and_decimal_without_fabricating_history(native_type, value):
    meta = metadata(name="Response", keys=[], metrics=[{"name":"State","dataType":native_type}])
    value_adapter, calls = adapter({"targetName":"mes-db","targetType":"oracle_database"}, meta,
        collection([{"State":value}], targetName="mes-db",targetTypeName="oracle_database",targetId=GUID,
            metricGroupName="Response",timeCollected="2026-10-28T06:50:00Z"))
    result = asyncio.run(value_adapter.query_latest_metric(request(),metric_name="Response/State"))
    assert result["points"][0][1] == ("1.25" if native_type == "NUMBER" else value)
    assert calls[-1][0].endswith("/metricGroups/Response/latestData")
    assert calls[-1][1]["params"] == {"metricName":"State","limit":100}


@pytest.mark.parametrize("scope,kind", [("DATABASE", "host"), ("HOST", "oracle_database")])
def test_catalog_never_substitutes_host_and_database_identity(scope, kind):
    value, calls = adapter({"targetName": "resource", "targetType": kind})
    with pytest.raises(DiagnosticSourceAdapterError) as caught:
        asyncio.run(value.discover_monitoring_metrics(request(True, scope=scope)))
    assert caught.value.code == "SOURCE_TARGET_MISMATCH" and len(calls) == 1


def native_lab(lab):
    configure_oem_host(lab)
    detail = asyncio.run(publish(lab))
    host = detail.resources[1].resource
    with Session(lab.engine) as session:
        source = session.get(DiagnosticSourceEntity, lab.source_id)
        source.declared_capabilities_json = {"metric.discover": {}, "metric.query_range": {}}
        session.commit()
    service = service_for(lab)
    now = datetime.now(UTC)-timedelta(minutes=1)
    meta = metadata(name="Filesystems", metrics=[{"name": "size", "dataType": "NUMBER", "unitDisplayName": "MB"}])
    value, calls = adapter({"targetName": "mes-db", "targetType": "host"}, collection([meta]),
        {"targetName": "mes-db", "targetType": "host"}, meta,
        collection([series(targetTypeName="host", targetId="4"*32,
            datapoints=[[now.isoformat(), "90071992547409931"]])]))
    # 保留实际 Adapter，只替换外部 HTTP 传输及测试来源上下文。
    value.context = value.context.model_copy(update={"declared_capabilities": {"metric.discover": {}, "metric.query_range": {}}})
    service._diagnostic_source_registry = SimpleNamespace(create=lambda context, **kwargs: value)
    return service, host.resource_id, calls


def test_monitoring_service_reads_real_adapter_catalog_and_nonzero_samples(lab):
    service, instance_id, calls = native_lab(lab)
    catalog = asyncio.run(service.native.catalog(scope=lab.scope, source_id=lab.source_id, instance_id=instance_id))
    assert catalog.discovered_count == 1 and catalog.entries[0].name == "Filesystems/size"
    result = asyncio.run(service.native.samples(scope=lab.scope, source_id=lab.source_id,
        instance_id=instance_id, query=MonitoringNativeQuery(metric_name="Filesystems/size")))
    assert result.points[0].value == "90071992547409931"
    assert result.points[0].dimensions == {"mountPoint": "/"} and not result.truncated
    payload = result.model_dump_json()
    assert all(secret not in payload for secret in ("source_locator", "password", "endpoint", "credential"))


def test_unauthorized_resource_never_calls_provider(lab):
    service, instance_id, calls = native_lab(lab)
    with pytest.raises(AIOpsApplicationError):
        asyncio.run(service.native.catalog(scope=lab.scope, source_id=lab.source_id, instance_id=uuid7()))
    assert not calls
    other = replace(lab.scope, domain_id=uuid7() if hasattr(lab.scope, "tenant_id") else lab.scope.domain_id + 1)
    with pytest.raises(AIOpsApplicationError):
        asyncio.run(service.native.catalog(scope=other, source_id=lab.source_id, instance_id=instance_id))
    assert not calls


def test_binding_removal_during_collection_discards_samples_and_cached_catalog(lab):
    service, instance_id, calls = native_lab(lab)
    actual = service._diagnostic_source_registry.create(None)
    original = actual.discover_monitoring_metrics
    async def revoke(request):
        result = await original(request)
        with Session(lab.engine) as session:
            binding = session.query(ResourceSourceBindingEntity).filter_by(resource_id=instance_id).one()
            binding.status = "DISABLED"
            session.commit()
        return result
    actual.discover_monitoring_metrics = revoke
    with pytest.raises(AIOpsApplicationError):
        asyncio.run(service.native.catalog(scope=lab.scope, source_id=lab.source_id, instance_id=instance_id))
    before = len(calls)
    with pytest.raises(AIOpsApplicationError):
        asyncio.run(service.native.catalog(scope=lab.scope, source_id=lab.source_id, instance_id=instance_id))
    assert len(calls) == before


def test_binding_capability_restrictions_are_not_bypassed_by_native_monitoring(lab):
    service, instance_id, calls = native_lab(lab)
    with Session(lab.engine) as session:
        binding = session.query(ResourceSourceBindingEntity).filter_by(resource_id=instance_id).one()
        binding.capability_scope_json = {"capabilities":["metric.query_range"]}
        session.commit()
    with pytest.raises(AIOpsApplicationError) as caught:
        asyncio.run(service.native.catalog(scope=lab.scope,source_id=lab.source_id,instance_id=instance_id))
    assert caught.value.code == "SOURCE_CAPABILITY_FORBIDDEN" and not calls


def test_public_snapshot_mode_has_no_history_window_or_false_coverage(lab):
    service,instance_id,calls = native_lab(lab)
    actual = service._diagnostic_source_registry.create(None)
    captured = datetime.now(UTC)
    actual.query_latest_metric = AsyncMock(return_value={"points":((captured-timedelta(days=5),"9.125"),),
        "dimensions":({"mountPoint":"/"},), "unit":"MB", "native_value_type":"NUMBER",
        "captured_at":captured,"truncated":False,"response_sha256":"a"*64})
    result=asyncio.run(service.native.samples(scope=lab.scope,source_id=lab.source_id,instance_id=instance_id,
        query=MonitoringNativeQuery(metric_name="Filesystems/size",mode="LATEST")))
    assert result.window_start is None and result.window_end is None
    assert result.query_mode == "LATEST" and result.history_resolution == "LATEST_SNAPSHOT"
    assert result.points[0].observed_at < captured-timedelta(days=4)
    assert result.points[0].value == "9.125" and result.warnings


def test_latest_snapshot_rejects_wrong_target_and_retains_old_collection_time():
    for target_id, expected in (("0"*32, "SOURCE_TARGET_MISMATCH"), (GUID, None)):
        meta = metadata(name="Response",keys=[],metrics=[{"name":"State","dataType":"STRING"}])
        value,calls = adapter({"targetName":"mes-db","targetType":"oracle_database"},meta,
            collection([{"State":"<script>not executable</script>"}],targetName="mes-db",targetTypeName="oracle_database",
                targetId=target_id,metricGroupName="Response",timeCollected="2020-01-01T00:00:00Z"))
        if expected:
            with pytest.raises(DiagnosticSourceAdapterError) as caught:
                asyncio.run(value.query_latest_metric(request(),metric_name="Response/State"))
            assert caught.value.code == expected
        else:
            result=asyncio.run(value.query_latest_metric(request(),metric_name="Response/State"))
            assert result["points"][0][0].year == 2020
            assert result["points"][0][1] == "<script>not executable</script>"
