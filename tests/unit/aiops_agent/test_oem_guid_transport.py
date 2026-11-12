"""OEM 子目标 GUID 通道的维度提升与预算错误回归。"""
import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import pytest

from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.application.exadata_catalog import load_catalog, guid_rest_key_alternatives
from aiops_agent.ports.exadata import ExadataMetricRead
from tests.unit.aiops_agent.test_aiops_oem_native_metrics import adapter, collection, request
from tests.unit.aiops_agent.test_oem_capability_closure import native_emcli_adapter, NOW
from tests.unit.aiops_agent.test_monitoring_oem_native import native_lab
from tests.unit.aiops_agent.test_business_system_publication_lifecycle import lab
from platform_core.contracts.aiops.monitoring_native import MonitoringNativeQuery
from platform_core.contracts.aiops.exadata import ExadataBaselineQuery
from tests.unit.aiops_agent.test_exadata_baseline_analysis import setup_service

GUID = "b" * 32
GROUPS = [
    ("oracle_si_hw_component_pdu", "PduModulePhaseStatus"),
    ("oracle_si_hw_component_pdu", "PduModuleStatus"),
    ("oracle_si_hw_component_server", "ComponentState"),
    ("oracle_si_hw_component_switch", "ComponentState"),
    ("oracle_si_network_port_server", "NetworkPortPerformance"),
    ("oracle_si_network_port_switch", "NetworkPortPerformance"),
    ("oracle_si_network_port_switch", "SwitchEthPortsMonitor"),
]


def metric(kind, group):
    candidates = [item for item in load_catalog()[2].values()
        if item["target_type"] == kind and item["selector"].split("/")[0] == group]
    return next((item for item in candidates if item["native_value_type"] == "NUMBER"), candidates[0])


def replies(definition, keys, *, latest=True, extra=None):
    kind = definition["target_type"]
    group, column = definition["selector"].split("/")
    identity = {"targetId": GUID, "targetType": kind, "targetName": "reviewed-component"}
    metadata = {"name": group, "keys": [{"name": name} for name in keys],
        "metrics": [{"name": column, "dataType": definition["native_value_type"], "unitDisplayName": "V"}]}
    now = datetime.now(UTC) - timedelta(minutes=1)
    if latest:
        payload = collection([{column: "230.0", **{key: "component-key" for key in keys}, **(extra or {})}],
            targetId=GUID, targetTypeName=kind, targetName=identity["targetName"],
            metricGroupName=group, timeCollected=now.isoformat())
    else:
        payload = collection([{"targetId": GUID, "targetTypeName": kind, "targetName": identity["targetName"],
            "metricGroupName": group, "metricName": column,
            "keyValues": [{"name": key, "value": "component-key"} for key in keys],
            "datapoints": [[now.isoformat(), "230.0"]], **(extra or {})}])
    return identity, metadata, payload


@pytest.mark.parametrize("kind,group", GROUPS)
@pytest.mark.parametrize("promoted", [True, False])
@pytest.mark.parametrize("operation", ["latest", "history", "baseline"])
def test_guid_component_retains_real_transport_keys_without_fabrication(kind, group, promoted, operation):
    definition = metric(kind, group)
    keys = [] if promoted else definition["keys"]
    value, calls = adapter(*replies(definition, keys, latest=operation == "latest"))
    value.context = value.context.model_copy(update={"config": {"exadata_metric_transport": "EMCLI"}})
    value._exadata_emcli_client = NS(read=AsyncMock())
    req = request(source_locator_key=GUID, window_start=datetime.now(UTC)-timedelta(hours=1),
        window_end=datetime.now(UTC), pagination=None)
    if operation == "baseline":
        # 不发布部分均值；此处直接验证唯一六天分片沿用相同维度合同。
        value.query_native_metric = AsyncMock(return_value={
            "series": ({"dimensions": {}, "total": "230.0", "sample_count": 1},),
            "unit": "V", "truncated": False, "response_sha256": "a"*64})
        req = ExadataMetricRead(source_locator_key=GUID, max_response_bytes=req.max_response_bytes,
            baseline_days=7, window_start=req.window_end-timedelta(days=7), window_end=req.window_end)
        if not definition["baseline_supported"]:
            with pytest.raises(DiagnosticSourceAdapterError) as caught:
                asyncio.run(value.exadata_baseline(req, definition, kind))
            assert caught.value.code == "SOURCE_QUERY_INVALID"
            value.query_native_metric.assert_not_awaited()
            return
        result = asyncio.run(value.exadata_baseline(req, definition, kind))
        assert result["series"][0]["sample_count"] == 2
        assert all(call.kwargs["_alternate_key_sets"] == ((),)
            for call in value.query_native_metric.call_args_list)
    else:
        if operation == "history" and definition["native_value_type"] != "NUMBER":
            with pytest.raises(DiagnosticSourceAdapterError) as caught:
                asyncio.run(value.exadata_history(req, definition, kind))
            assert caught.value.code == "SOURCE_METRIC_HISTORY_UNSUPPORTED"
            assert not calls
            return
        result = asyncio.run(getattr(value, "exadata_" + operation)(req, definition, kind))
        assert result["points"][0][1] == "230.0"
        assert result["dimensions"] == (({} if promoted else {key: "component-key" for key in keys}),)
        assert all(GUID in url or url.endswith("/metricTimeSeries") for url, _ in calls)
    value._exadata_emcli_client.read.assert_not_awaited()


@pytest.mark.parametrize("operation", ["latest", "history"])
@pytest.mark.parametrize("bad_keys", [["other"], ["ComponentId", "other"]])
def test_guid_rule_does_not_allow_unknown_or_additional_dimension(operation, bad_keys):
    definition = metric(*GROUPS[0])
    value, _ = adapter(*replies(definition, bad_keys, latest=operation == "latest"))
    with pytest.raises(DiagnosticSourceAdapterError) as caught:
        asyncio.run(getattr(value, "exadata_" + operation)(request(source_locator_key=GUID),
            definition, definition["target_type"]))
    assert caught.value.code == "SOURCE_METRIC_DIMENSION_MISMATCH"


def test_promoted_latest_cannot_silently_drop_nonempty_component_key():
    definition = metric(*GROUPS[0])
    value, _ = adapter(*replies(definition, [], extra={"ComponentId": "other-component"}))
    with pytest.raises(DiagnosticSourceAdapterError) as caught:
        asyncio.run(value.exadata_latest(request(source_locator_key=GUID), definition, definition["target_type"]))
    assert caught.value.code == "SOURCE_METRIC_DIMENSION_MISMATCH"


def test_only_explicit_audited_groups_get_alternative_and_repository_catalog_stays_exact():
    for definition in load_catalog()[2].values():
        if (definition["target_type"], definition["selector"].split("/")[0]) in GROUPS:
            assert guid_rest_key_alternatives(definition, definition["target_type"]) == ((),)
            assert definition["keys"] in (["ComponentId"], ["PortId"])
        else:
            assert guid_rest_key_alternatives(definition, definition["target_type"]) == ()


def test_pdu_physical_gauges_support_baseline_but_energy_identifiers_and_thresholds_do_not():
    pdu = [definition for definition in load_catalog()[2].values()
        if definition["target_type"] == "oracle_si_hw_component_pdu"]
    supported = {definition["selector"] for definition in pdu if definition["baseline_supported"]}
    assert supported == {"PduModulePhaseStatus/" + name for name in
        ("ModulePhaseActivePower", "ModulePhaseAmp", "ModulePhaseApparentPower", "ModulePhaseVoltage")} | {
        "PduModuleStatus/OutOfBalanceAmpLevel", "PduModuleStatus/TotalInputActivePower"}


@pytest.mark.parametrize("upstream,expected", [
    ("SOURCE_RESPONSE_TOO_LARGE", "SOURCE_RESULT_TRUNCATED"),
    ("SOURCE_EMCLI_TIMEOUT", "SOURCE_EMCLI_TIMEOUT"),
    ("SOURCE_PROVIDER_ERROR", "SOURCE_PROVIDER_ERROR"),
])
def test_native_emcli_budget_failure_has_public_error_without_fallback(upstream, expected):
    value = native_emcli_adapter()
    value._exadata_emcli_client.read_native.side_effect = DiagnosticSourceAdapterError(upstream, "测试错误")
    with patch("aiops_agent.adapters.diagnostic_sources.oem_native_emcli.datetime") as clock:
        clock.now.return_value = NOW
        with pytest.raises(DiagnosticSourceAdapterError) as caught:
            asyncio.run(value.query_native_metric(request(window_start=NOW-timedelta(hours=1), window_end=NOW),
                metric_name="Filesystems/size", limit=1000))
    assert caught.value.code == expected
    assert value._exadata_emcli_client.read_native.await_count == 1


def test_native_emcli_uses_only_actual_remaining_budget_not_a_fixed_half_or_expansion():
    value = native_emcli_adapter()
    req = request(window_start=NOW-timedelta(hours=1), window_end=NOW, max_response_bytes=262144)
    with patch("aiops_agent.adapters.diagnostic_sources.oem_native_emcli.datetime") as clock:
        clock.now.return_value = NOW
        asyncio.run(value.query_native_metric(req, metric_name="Filesystems/size", limit=1000))
    budget = value._exadata_emcli_client.read_native.call_args.kwargs["max_bytes"]
    assert req.max_response_bytes // 2 < budget == value._remaining_response_bytes + 4096 < req.max_response_bytes


def test_native_baseline_budget_failure_keeps_current_and_never_doubles_budget(lab):
    service, instance_id, _ = native_lab(lab)
    actual = service._diagnostic_source_registry.create(None)
    original = actual.query_native_metric
    requests = []
    async def read(req, **options):
        requests.append(req)
        if options.get("_summary_only"):
            assert req.max_response_bytes == actual._remaining_response_bytes
            raise DiagnosticSourceAdapterError("SOURCE_RESULT_TRUNCATED", "基线超过剩余预算")
        return await original(req, **options)
    actual.query_native_metric = read
    result = asyncio.run(service.native.samples(scope=lab.scope, source_id=lab.source_id,
        instance_id=instance_id, query=MonitoringNativeQuery(metric_name="Filesystems/size", baseline="1w")))
    assert result.points[0].value == "90071992547409931"
    assert result.baseline.unavailable_reason and not result.baseline.series
    assert len(requests) == 2
    frozen = asyncio.run(service.native._binding(lab.scope, lab.source_id, instance_id))
    assert requests[0].max_response_bytes == frozen["max_response_bytes"]
    assert 1024 <= requests[1].max_response_bytes < requests[0].max_response_bytes


def test_native_baseline_with_insufficient_remaining_bytes_does_not_call_provider(lab):
    service, _, _ = native_lab(lab)
    actual = NS(_remaining_response_bytes=512, query_native_metric=AsyncMock())
    frozen = {"binding_id": "test"}
    result = asyncio.run(service.native._baseline(actual, request(), frozen, NS(name="Filesystems/size",
        native_value_type="NUMBER"), "1w", 0))
    assert not result.series and "剩余字节不足" in result.unavailable_reason
    actual.query_native_metric.assert_not_awaited()


@pytest.mark.parametrize("latest_keys,history_keys,accepted", [
    ({}, {}, True),
    ({"ComponentId": "phase1"}, {"ComponentId": "phase1"}, True),
    ({}, {"ComponentId": "phase1"}, False),
    ({"ComponentId": "phase1"}, {}, False),
    ({}, {"unexpected": "phase1"}, False),
])
def test_pdu_application_baseline_requires_consistent_reviewed_transport_dimensions(latest_keys, history_keys, accepted):
    service, value, args = setup_service()
    frozen = service._snapshot.return_value[0]
    frozen["inventory"]["members"][0]["target_type"] = "oracle_si_hw_component_pdu"
    args["query"] = ExadataBaselineQuery(component_id=GUID,
        metric_code="oracle_si_hw_component_pdu.PduModulePhaseStatus.ModulePhaseVoltage")
    value.exadata_latest.return_value.update(points=((datetime.now(UTC), "230.0"),),
        dimensions=(latest_keys,), unit="VOLT")
    value.exadata_baseline.return_value.update(
        series=({"dimensions": history_keys, "total": "460", "sample_count": 2},), unit="VOLT")
    result = asyncio.run(service.baseline(**args))
    assert bool(result.comparisons) is accepted
    assert bool(result.baseline.series) is accepted
    assert result.current_samples[0].value == "230.0"
    if accepted:
        assert result.baseline.series[0].average == "230"
    else:
        assert "SOURCE_METADATA_CHANGED" in result.baseline.unavailable_reason


def test_pdu_baseline_cache_cannot_reuse_a_different_key_representation():
    service, value, args = setup_service()
    service._snapshot.return_value[0]["inventory"]["members"][0]["target_type"] = "oracle_si_hw_component_pdu"
    args["query"] = ExadataBaselineQuery(component_id=GUID,
        metric_code="oracle_si_hw_component_pdu.PduModulePhaseStatus.ModulePhaseVoltage")
    value.exadata_latest.return_value.update(points=((datetime.now(UTC), "230.0"),),
        dimensions=({},), unit="VOLT")
    value.exadata_baseline.return_value.update(
        series=({"dimensions": {}, "total": "460", "sample_count": 2},), unit="VOLT")
    first = asyncio.run(service.baseline(**args))
    assert first.comparisons
    value.exadata_latest.return_value["dimensions"] = ({"ComponentId": "phase1"},)
    value.exadata_baseline.return_value["series"] = (
        {"dimensions": {"ComponentId": "phase1"}, "total": "460", "sample_count": 2},)
    second = asyncio.run(service.baseline(**args))
    assert second.comparisons and value.exadata_baseline.await_count == 2
    assert second.baseline.series[0].dimensions == {"ComponentId": "phase1"}
