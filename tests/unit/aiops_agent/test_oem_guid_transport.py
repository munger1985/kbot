"""OEM 子目标 GUID 通道的维度提升与预算错误回归。"""
import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import pytest

from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.application.exadata_catalog import load_catalog, guid_rest_key_alternatives
from tests.unit.aiops_agent.test_aiops_oem_native_metrics import adapter, collection, request
from tests.unit.aiops_agent.test_oem_capability_closure import native_emcli_adapter, NOW

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
        req = req.model_copy(update={"baseline_days": 7,
            "window_start": req.window_end-timedelta(days=7)})
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
    assert req.max_response_bytes // 2 < budget == value._remaining_response_bytes < req.max_response_bytes
