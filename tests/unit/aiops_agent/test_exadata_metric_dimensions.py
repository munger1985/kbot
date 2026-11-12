"""Exadata 专用读取的具名维度与上游故障隔离回归；不改变普通 OEM 请求。"""

import asyncio
from datetime import timedelta

import aiohttp
import pytest

from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from tests.unit.aiops_agent.test_aiops_oem_native_metrics import adapter, collection, request, NOW


CELL = "b" * 32
TARGET = {"targetId": CELL, "targetType": "oracle_exadata", "targetName": "SIM Cell"}
DEFINITION = {"selector": "Filesystem_Utilization_Metric/utilization",
              "keys": ["name", "cell_Id"], "native_value_type": "NUMBER", "semantics": "GAUGE"}


def metadata(keys):
    return {"name": "Filesystem_Utilization_Metric",
            "keys": [{"name": key} for key in keys],
            "metrics": [{"name": "utilization", "dataType": "NUMBER", "unitDisplayName": "%"}]}


def latest():
    return collection([{"name": "/opt", "cell_Id": "SIM-CELL-0001", "utilization": "23.50000000001"}],
        targetName="SIM Cell", targetTypeName="oracle_exadata", targetId=CELL,
        metricGroupName="Filesystem_Utilization_Metric", timeCollected=NOW.isoformat())


def history():
    return collection([{"targetName": "SIM Cell", "targetTypeName": "oracle_exadata",
        "targetId": CELL, "metricGroupName": "Filesystem_Utilization_Metric", "metricName": "utilization",
        "keyValues": [{"name": "cell_Id", "value": "SIM-CELL-0001"}, {"name": "name", "value": "/opt"}],
        "datapoints": [[(NOW - timedelta(minutes=1)).isoformat(), "23.50000000001"]]}])


@pytest.mark.parametrize("mode", ["latest", "history"])
@pytest.mark.parametrize("keys", [["name", "cell_Id"], ["cell_Id", "name"]])
def test_named_dimensions_accept_reordered_metadata_without_merging_values(mode, keys):
    value, calls = adapter(TARGET, metadata(keys), latest() if mode == "latest" else history())
    result = asyncio.run(getattr(value, "exadata_" + mode)(
        request(source_locator_key=CELL), DEFINITION, "oracle_exadata"))
    assert result["points"][0][1] == "23.50000000001"
    assert result["dimensions"] == ({"name": "/opt", "cell_Id": "SIM-CELL-0001"},)
    assert not result["truncated"]
    assert all(call[1]["allow_redirects"] is False for call in calls)


@pytest.mark.parametrize("mode", ["latest", "history"])
@pytest.mark.parametrize("keys", [["name"], ["name", "other"], ["name", "cell_Id", "extra"],
                                  ["name", "name"], []])
def test_missing_extra_wrong_or_duplicate_dimensions_still_fail_closed(mode, keys):
    value, calls = adapter(TARGET, metadata(keys))
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(getattr(value, "exadata_" + mode)(
            request(source_locator_key=CELL), DEFINITION, "oracle_exadata"))
    assert len(calls) == 2


@pytest.mark.parametrize("mode", ["latest", "history"])
@pytest.mark.parametrize("status,code", [(500, "SOURCE_PROVIDER_ERROR"), (503, "SOURCE_PROVIDER_ERROR"),
                                        (401, "SOURCE_AUTH_FAILED"), (429, "SOURCE_RATE_LIMITED")])
def test_exadata_http_failures_are_not_no_data_or_network_errors(mode, status, code):
    value, _ = adapter(TARGET, metadata(DEFINITION["keys"]), {}, statuses=[200, 200, status])
    with pytest.raises(DiagnosticSourceAdapterError) as exc:
        asyncio.run(getattr(value, "exadata_" + mode)(
            request(source_locator_key=CELL), DEFINITION, "oracle_exadata"))
    assert exc.value.code == code


@pytest.mark.parametrize("mode", ["latest", "history"])
@pytest.mark.parametrize("error", [aiohttp.ClientConnectionError(), TimeoutError()])
def test_transport_failures_remain_source_unreachable(mode, error):
    value, _ = adapter()
    def failed(*args, **kwargs):
        raise error
    value._session.get = failed
    with pytest.raises(DiagnosticSourceAdapterError) as exc:
        asyncio.run(getattr(value, "exadata_" + mode)(
            request(source_locator_key=CELL), DEFINITION, "oracle_exadata"))
    assert exc.value.code == "SOURCE_UNREACHABLE"


def test_ordinary_oem_http_failure_is_not_reported_as_a_network_failure():
    value, _ = adapter(TARGET, metadata(DEFINITION["keys"]), {}, statuses=[200, 200, 500])
    with pytest.raises(DiagnosticSourceAdapterError) as exc:
        asyncio.run(value.query_native_metric(request(source_locator_key=CELL),
            metric_name=DEFINITION["selector"], limit=10))
    assert exc.value.code == "SOURCE_PROVIDER_ERROR"
