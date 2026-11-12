"""真实 OEM 设备状态、采集周期和无样本集合的专用回归。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.application.exadata_catalog import load_catalog
from platform_core.contracts.aiops.exadata import ExadataQuery
from tests.unit.aiops_agent.test_aiops_oem_native_metrics import adapter
from tests.unit.aiops_agent.test_exadata_oem import exadata_lab, publish_exadata, args, ROOT_ID, MEMBERS
from tests.unit.aiops_agent.test_business_system_publication_lifecycle import lab
from aiops_agent.adapters.diagnostic_sources.oem_exadata import inventory_digest


HOST = "c" * 32


@pytest.mark.parametrize("state", ["UP", "DOWN", "AGENT_UNREACHABLE", "BLACKOUT", "FUTURE_STATUS", None])
def test_availability_reads_verified_target_only_and_preserves_unknown(state):
    definition = load_catalog()[2]["host.OEMTarget.status"]
    value, calls = adapter({"id": HOST, "name": "SIM host", "typeName": "host", "status": state})
    result = asyncio.run(value.exadata_latest(SimpleNamespace(source_locator_key=HOST, max_response_bytes=262144),
        definition, "host"))
    assert len(calls) == 1 and calls[0][0].endswith("/targets/" + HOST)
    assert result["native_value_type"] == "STRING" and result["unit"] is None
    assert result["points"] == ((result["captured_at"], state),) if state else result["points"] == ()
    assert state != "FUTURE_STATUS" or state not in definition["status"]["values"]


@pytest.mark.parametrize("kind,status,error", [
    ("oracle_database", "UP", "SOURCE_TARGET_MISMATCH"),
    ("host", 1, "SOURCE_RESPONSE_INVALID"), ("host", "UP\nDOWN", "SOURCE_RESPONSE_INVALID"),
    ("host", "UP\x7fDOWN", "SOURCE_RESPONSE_INVALID")])
def test_availability_fails_closed_on_wrong_identity_or_invalid_status(kind, status, error):
    definition = load_catalog()[2]["host.OEMTarget.status"]
    value, _ = adapter({"id": HOST, "name": "SIM host", "typeName": kind, "status": status})
    with pytest.raises(DiagnosticSourceAdapterError) as caught:
        asyncio.run(value.exadata_latest(SimpleNamespace(source_locator_key=HOST, max_response_bytes=262144), definition, "host"))
    assert caught.value.code == error


def test_current_availability_is_not_a_fabricated_history_metric():
    definition = load_catalog()[2]["host.OEMTarget.status"]
    value, calls = adapter()
    with pytest.raises(DiagnosticSourceAdapterError) as caught:
        asyncio.run(value.exadata_history(SimpleNamespace(source_locator_key=HOST), definition, "host"))
    assert caught.value.code == "SOURCE_METRIC_HISTORY_UNSUPPORTED" and not calls
    assert not definition["baseline_supported"]


@pytest.mark.parametrize("age,quality", [(1900, "OBSERVED"), (4200, "OBSERVED"), (4501, "STALE")])
def test_hourly_cpu_freshness_is_shared_by_monitoring_and_diagnosis(exadata_lab, age, quality):
    f = exadata_lab
    members = [*MEMBERS, {"target_id": HOST, "target_type": "host", "display_name": "SIM DB Server"}]
    digest = inventory_digest(ROOT_ID, members)
    f.adapter.exadata_inventory.return_value = (ROOT_ID, members, digest)
    f.adapter.discover_exadata.return_value = [(ROOT_ID, members, digest)]
    now = datetime.now(UTC)
    f.adapter.exadata_latest.return_value.update(points=((now - timedelta(seconds=age), "5.27"),),
        captured_at=now, native_value_type="NUMBER", unit="%")
    async def run():
        await publish_exadata(f)
        view = await f.exadata.collect(**args(f, scene="dbserver", component_id=HOST, limit=1))
        assert view.samples[0].quality == quality
        report = await f.exadata.diagnose(**args(f, scene="dbserver", component_id=HOST, limit=1),
            idempotency_key="hourly-cpu-" + str(age))
        assert report.evidence.samples[0].quality == quality
    asyncio.run(run())


def test_native_health_sources_are_in_monitoring_and_agent_diagnosis_profiles():
    _, _, metrics, monitoring, diagnosis = load_catalog()
    assert [p["metric_codes"] for p in monitoring] == [p["metric_codes"] for p in diagnosis]
    overview = next(p for p in monitoring if p["id"] == "overview")["metric_codes"]
    assert "host.OEMTarget.status" in overview and "host.Response.Status" not in metrics
    for parent in ("oracle_si_netswitch", "oracle_si_pdu"):
        assert parent + ".ComponentState.ComponentState" not in overview
    for col in ("HealthStatus", "CoolingState", "PowerState", "TemperatureState"):
        definition = metrics["oracle_si_netswitch.SwitchSystemStatus." + col]
        assert definition["keys"] == [] and definition["status"]["values"]["OK"]["label"] == "正常"
        network = next(p for p in monitoring if p["id"] == "network")["metric_codes"]
        assert definition["code"] in network
    assert "oracle_si_netswitch.SensorStatus.Status" in overview
    assert "oracle_si_netswitch.PowerSupplyStatus.Status" in overview
    assert metrics["oracle_si_netswitch.SensorStatus.Status"]["keys"] == ["SensorId"]
    assert metrics["oracle_si_netswitch.PowerSupplyStatus.Status"]["keys"] == ["Name"]
    assert not metrics["oracle_si_pdu.PduSummaryStatus.TotalPduEnergy"]["baseline_supported"]
    assert metrics["oracle_si_pdu.PduSummaryStatus.TotalPduActivePower"]["baseline_supported"]
    assert ExadataQuery(scene="capacity").scene == "capacity"
