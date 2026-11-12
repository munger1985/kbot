"""Exadata 专用合同、身份、指标保真、隔离及报告持久化回归。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from pydantic import ValidationError

from aiops_agent.application.exadata import ExadataOEMService
from aiops_agent.application.exadata_catalog import load_catalog
from aiops_agent.application.configuration.instance_discovery import InstanceCandidateRefCodec
from aiops_agent.application.configuration.resource_monitoring import HOST_RESOURCE_TYPES
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.entities import DiagnosticSourceEntity, ManagedResourceEntity, ResourceSourceBindingEntity, ComputeResourceEntity
from aiops_agent.entities.exadata import ExadataInventoryEntity, ExadataReportEntity
from aiops_agent.repositories.exadata import ExadataRepository
from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.adapters.diagnostic_sources.oem_exadata import inventory_digest
from platform_core.contracts.aiops import BusinessSystemResourceGraphCommit
from platform_core.contracts.aiops.exadata import ExadataQuery
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_business_system_publication_lifecycle import lab, LocalUow
from tests.unit.aiops_agent.test_aiops_oem_native_metrics import adapter, collection
from tests.unit.aiops_agent.test_exadata_oem_inventory import association_rows

ROOT, CELL = "a" * 32, "b" * 32
ROOT_ID = {"target_id": ROOT, "target_type": "oracle_dbmachine", "display_name": "测试整机"}
MEMBERS = [{"target_id": CELL, "target_type": "oracle_exadata", "display_name": "测试 Cell"}]


class ExadataUow(LocalUow):
    def __init__(self, engine):
        super().__init__(engine)
        self.exadata = ExadataRepository(self.session)


@pytest.fixture
def exadata_lab(lab):
    f = lab
    ExadataInventoryEntity.metadata.create_all(f.engine, tables=[
        ExadataInventoryEntity.__table__, ExadataReportEntity.__table__])
    f.service._uow_factory = lambda: ExadataUow(f.engine)
    f.service._instance_candidate_refs = InstanceCandidateRefCodec(cipher=f.service._credential_cipher)
    f.digest = inventory_digest(ROOT_ID, MEMBERS)
    f.adapter = SimpleNamespace(
        exadata_inventory=AsyncMock(return_value=(ROOT_ID, MEMBERS, f.digest)),
        discover_exadata=AsyncMock(return_value=[(ROOT_ID, MEMBERS, f.digest)]),
        exadata_latest=AsyncMock(return_value={
            "points": ((datetime.now(UTC), "9007199254740993.123456789"),),
            "dimensions": ({},), "unit": "%", "native_value_type": "STRING",
            "captured_at": datetime.now(UTC), "truncated": False, "response_sha256": "c" * 64,
        }))
    f.service._instance_discovery_adapter = AsyncMock(return_value=f.adapter)
    f.adapter.exadata_history = AsyncMock(return_value={**f.adapter.exadata_latest.return_value, "native_value_type": "NUMBER"})
    f.exadata = ExadataOEMService(f.service)
    yield f


async def publish_exadata(f):
    page = await f.exadata.discover(scope=f.scope, source_id=f.source_id)
    value = f.request.model_dump(mode="json")
    value["resources"] = [{"draft_resource_id": str(uuid7()), "mode": "NEW", "resource_type_key": "EXADATA",
        "display_name": "测试 Exadata", "owner_group_id": value["system"]["responsibility_group_id"],
        "architecture_layer_key": "INFRASTRUCTURE",
        "monitoring": {"source_id": str(f.source_id), "candidate_ref": page.items[0].candidate_ref}}]
    f.request = BusinessSystemResourceGraphCommit.model_validate(value)
    detail = await f.service.commit_business_system(scope=f.scope, request=f.request, idempotency_key=str(uuid7()))
    f.system_id = detail.business_system_id
    f.resource_id = detail.resources[0].resource.resource_id
    return detail


def args(f, **query):
    return dict(scope=f.scope, system_id=f.system_id, resource_id=f.resource_id,
        source_id=f.source_id, query=ExadataQuery(**query))


def test_catalog_is_independent_and_preserves_audited_semantics():
    version, digest, metrics, profiles, diagnoses = load_catalog()
    assert version == "exadata-oem-v1" and len(digest) == 64 and len(metrics) == 267
    by_selector = {item["selector"]: item for item in metrics.values()}
    assert by_selector["Aggregated_HardNFlashDisk_Metric/avg_cd_io_load"]["keys"] == ["celldisk_type"]
    assert next(item for item in metrics.values() if item["selector"].startswith("HostInterConnect_Statistics/"))["keys"] == ["name", "cell_Id"]
    assert by_selector["Aggregated_Exadata_Diskgroup_Capacity_Metric/size"]["native_value_type"] == "STRING"
    assert by_selector["Aggregated_Exadata_Capacity_Metric/Usage"]["display_name"] == "Allocated (%)"
    assert "EXADATA" not in HOST_RESOURCE_TYPES
    assert [p["metric_codes"] for p in profiles] == [p["metric_codes"] for p in diagnoses]
    assert all(m["aggregation"] == "NONE" and m["unit_conversion"] == "NONE" for m in metrics.values())


def test_discovery_only_dbmachine_and_verified_members_no_host_guessing():
    value, calls = adapter(collection([
        {"targetId": ROOT, "targetType": "oracle_dbmachine", "targetName": "测试整机"},
        {"targetId": CELL, "targetType": "oracle_exadata", "targetName": "测试 Cell"}]),
        {"targetId": ROOT, "targetType": "oracle_dbmachine", "targetName": "测试整机"},
        association_rows(("测试整机", "oracle_dbmachine", "测试 Cell", "oracle_exadata")),
        collection([{"targetId": CELL, "targetType": "oracle_exadata", "targetName": "测试 Cell"}]),
        {"targetId": CELL, "targetType": "oracle_exadata", "targetName": "测试 Cell"})
    result = asyncio.run(value.discover_exadata())
    assert len(result) == 1 and result[0][0]["target_id"] == ROOT
    assert result[0][1] == MEMBERS
    assert any(url.endswith("/list/dataservice") for url, _ in calls)
    assert not any(url.endswith("/members") for url, _ in calls)


@pytest.mark.parametrize("kind", ["oracle_exadata", "host", "oracle_database"])
def test_cell_or_host_cannot_bind_as_machine(kind):
    value, _ = adapter({"targetId": ROOT, "targetType": kind, "targetName": "错误类型"})
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.exadata_inventory(ROOT))


def test_unverified_or_duplicate_members_fail_closed():
    value, _ = adapter({"targetId": ROOT, "targetType": "oracle_dbmachine", "targetName": "测试整机"}, association_rows())
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.exadata_inventory(ROOT))


def test_binding_persists_inventory_not_compute_or_database(exadata_lab):
    f = exadata_lab
    detail = asyncio.run(publish_exadata(f))
    assert detail.status == "ENABLED"
    with Session(f.engine) as session:
        binding = session.scalar(select(ResourceSourceBindingEntity).where(ResourceSourceBindingEntity.resource_id == f.resource_id))
        assert binding.target_id is None and binding.source_locator_json["resource_scope"] == "EXADATA"
        inventory = session.get(ExadataInventoryEntity, binding.resource_source_binding_id)
        assert inventory.members_json["members"] == MEMBERS and inventory.inventory_sha256 == f.digest
        assert session.get(ComputeResourceEntity, f.resource_id) is None


def test_monitoring_retains_text_precision_and_quality_and_no_health_inference(exadata_lab):
    f = exadata_lab
    async def run():
        await publish_exadata(f)
        result = await f.exadata.collect(**args(f))
        assert result.query_mode == "LATEST_SNAPSHOT" and result.partial
        assert result.samples[0].value == "9007199254740993.123456789"
        assert result.samples[0].native_value_type == "STRING"
        assert {g.code for g in result.gaps} >= {"THRESHOLDS_UNVERIFIED", "HISTORY_UNAVAILABLE"}
    asyncio.run(run())


@pytest.mark.parametrize("value,age,truncated,quality", [
    (None, 0, False, "EMPTY"), ("", 0, False, "EMPTY"), ("1", 3600, False, "STALE"), ("1", 0, True, "TRUNCATED")])
def test_empty_old_and_truncated_not_healthy(exadata_lab, value, age, truncated, quality):
    f = exadata_lab
    raw = f.adapter.exadata_latest.return_value
    raw["points"] = ((datetime.now(UTC) - timedelta(seconds=age), value),)
    raw["truncated"] = truncated
    async def run():
        await publish_exadata(f)
        result = await f.exadata.collect(**args(f, limit=1))
        assert result.samples[0].quality == quality and result.partial
    asyncio.run(run())


def test_member_change_rejects_collection_before_metric_io(exadata_lab):
    f = exadata_lab
    async def run():
        await publish_exadata(f)
        f.adapter.exadata_inventory.return_value = (ROOT_ID, MEMBERS, "d" * 64)
        with pytest.raises(AIOpsApplicationError):
            await f.exadata.collect(**args(f))
        f.adapter.exadata_latest.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("change", ["resource", "source", "binding", "system", "component"])
def test_scope_source_and_type_isolation_before_metric_io(exadata_lab, change):
    f = exadata_lab
    async def run():
        await publish_exadata(f)
        query = args(f)
        if change == "system": query["system_id"] = uuid7()
        elif change == "component": query["query"] = ExadataQuery(component_id="d" * 32)
        else:
            with Session(f.engine) as session:
                if change == "resource": session.get(ManagedResourceEntity, f.resource_id).resource_type_key = "PHYSICAL_SERVER"
                elif change == "source": session.get(DiagnosticSourceEntity, f.source_id).source_type = "PROMETHEUS"
                else: session.scalar(select(ResourceSourceBindingEntity).where(ResourceSourceBindingEntity.resource_id == f.resource_id)).status = "INACTIVE"
                session.commit()
        with pytest.raises(AIOpsApplicationError):
            await f.exadata.collect(**query)
        f.adapter.exadata_latest.assert_not_awaited()
    asyncio.run(run())


def test_source_change_during_collection_rejects_return(exadata_lab):
    f = exadata_lab
    original = f.adapter.exadata_latest.return_value
    async def read(*args):
        with Session(f.engine) as session:
            source = session.get(DiagnosticSourceEntity, f.source_id)
            source.display_name = "已变更配置"
            session.commit()
        return original
    async def run():
        await publish_exadata(f)
        f.adapter.exadata_latest.side_effect = read
        with pytest.raises(AIOpsApplicationError):
            await f.exadata.collect(**args(f))
    asyncio.run(run())


def test_partial_metric_failure_is_a_gap_and_unknown_member_rejected(exadata_lab):
    f = exadata_lab
    f.adapter.exadata_latest.side_effect = DiagnosticSourceAdapterError("SOURCE_METRIC_TYPE_MISMATCH", "测试不匹配")
    async def run():
        await publish_exadata(f)
        result = await f.exadata.collect(**args(f))
        assert not result.samples and any(g.code == "SOURCE_METRIC_TYPE_MISMATCH" for g in result.gaps)
    asyncio.run(run())


def test_readonly_diagnosis_persisted_without_database_target(exadata_lab):
    f = exadata_lab
    async def run():
        await publish_exadata(f)
        report = await f.exadata.diagnose(**args(f), idempotency_key="test-exadata-report")
        assert report.resource_id == f.resource_id and report.status == "INCONCLUSIVE"
        assert report.findings and not hasattr(report, "target_id")
        saved = await f.exadata.get_report(scope=f.scope, system_id=f.system_id, resource_id=f.resource_id,
            source_id=f.source_id, report_id=report.report_id)
        assert saved == report
        replay = await f.exadata.diagnose(**args(f), idempotency_key="test-exadata-report")
        assert replay.report_id == report.report_id
    asyncio.run(run())


def test_diagnosis_freezes_state_dictionary_and_never_ranges_boolean_state(exadata_lab):
    f = exadata_lab
    async def run():
        await publish_exadata(f)
        view = await f.exadata.collect(**args(f))
        service = next(item for item in view.definitions if item.selector == "CellSrv_Status_Metric/Status")
        assert service.status.values["1"].label == "可用"
        sample = view.samples[0]
        event_code = "oracle_si_network_port_switch.NetworkPortPerformance.OperStatusDown"
        event = load_catalog()[2][event_code]
        view = view.model_copy(update={
            "definitions": (*view.definitions, type(service).model_validate(event)),
            "samples": (sample.model_copy(update={"metric_code": event_code,
                "selector": event["selector"], "value": "0", "native_value_type": "NUMBER",
                "unit": "BOOLEAN", "quality": "OBSERVED"}),),
        })
        f.exadata.collect = AsyncMock(return_value=view)
        report = await f.exadata.diagnose(**args(f), idempotency_key="state-dictionary-report")
        assert not any(item.code == "OBSERVED_RANGE_IN_PAGE" for item in report.findings)
        saved = await f.exadata.get_report(scope=f.scope, system_id=f.system_id, resource_id=f.resource_id,
            source_id=f.source_id, report_id=report.report_id)
        assert saved.evidence.definitions[0].status == view.definitions[0].status
        assert saved.evidence.definitions[-1].status.category == "EVENT"
        assert saved.evidence.samples[0].value == "0"
    asyncio.run(run())


def test_query_contract_has_no_target_guid_url_or_sql():
    with pytest.raises(ValidationError):
        ExadataQuery(target_id=CELL)
    with pytest.raises(ValidationError):
        ExadataQuery(limit=25)
    with pytest.raises(ValidationError):
        ExadataQuery(component_id="../other")


def test_dimension_mismatch_rejected_by_specialized_reader():
    value, _ = adapter({"targetId": CELL, "targetType": "oracle_exadata", "targetName": "测试 Cell"},
        {"name": "Aggregated_HardNFlashDisk_Metric", "keys": [{"name": "object_name"}],
            "metrics": [{"name": "avg_cd_io_load", "dataType": "NUMBER", "unitDisplayName": "%"}]})
    definition = next(m for m in load_catalog()[2].values() if m["selector"] == "Aggregated_HardNFlashDisk_Metric/avg_cd_io_load")
    with pytest.raises(DiagnosticSourceAdapterError) as error:
        asyncio.run(value.exadata_latest(SimpleNamespace(source_locator_key=CELL, max_response_bytes=262144),
            definition, "oracle_exadata"))
    assert error.value.code == "SOURCE_METRIC_DIMENSION_MISMATCH"


@pytest.mark.parametrize("change", ["actor", "source_version", "members"])
def test_binding_rejects_changed_signed_candidate_context(exadata_lab, change):
    f = exadata_lab
    async def run():
        if change == "members":
            f.adapter.exadata_inventory.return_value = (ROOT_ID, MEMBERS, "e" * 64)
        elif change == "actor":
            original = f.service._instance_candidate_refs.encode
            f.service._instance_candidate_refs.encode = lambda **kw: original(**{**kw, "actor_id": "other-actor"})
        else:
            async def discover():
                with Session(f.engine) as session:
                    source = session.get(DiagnosticSourceEntity, f.source_id)
                    source.display_name = "发现期间更改"
                    session.commit()
                return [(ROOT_ID, MEMBERS, f.digest)]
            f.adapter.discover_exadata.side_effect = discover
        with pytest.raises(AIOpsApplicationError):
            await publish_exadata(f)
        with Session(f.engine) as session:
            assert not session.scalars(select(ExadataInventoryEntity)).all()
    asyncio.run(run())


def test_report_numeric_observations_reference_raw_evidence_without_threshold_inference(exadata_lab):
    f = exadata_lab
    f.adapter.exadata_latest.return_value["native_value_type"] = "NUMBER"
    async def run():
        await publish_exadata(f)
        report = await f.exadata.diagnose(**args(f, scene="cell", limit=1), idempotency_key="range-test")
        finding = next(item for item in report.findings if item.code == "OBSERVED_RANGE_IN_PAGE")
        assert finding.severity == "INCONCLUSIVE" and finding.evidence_indexes == (0,)
        assert "9007199254740993.123456789" in finding.detail
        assert report.evidence.definitions[0].unit_conversion == "NONE"
    asyncio.run(run())


def test_bound_query_budget_and_pagination(exadata_lab):
    f = exadata_lab
    async def run():
        await publish_exadata(f)
        result = await f.exadata.collect(**args(f, scene="disk", limit=1))
        assert result.total_queries > 1 and result.next_offset == 1
        assert f.adapter.exadata_latest.await_count == 1
        assert any(g.code == "QUERY_PAGE_PARTIAL" for g in result.gaps)
        await f.exadata.collect(**args(f, scene="disk", limit=1, offset=1))
        assert f.adapter.exadata_latest.await_count == 2
    asyncio.run(run())


def test_no_data_response_remains_a_gap(exadata_lab):
    f = exadata_lab
    f.adapter.exadata_latest.return_value["points"] = ()
    f.adapter.exadata_latest.return_value["dimensions"] = ()
    async def run():
        await publish_exadata(f)
        result = await f.exadata.collect(**args(f))
        assert not result.samples and any(g.code == "NO_DATA" for g in result.gaps)
    asyncio.run(run())


def test_incomplete_association_schema_never_follows_external_urls():
    value, calls = adapter({"targetId": ROOT, "targetType": "oracle_dbmachine", "targetName": "测试整机"},
        collection([], links={"next": {"href": "https://outside.invalid/em/api/targets/" + ROOT + "/members?page=foo"}}))
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.exadata_inventory(ROOT))
    assert len(calls) == 2


def test_specialized_network_failure_is_normalized():
    import aiohttp
    value, _ = adapter()
    value._native_get = AsyncMock(side_effect=aiohttp.ClientConnectionError())
    with pytest.raises(DiagnosticSourceAdapterError) as error:
        asyncio.run(value.exadata_inventory(ROOT))
    assert error.value.code == "SOURCE_UNREACHABLE" and error.value.retryable


def test_public_and_internal_contracts_are_wired_and_strictly_typed():
    from aiops_agent.bootstrap.openapi import create_internal_contract_app
    from main_api.api.ops import router
    internal = create_internal_contract_app().openapi()
    assert "/internal/v1/aiops/exadata/sources/{source_id}/discoveries" in internal["paths"]
    public = {route.endpoint.__name__: route for route in router.routes}
    assert {"discover_exadata", "get_exadata_view", "diagnose_exadata", "get_exadata_report"} <= public.keys()
    request_schema = internal["components"]["schemas"]["ExadataQuery"]
    assert request_schema["additionalProperties"] is False
    assert "target_id" not in request_schema["properties"]


def test_cross_service_client_forwards_resource_path_and_idempotency():
    from platform_clients.aiops import AIOpsManagementClient
    client = object.__new__(AIOpsManagementClient)
    client._json = AsyncMock(return_value={})
    system, resource, source = uuid7(), uuid7(), uuid7()
    asyncio.run(client.diagnose_exadata(system, resource, source, {"scene": "disk"},
        auth_context="trusted-test-context", idempotency_key="diagnosis-123"))
    _, kwargs = client._json.await_args
    assert kwargs["idempotency_key"] == "diagnosis-123"
    assert "/resources/" + str(resource) + "/sources/" + str(source) in client._json.await_args.args[1]
    assert "target" not in client._json.await_args.args[1]


def test_empty_numeric_snapshot_is_preserved_only_on_exadata_branch():
    now = datetime.now(UTC).isoformat()
    definition = next(item for item in load_catalog()[2].values() if item["selector"] == "Cell_Metric/cpu_utilization")
    def reader():
        value, _ = adapter({"targetId": CELL, "targetType": "oracle_exadata", "targetName": "测试 Cell"},
            {"name": "Cell_Metric", "keys": [], "metrics": [{"name": "cpu_utilization", "dataType": "NUMBER", "unitDisplayName": "%"}]},
            collection([{"cpu_utilization": ""}], targetName="测试 Cell", targetTypeName="oracle_exadata",
                targetId=CELL, metricGroupName="Cell_Metric", timeCollected=now))
        return value
    request = SimpleNamespace(source_locator_key=CELL, max_response_bytes=262144)
    result = asyncio.run(reader().exadata_latest(request, definition, "oracle_exadata"))
    assert result["points"][0][1] == ""
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(reader().query_latest_metric(request, metric_name="Cell_Metric/cpu_utilization"))


def test_exadata_relationship_does_not_accept_host_in_machine_role(exadata_lab):
    f = exadata_lab
    async def run():
        await publish_exadata(f)
        value = f.request.model_dump(mode="json")
        value["relations"] = [{"draft_relation_id": str(uuid7()), "relation_type_key": "USES_EXADATA", "participants": [
            {"draft_resource_id": value["resources"][0]["draft_resource_id"], "participant_role": "DATABASE"},
            {"draft_resource_id": value["resources"][0]["draft_resource_id"], "participant_role": "EXADATA"}]}]
        with pytest.raises(ValidationError):
            BusinessSystemResourceGraphCommit.model_validate(value)
        value["relations"] = []
        value["resources"][0]["architecture_layer_key"] = "DATA"
        with pytest.raises(ValidationError):
            BusinessSystemResourceGraphCommit.model_validate(value)
    asyncio.run(run())


def test_history_uses_resource_read_context_without_fake_database_identity():
    from aiops_agent.ports.exadata import ExadataMetricRead
    now = datetime.now(UTC)
    definition = next(item for item in load_catalog()[2].values() if item["selector"] == "Cell_Metric/cpu_utilization")
    value, calls = adapter({"targetId": CELL, "targetType": "oracle_exadata", "targetName": "测试 Cell"},
        {"name": "Cell_Metric", "keys": [], "metrics": [{"name": "cpu_utilization", "dataType": "NUMBER", "unitDisplayName": "%"}]},
        collection([{"targetId": CELL, "targetName": "测试 Cell", "targetTypeName": "oracle_exadata",
            "metricGroupName": "Cell_Metric", "metricName": "cpu_utilization", "keyValues": [],
            "datapoints": [[(now-timedelta(minutes=5)).isoformat(), "0.123456789123456789"]]}]))
    request = ExadataMetricRead(source_locator_key=CELL, window_start=now-timedelta(hours=1), window_end=now)
    assert not hasattr(request, "target_id")
    result = asyncio.run(value.exadata_history(request, definition, "oracle_exadata"))
    assert result["points"][0][1] == "0.123456789123456789"
    assert calls[-1][0].endswith("/metricTimeSeries")
    assert calls[-1][1]["params"]["targetId"] == CELL


def test_history_never_falls_back_to_text_snapshot():
    from aiops_agent.ports.exadata import ExadataMetricRead
    definition = next(item for item in load_catalog()[2].values() if item["selector"] == "CellSrv_Status_Metric/Status")
    value, calls = adapter()
    with pytest.raises(DiagnosticSourceAdapterError) as error:
        asyncio.run(value.exadata_history(ExadataMetricRead(source_locator_key=CELL), definition, "oracle_exadata"))
    assert error.value.code == "SOURCE_METRIC_HISTORY_UNSUPPORTED" and not calls


def test_collection_history_retains_provider_resolution_and_avoids_latest(exadata_lab):
    f = exadata_lab
    async def run():
        await publish_exadata(f)
        result = await f.exadata.collect(**args(f, scene="cell", limit=1, mode="HISTORY", window_minutes=30))
        assert result.query_mode == "HISTORY" and result.history_resolution == "PROVIDER_MANAGED"
        assert (result.window_end - result.window_start).total_seconds() == 1800
        assert result.samples and result.partial
        f.adapter.exadata_history.assert_awaited_once()
        f.adapter.exadata_latest.assert_not_awaited()
    asyncio.run(run())
