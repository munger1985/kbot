"""统一绑定、主机归属与实时查询的事务回归；不连接真实数据库或监控源。"""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from aiops_agent.adapters.diagnostic_sources.catalog import load_metric_catalog
from aiops_agent.application.configuration.instance_discovery import InstanceCandidateRefCodec
from aiops_agent.application.configuration.resource_monitoring import apply_hardware_monitoring
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.monitoring_views import MonitoringApplicationService
from aiops_agent.entities import ComputeResourceEntity, DiagnosticSourceEntity, ResourceSourceBindingEntity, ResourceRelationEntity
from aiops_agent.monitoring import load_monitoring_profile_catalog
from aiops_agent.ports.diagnostic_source import HostDiscoveryCandidate, InstanceDiscoveryResult, MetricsEvidenceResult
from platform_core.contracts.aiops import BusinessSystemResourceGraphCommit, ManagedResourcePatch
from platform_core.contracts.aiops.monitoring import MonitoringViewRequest
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_business_system_publication_lifecycle import lab, publish, transition


def configure_host(lab):
    now = datetime.now(UTC)
    with Session(lab.engine) as session:
        source = session.get(DiagnosticSourceEntity, lab.source_id)
        source.source_type = "PROMETHEUS"
        source.adapter_id = "prometheus"
        source.connectivity_status = "CONNECTED"
        source.declared_capabilities_json = {"metric.query_range": {}}
        source.endpoint = "http://prometheus.test"
        session.commit()
    lab.service._instance_candidate_refs = InstanceCandidateRefCodec(cipher=lab.service._credential_cipher)
    lab.adapter = SimpleNamespace(discover_instances=AsyncMock(return_value=InstanceDiscoveryResult(
        host_candidates=(HostDiscoveryCandidate(source_locator_key="node-01", display_name="主机"),),
    )))
    lab.service._instance_discovery_adapter = AsyncMock(return_value=lab.adapter)
    reference = lab.service._instance_candidate_refs.encode(
        **lab.owned, source_id=lab.source_id, actor_id=lab.scope.actor_id,
        purpose="candidate", value={"candidate_kind": "HOST", "host_target_key": "node-01"}, now=now,
    )
    payload = lab.request.model_dump(mode="json")
    host_id = uuid7()
    payload["resources"].append({
        "draft_resource_id": str(host_id), "resource_type_key": "PHYSICAL_SERVER", "display_name": "数据库主机",
        "owner_group_id": str(lab.request.system.responsibility_group_id), "architecture_layer_key": "INFRASTRUCTURE",
        "attributes": {"platform_name": "Oracle Linux"},
        "monitoring": {"source_id": str(lab.source_id), "candidate_ref": reference},
    })
    payload["relations"] = [{
        "draft_relation_id": str(uuid7()), "relation_type_key": "RUNS_ON", "participants": [
            {"draft_resource_id": payload["resources"][0]["draft_resource_id"], "participant_role": "WORKLOAD"},
            {"draft_resource_id": str(host_id), "participant_role": "HOST"},
        ],
    }]
    lab.request = BusinessSystemResourceGraphCommit.model_validate(payload)
    return lab


def test_unified_binding_has_resource_fk_and_no_persisted_target_column():
    columns = ResourceSourceBindingEntity.__table__.columns
    assert "resource_id" in columns and not columns.resource_id.nullable
    assert "target_id" not in columns
    assert "resource_source_binding" in ResourceSourceBindingEntity.__tablename__.lower()


def test_database_and_host_publish_as_distinct_resources_and_topology(lab):
    configure_host(lab)
    detail = asyncio.run(publish(lab))
    db, host = detail.resources
    assert db.resource.database.target_id
    assert host.resource.database is None
    assert host.resource.attributes["platform_name"] == "Oracle Linux"
    binding = host.resource.monitoring_bindings[0]
    assert binding.resource_id == host.resource.resource_id
    assert binding.target_id is None
    assert binding.source_locator == {"target_key": "node-01", "resource_scope": "HOST"}
    assert "host_target_key" not in db.resource.monitoring_bindings[0].source_locator
    assert detail.relations[0].relation_type_key == "RUNS_ON"
    with Session(lab.engine) as session:
        rows = session.execute(select(ResourceSourceBindingEntity)).scalars().all()
        assert len(rows) == 2 and len({row.resource_id for row in rows}) == 2


def test_host_discovery_failure_rolls_back_whole_publication(lab):
    configure_host(lab)
    lab.adapter.discover_instances.return_value = InstanceDiscoveryResult()
    with pytest.raises(AIOpsApplicationError, match="主机候选"):
        asyncio.run(publish(lab))
    with Session(lab.engine) as session:
        assert not session.execute(select(ResourceSourceBindingEntity)).scalars().all()
        assert not session.execute(select(ComputeResourceEntity)).scalars().all()
        assert not session.execute(select(ResourceRelationEntity)).scalars().all()


def test_host_binding_rejects_database_candidate_and_cross_actor(lab):
    configure_host(lab)
    item = lab.request.resources[1]
    for kind, actor in (("DATABASE", lab.scope.actor_id), ("HOST", str(uuid7()))):
        reference = lab.service._instance_candidate_refs.encode(
            **lab.owned, source_id=lab.source_id, actor_id=actor,
            purpose="candidate", value={"candidate_kind": kind, "host_target_key": "node-01"},
        )
        payload = lab.request.model_dump(mode="json")
        payload["resources"][1]["monitoring"]["candidate_ref"] = reference
        lab.request = BusinessSystemResourceGraphCommit.model_validate(payload)
        with pytest.raises(AIOpsApplicationError):
            asyncio.run(publish(lab))


def test_database_contract_no_longer_accepts_nested_host_candidate(lab):
    payload = lab.request.model_dump(mode="json")
    payload["resources"][0]["database"]["monitoring"]["host_candidate_ref"] = "old-host"
    with pytest.raises(ValidationError):
        BusinessSystemResourceGraphCommit.model_validate(payload)


def test_host_metrics_are_not_selected_for_database_profiles():
    metrics = load_metric_catalog()
    profiles = load_monitoring_profile_catalog(metrics)
    for db_type, version in (("ORACLE", "19c"), ("MYSQL", "8.4"), ("POSTGRESQL", "16")):
        profile = profiles.overview_for_database(db_type, version)
        assert all(metrics.get(code).resource_scope == "DATABASE" for code in profile.metric_codes)
        assert metrics.select(("host.cpu.utilization",), db_type=db_type) == ()
    host = profiles.resolve_resource("host-overview", ("PHYSICAL_SERVER",))
    assert host.supported_db_types == ()
    assert all(metrics.get(code).resource_scope == "HOST" for code in host.metric_codes)


def test_hardware_query_uses_own_binding_and_respects_system_offline(lab):
    configure_host(lab)

    async def run():
        detail = await publish(lab)
        resource_id = detail.resources[1].resource.resource_id
        metrics = load_metric_catalog()
        service = MonitoringApplicationService(
            uow_factory=lab.service._uow_factory, metric_catalog=metrics,
            profile_catalog=load_monitoring_profile_catalog(metrics), diagnostic_source_registry=None,
            secret_store=None, max_response_bytes=262144,
        )
        service._query_binding = AsyncMock(return_value=MetricsEvidenceResult())
        profiles = await service.list_profiles(scope=lab.scope, source_id=lab.source_id)
        assert any(profile.profile_id == "host-overview" for profile in profiles)
        instances = await service.list_instances(scope=lab.scope, source_id=lab.source_id)
        assert any(instance.resource_id == resource_id and instance.db_type is None for instance in instances)
        request = MonitoringViewRequest(profile_id="host-overview", instance_ids=(resource_id,), window="1h")
        view = await service.get_view(scope=lab.scope, source_id=lab.source_id, request=request)
        assert view.instances[0].db_type is None
        assert view.instances[0].resource_id == resource_id
        assert len(view.panels) == 5
        assert service._query_binding.call_args.kwargs["frozen"]["source_locator"] == {"target_key": "node-01", "resource_scope": "HOST"}
        await transition(lab, detail, "disable", reason="测试下线")
        service._query_binding.reset_mock()
        with pytest.raises(AIOpsApplicationError):
            await service.get_view(scope=lab.scope, source_id=lab.source_id, request=request)
        service._query_binding.assert_not_awaited()

    asyncio.run(run())


def test_hardware_binding_edit_requires_offline_system_and_current_version(lab):
    configure_host(lab)

    async def run():
        detail = await publish(lab)
        host = detail.resources[1]
        binding = host.resource.monitoring_bindings[0]
        request = ManagedResourcePatch(
            business_system_id=detail.business_system_id, system_resource_id=host.system_resource_id,
            attributes={"platform_name": "Oracle Linux", "platform_version": "9"},
            monitoring={**lab.request.resources[1].monitoring.model_dump(), "binding_id": binding.binding_id, "binding_row_version": binding.row_version},
        )
        with pytest.raises(AIOpsApplicationError, match="运行中"):
            await lab.service.patch_managed_resource(scope=lab.scope, resource_id=host.resource.resource_id, request=request, expected_version=host.resource.row_version)
        await transition(lab, detail, "disable", reason="测试下线")
        updated = await lab.service.patch_managed_resource(scope=lab.scope, resource_id=host.resource.resource_id, request=request, expected_version=host.resource.row_version)
        assert updated.monitoring_bindings[0].binding_id == binding.binding_id
        assert updated.attributes["platform_version"] == "9"
        with Session(lab.engine) as session:
            assert session.get(ComputeResourceEntity, host.resource.resource_id).platform_version == "9"
        stale = request.model_copy(update={"monitoring": request.monitoring.model_copy(update={"binding_row_version": 999})})
        with pytest.raises(AIOpsApplicationError):
            await lab.service.patch_managed_resource(scope=lab.scope, resource_id=host.resource.resource_id, request=stale, expected_version=updated.row_version)

    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["source", "binding", "membership"])
def test_hardware_query_discards_results_after_configuration_revocation(lab, mutation):
    configure_host(lab)

    async def run():
        detail = await publish(lab)
        host = detail.resources[1]
        metrics = load_metric_catalog()
        service = MonitoringApplicationService(
            uow_factory=lab.service._uow_factory, metric_catalog=metrics,
            profile_catalog=load_monitoring_profile_catalog(metrics),
            diagnostic_source_registry=None, secret_store=None, max_response_bytes=262144,
        )

        async def revoke(**_kwargs):
            from aiops_agent.entities import BusinessSystemResourceEntity
            with Session(lab.engine) as session:
                if mutation == "source":
                    session.get(DiagnosticSourceEntity, lab.source_id).status = "DISABLED"
                elif mutation == "binding":
                    binding = session.get(ResourceSourceBindingEntity, host.resource.monitoring_bindings[0].binding_id)
                    binding.source_locator_key = "changed-node"
                else:
                    session.get(BusinessSystemResourceEntity, host.system_resource_id).status = "REMOVED"
                session.commit()
            return MetricsEvidenceResult()

        service._query_binding = AsyncMock(side_effect=revoke)
        with pytest.raises(AIOpsApplicationError):
            await service.get_view(scope=lab.scope, source_id=lab.source_id, request=MonitoringViewRequest(
                profile_id="host-overview", instance_ids=(host.resource.resource_id,), window="1h",
            ))

    asyncio.run(run())


def test_hardware_query_returns_gaps_and_enforces_response_budget(lab):
    configure_host(lab)

    async def run():
        from aiops_agent.contracts.evidence import ObservationGap
        detail = await publish(lab)
        metrics = load_metric_catalog()
        service = MonitoringApplicationService(
            uow_factory=lab.service._uow_factory, metric_catalog=metrics,
            profile_catalog=load_monitoring_profile_catalog(metrics),
            diagnostic_source_registry=None, secret_store=None, max_response_bytes=262144,
        )
        service._query_binding = AsyncMock(return_value=MetricsEvidenceResult(gaps=(
            ObservationGap(metric_code="host.cpu.utilization", source_id=str(lab.source_id), binding_id=str(detail.resources[1].resource.monitoring_bindings[0].binding_id),
                code="NO_DATA", detail="当前主机无采样"),
        )))
        request = MonitoringViewRequest(profile_id="host-overview",
            instance_ids=(detail.resources[1].resource.resource_id,), window="1h")
        view = await service.get_view(scope=lab.scope, source_id=lab.source_id, request=request)
        assert view.partial
        assert any(gap.code == "NO_DATA" for gap in view.gaps)
        service._max_response_bytes = 1024
        with pytest.raises(AIOpsApplicationError, match="预算"):
            await service.get_view(scope=lab.scope, source_id=lab.source_id, request=request)

    asyncio.run(run())
