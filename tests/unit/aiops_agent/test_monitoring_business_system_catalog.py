"""系统优先监控目录的真实仓储、范围隔离、多源和 OEM 查询回归。"""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from aiops_agent.adapters.diagnostic_sources.catalog import load_metric_catalog
from aiops_agent.application.monitoring_views import MonitoringApplicationService
from aiops_agent.entities import (
    BusinessSystemEntity, BusinessSystemResourceEntity, DiagnosticSourceEntity,
    ResourceSourceBindingEntity,
)
from aiops_agent.monitoring import load_monitoring_profile_catalog
from aiops_agent.ports.diagnostic_source import MetricsEvidenceResult
from platform_core.contracts.aiops.monitoring import MonitoringViewRequest
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_business_system_publication_lifecycle import lab, publish, transition
from tests.unit.aiops_agent.test_resource_monitoring import configure_host, configure_oem_host


def service_for(lab):
    metrics = load_metric_catalog()
    return MonitoringApplicationService(
        uow_factory=lab.service._uow_factory, metric_catalog=metrics,
        profile_catalog=load_monitoring_profile_catalog(metrics),
        diagnostic_source_registry=None, secret_store=None, max_response_bytes=262144,
    )


def clone(entity, **changes):
    values = {column.name: getattr(entity, column.name) for column in entity.__table__.columns}
    return type(entity)(**(values | changes))


@pytest.mark.parametrize("provider", ["PROMETHEUS", "OEM"])
def test_catalog_includes_published_resources_without_source_first_filter(lab, provider):
    (configure_oem_host if provider == "OEM" else configure_host)(lab)
    detail = asyncio.run(publish(lab))
    service = service_for(lab)
    catalog = asyncio.run(service.list_business_systems(scope=lab.scope))
    assert len(catalog) == 1
    assert catalog[0].business_system_id == detail.business_system_id
    assert len(catalog[0].resources) == 2
    for resource in catalog[0].resources:
        assert resource.sources[0].source_type == provider
        assert resource.sources[0].monitoring_readiness == "READY"
        assert resource.business_system_id == catalog[0].business_system_id
        payload = resource.model_dump_json()
        assert all(secret not in payload for secret in ("endpoint", "source_locator", "credential", "config_json"))
    service._query_binding = AsyncMock(return_value=MetricsEvidenceResult())
    host = next(resource for resource in catalog[0].resources if resource.resource_type_key != "DATABASE")
    view = asyncio.run(service.get_view(
        scope=lab.scope, source_id=lab.source_id,
        request=MonitoringViewRequest(profile_id="host-overview", instance_ids=(host.instance_id,), window="1h"),
    ))
    assert view.instances[0].resource_id == host.resource_id
    assert view.source.source_type == provider
    assert len(view.panels) == 5
    assert view.partial and all(panel.quality == "NO_DATA" for panel in view.panels)
    database = next(resource for resource in catalog[0].resources if resource.resource_type_key == "DATABASE")
    profile = (asyncio.run(service.list_target_profiles(scope=lab.scope, target_id=database.instance_id)))[0]
    database_view = asyncio.run(service.get_target_view(
        scope=lab.scope, target_id=database.instance_id, source_id=lab.source_id,
        profile_id=profile.profile_id, window="1h",
    ))
    assert database_view.source.source_type == provider
    assert database_view.instances[0].instance_id == database.instance_id


def test_resource_supports_multiple_bound_sources_without_duplicates(lab):
    configure_host(lab)
    asyncio.run(publish(lab))
    with Session(lab.engine) as session:
        source = session.get(DiagnosticSourceEntity, lab.source_id)
        secondary_id = uuid7()
        session.add(clone(source, diagnostic_source_id=secondary_id, display_name="备用 OEM", source_type="OEM"))
        binding = session.execute(select(ResourceSourceBindingEntity)).scalars().first()
        session.add(clone(binding, resource_source_binding_id=uuid7(),
                          diagnostic_source_id=secondary_id, priority=200))
        session.commit()
    catalog = asyncio.run(service_for(lab).list_business_systems(scope=lab.scope))
    assert len(catalog[0].resources) == 2
    mapped = next(resource for resource in catalog[0].resources if len(resource.sources) == 2)
    assert [source.source_id for source in mapped.sources] == [lab.source_id, secondary_id]


def test_no_binding_and_disconnected_source_do_not_hide_system_or_resources(lab):
    configure_host(lab)
    asyncio.run(publish(lab))
    with Session(lab.engine) as session:
        binding = session.execute(select(ResourceSourceBindingEntity)).scalars().first()
        session.delete(binding)
        source = session.get(DiagnosticSourceEntity, lab.source_id)
        source.connectivity_status = "FAILED"
        session.commit()
    catalog = asyncio.run(service_for(lab).list_business_systems(scope=lab.scope))
    assert len(catalog) == 1 and len(catalog[0].resources) == 2
    unbound = next(resource for resource in catalog[0].resources if not resource.sources)
    assert unbound.capability_gaps[0].code == "RESOURCE_MONITORING_UNAVAILABLE"
    bound = next(resource for resource in catalog[0].resources if resource.sources)
    assert bound.sources[0].monitoring_readiness == "DISCONNECTED"


def test_empty_system_shared_resource_and_scope_isolation(lab):
    configure_host(lab)
    asyncio.run(publish(lab))
    with Session(lab.engine) as session:
        system = session.execute(select(BusinessSystemEntity)).scalars().one()
        empty_id, shared_id = uuid7(), uuid7()
        session.add(clone(system, business_system_id=empty_id, system_code="empty", display_name="空系统"))
        session.add(clone(system, business_system_id=shared_id, system_code="shared", display_name="共享系统"))
        member = session.execute(select(BusinessSystemResourceEntity)).scalars().first()
        session.add(clone(member, system_resource_id=uuid7(), business_system_id=shared_id))
        session.commit()
    service = service_for(lab)
    catalog = asyncio.run(service.list_business_systems(scope=lab.scope))
    assert len(catalog) == 3
    assert next(system for system in catalog if system.business_system_id == empty_id).resources == ()
    shared = next(system for system in catalog if system.business_system_id == shared_id)
    assert len(shared.resources) == 1
    assert shared.resources[0].business_system_id == shared_id
    other_domain = uuid7() if hasattr(lab.scope, "tenant_id") else lab.scope.domain_id + 1
    assert asyncio.run(service.list_business_systems(scope=replace(lab.scope, domain_id=other_domain))) == ()
    if hasattr(lab.scope, "tenant_id"):
        assert asyncio.run(service.list_business_systems(scope=replace(lab.scope, tenant_id=uuid7()))) == ()


def test_offline_system_is_removed_and_oem_container_is_queryable(lab):
    configure_oem_host(lab)
    payload = lab.request.model_dump(mode="json")
    payload["resources"][1]["resource_type_key"] = "CONTAINER"
    lab.request = type(lab.request).model_validate(payload)
    detail = asyncio.run(publish(lab))
    service = service_for(lab)
    service._query_binding = AsyncMock(return_value=MetricsEvidenceResult())
    host = next(resource for resource in asyncio.run(service.list_business_systems(scope=lab.scope))[0].resources
                if resource.resource_type_key == "CONTAINER")
    view = asyncio.run(service.get_view(
        scope=lab.scope, source_id=lab.source_id,
        request=MonitoringViewRequest(profile_id="host-overview", instance_ids=(host.instance_id,), window="1h"),
    ))
    assert view.instances[0].resource_type_key == "CONTAINER"
    asyncio.run(transition(lab, detail, "disable", reason="监控目录下线回归"))
    assert asyncio.run(service.list_business_systems(scope=lab.scope)) == ()
