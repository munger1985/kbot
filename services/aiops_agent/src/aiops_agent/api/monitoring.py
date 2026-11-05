"""Provider 无关的实时监控 Internal API。"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request

from aiops_agent.api.dependencies import (
    get_aiops_auth_context,
    require_service_scope,
)
from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.application.monitoring_views import MonitoringApplicationService
from platform_core.contracts.aiops.monitoring import (
    MonitoringInstanceSummary,
    MonitoringBusinessSystemSummary,
    MonitoringProfileSummary,
    MonitoringSourceSummary,
    MonitoringView,
    MonitoringViewRequest,
    MonitoringBaselinePeriod,
    MonitoringWindowName,
)
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops.monitoring_native import (
    MonitoringNativeCatalog, MonitoringNativeQuery, MonitoringNativeSamples,
)


router = APIRouter(
    prefix="/internal/v1/aiops/monitoring", tags=["AIOps Monitoring"]
)
target_router = APIRouter(
    prefix="/internal/v1/aiops/targets", tags=["AIOps Monitoring"]
)


def get_service(request: Request) -> MonitoringApplicationService:
    return request.app.state.monitoring_service


def get_scope(
    request: Request,
    auth_context: AuthContext = Depends(get_aiops_auth_context),
) -> ConfigurationScope:
    require_service_scope(request, "aiops.run")
    return ConfigurationScope.from_auth(auth_context=auth_context)


Service = Annotated[MonitoringApplicationService, Depends(get_service)]
Scope = Annotated[ConfigurationScope, Depends(get_scope)]


@router.get("/business-systems", response_model=tuple[MonitoringBusinessSystemSummary, ...])
async def list_business_systems(service: Service, scope: Scope):
    return await service.list_business_systems(scope=scope)


@router.get("/sources", response_model=tuple[MonitoringSourceSummary, ...])
async def list_sources(service: Service, scope: Scope):
    return await service.list_sources(scope=scope)


@router.get("/sources/{source_id}/resources/{instance_id}/metrics", response_model=MonitoringNativeCatalog)
async def list_native_metrics(
    source_id: UUID, instance_id: UUID, service: Service, scope: Scope,
    search: str = Query(default="", max_length=128),
    offset: int = Query(default=0, ge=0, le=100000),
    limit: int = Query(default=100, ge=1, le=100),
):
    return await service.native.catalog(scope=scope, source_id=source_id, instance_id=instance_id,
        search=search, offset=offset, limit=limit)


@router.post("/sources/{source_id}/resources/{instance_id}/metric-samples", response_model=MonitoringNativeSamples)
async def query_native_metric(
    source_id: UUID, instance_id: UUID, body: MonitoringNativeQuery, service: Service, scope: Scope,
):
    return await service.native.samples(scope=scope, source_id=source_id, instance_id=instance_id, query=body)


@router.get(
    "/sources/{source_id}/instances",
    response_model=tuple[MonitoringInstanceSummary, ...],
)
async def list_instances(source_id: UUID, service: Service, scope: Scope):
    return await service.list_instances(scope=scope, source_id=source_id)


@router.get(
    "/sources/{source_id}/profiles",
    response_model=tuple[MonitoringProfileSummary, ...],
)
async def list_profiles(source_id: UUID, service: Service, scope: Scope):
    return await service.list_profiles(scope=scope, source_id=source_id)


@router.post("/sources/{source_id}/views", response_model=MonitoringView)
async def get_view(
    source_id: UUID,
    body: MonitoringViewRequest,
    service: Service,
    scope: Scope,
):
    return await service.get_view(
        scope=scope, source_id=source_id, request=body
    )


@target_router.get(
    "/{target_id}/monitoring/profiles",
    response_model=tuple[MonitoringProfileSummary, ...],
)
async def list_target_profiles(
    target_id: UUID, service: Service, scope: Scope
):
    return await service.list_target_profiles(
        scope=scope, target_id=target_id
    )


@target_router.get("/{target_id}/monitoring/view", response_model=MonitoringView)
async def get_target_view(
    target_id: UUID,
    source_id: UUID,
    profile_id: str,
    service: Service,
    scope: Scope,
    compare_source_id: UUID | None = Query(default=None),
    window: MonitoringWindowName = Query(default="1h"),
    window_start: datetime | None = Query(default=None),
    window_end: datetime | None = Query(default=None),
    baseline: MonitoringBaselinePeriod = Query(default="1w"),
):
    return await service.get_target_view(
        scope=scope,
        target_id=target_id,
        source_id=source_id,
        compare_source_id=compare_source_id,
        profile_id=profile_id,
        window=window,
        window_start=window_start,
        window_end=window_end,
        baseline=baseline,
    )
