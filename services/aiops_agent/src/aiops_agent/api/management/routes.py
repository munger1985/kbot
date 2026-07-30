"""AIOps 配置管理 Internal API。"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Request, Response, status

from aiops_agent.api.dependencies import (
    get_aiops_auth_context,
    require_service_scope,
)
from aiops_agent.application.configuration import (
    AIOpsConfigurationService,
    ConfigurationScope,
)
from aiops_agent.application.configuration.common import (
    format_etag,
    parse_etag,
)
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops import (
    AgentBindingCreate,
    AgentBindingPatch,
    AgentBindingView,
    HealthCheckReceipt,
    InspectionPlanCreate,
    InspectionPlanDetail,
    InspectionPlanPage,
    InspectionPlanPatch,
    InspectionTargetCreate,
    InspectionTargetPatch,
    InspectionTargetView,
    MonitorBindingCreate,
    MonitorBindingPatch,
    MonitorBindingView,
    MonitorSourceCreate,
    MonitorSourceDetail,
    MonitorSourcePage,
    MonitorSourcePatch,
    PolicyCreate,
    PolicyDetail,
    PolicyPage,
    TargetCreate,
    TargetDetail,
    TargetPage,
    TargetPatch,
    WebhookKeyRotation,
)


router = APIRouter(prefix="/internal/v1/aiops/config", tags=["AIOps Config"])
IdempotencyKey = Annotated[str, Header(alias="Idempotency-Key")]
IfMatch = Annotated[str | None, Header(alias="If-Match")]


def get_service(request: Request) -> AIOpsConfigurationService:
    return request.app.state.configuration_service


def get_scope(
    request: Request,
    auth_context: AuthContext = Depends(get_aiops_auth_context),
) -> ConfigurationScope:
    require_service_scope(request, "aiops.manage")
    return ConfigurationScope.from_auth(
        auth_context=auth_context,
    )


Service = Annotated[AIOpsConfigurationService, Depends(get_service)]
Scope = Annotated[ConfigurationScope, Depends(get_scope)]
Auth = Annotated[AuthContext, Depends(get_aiops_auth_context)]


def _etag(response: Response, row_version: int) -> None:
    response.headers["ETag"] = format_etag(row_version)


@router.post("/targets", response_model=TargetDetail, status_code=201)
async def create_target(
    body: TargetCreate,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
) -> TargetDetail:
    result = await service.create_target(
        scope=scope,
        request=body,
        idempotency_key=idempotency_key,
    )
    _etag(response, result.row_version)
    return result


@router.get("/targets", response_model=TargetPage)
async def list_targets(
    service: Service,
    scope: Scope,
    resource_status: str | None = Query(default=None, alias="status"),
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> TargetPage:
    return await service.list_targets(
        scope=scope,
        status=resource_status,
        cursor=cursor,
        limit=limit,
    )


@router.get("/targets/{target_id}", response_model=TargetDetail)
async def get_target(
    target_id: UUID,
    response: Response,
    service: Service,
    scope: Scope,
) -> TargetDetail:
    result = await service.get_target(scope=scope, target_id=target_id)
    _etag(response, result.row_version)
    return result


@router.patch("/targets/{target_id}", response_model=TargetDetail)
async def patch_target(
    target_id: UUID,
    body: TargetPatch,
    response: Response,
    service: Service,
    scope: Scope,
    if_match: IfMatch = None,
) -> TargetDetail:
    result = await service.patch_target(
        scope=scope,
        target_id=target_id,
        request=body,
        expected_version=parse_etag(if_match),
    )
    _etag(response, result.row_version)
    return result


async def _command_target(
    target_id: UUID,
    command: str,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> TargetDetail:
    result = await service.command_target(
        scope=scope,
        target_id=target_id,
        command=command,
        expected_version=parse_etag(if_match),
        idempotency_key=idempotency_key,
    )
    _etag(response, result.row_version)
    return result


@router.post("/targets/{target_id}/activate", response_model=TargetDetail)
async def activate_target(
    target_id: UUID,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> TargetDetail:
    return await _command_target(
        target_id, "activate", response, service, scope, idempotency_key, if_match
    )


@router.post("/targets/{target_id}/maintenance", response_model=TargetDetail)
async def maintain_target(
    target_id: UUID,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> TargetDetail:
    return await _command_target(
        target_id,
        "maintenance",
        response,
        service,
        scope,
        idempotency_key,
        if_match,
    )


@router.post("/targets/{target_id}/disable", response_model=TargetDetail)
async def disable_target(
    target_id: UUID,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> TargetDetail:
    return await _command_target(
        target_id, "disable", response, service, scope, idempotency_key, if_match
    )


@router.get(
    "/targets/{target_id}/agent-bindings",
    response_model=tuple[AgentBindingView, ...],
)
async def list_agent_bindings(
    target_id: UUID, service: Service, scope: Scope
) -> tuple[AgentBindingView, ...]:
    return await service.list_agent_bindings(scope=scope, target_id=target_id)


@router.post(
    "/targets/{target_id}/agent-bindings",
    response_model=AgentBindingView,
    status_code=201,
)
async def create_agent_binding(
    target_id: UUID,
    body: AgentBindingCreate,
    response: Response,
    service: Service,
    scope: Scope,
    auth_context: Auth,
    idempotency_key: IdempotencyKey,
) -> AgentBindingView:
    result = await service.create_agent_binding(
        scope=scope,
        auth_context=auth_context,
        target_id=target_id,
        request=body,
        idempotency_key=idempotency_key,
    )
    _etag(response, result.row_version)
    return result


@router.patch(
    "/targets/{target_id}/agent-bindings/{binding_id}",
    response_model=AgentBindingView,
)
async def patch_agent_binding(
    target_id: UUID,
    binding_id: UUID,
    body: AgentBindingPatch,
    response: Response,
    service: Service,
    scope: Scope,
    auth_context: Auth,
    if_match: IfMatch = None,
) -> AgentBindingView:
    result = await service.patch_agent_binding(
        scope=scope,
        auth_context=auth_context,
        target_id=target_id,
        binding_id=binding_id,
        request=body,
        expected_version=parse_etag(if_match),
    )
    _etag(response, result.row_version)
    return result


@router.post(
    "/targets/{target_id}/agent-bindings/{binding_id}/{command}",
    response_model=AgentBindingView,
)
async def command_agent_binding(
    target_id: UUID,
    binding_id: UUID,
    command: str,
    response: Response,
    service: Service,
    scope: Scope,
    auth_context: Auth,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> AgentBindingView:
    result = await service.command_agent_binding(
        scope=scope,
        auth_context=auth_context,
        target_id=target_id,
        binding_id=binding_id,
        command=command,
        expected_version=parse_etag(if_match),
        idempotency_key=idempotency_key,
    )
    _etag(response, result.row_version)
    return result


@router.post(
    "/monitor-sources",
    response_model=MonitorSourceDetail,
    status_code=201,
)
async def create_monitor_source(
    body: MonitorSourceCreate,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
) -> MonitorSourceDetail:
    result = await service.create_monitor_source(
        scope=scope,
        request=body,
        idempotency_key=idempotency_key,
    )
    _etag(response, result.row_version)
    return result


@router.get("/monitor-sources", response_model=MonitorSourcePage)
async def list_monitor_sources(
    service: Service,
    scope: Scope,
    resource_status: str | None = Query(default=None, alias="status"),
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> MonitorSourcePage:
    return await service.list_monitor_sources(
        scope=scope,
        status=resource_status,
        cursor=cursor,
        limit=limit,
    )


@router.get(
    "/monitor-sources/{source_id}", response_model=MonitorSourceDetail
)
async def get_monitor_source(
    source_id: UUID,
    response: Response,
    service: Service,
    scope: Scope,
) -> MonitorSourceDetail:
    result = await service.get_monitor_source(
        scope=scope, source_id=source_id
    )
    _etag(response, result.row_version)
    return result


@router.patch(
    "/monitor-sources/{source_id}", response_model=MonitorSourceDetail
)
async def patch_monitor_source(
    source_id: UUID,
    body: MonitorSourcePatch,
    response: Response,
    service: Service,
    scope: Scope,
    if_match: IfMatch = None,
) -> MonitorSourceDetail:
    result = await service.patch_monitor_source(
        scope=scope,
        source_id=source_id,
        request=body,
        expected_version=parse_etag(if_match),
    )
    _etag(response, result.row_version)
    return result


@router.post(
    "/monitor-sources/{source_id}/health-checks",
    response_model=HealthCheckReceipt,
    status_code=status.HTTP_202_ACCEPTED,
)
async def request_monitor_health_check(
    source_id: UUID,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> HealthCheckReceipt:
    return await service.request_monitor_health_check(
        scope=scope,
        source_id=source_id,
        expected_version=parse_etag(if_match),
        idempotency_key=idempotency_key,
    )


@router.post(
    "/monitor-sources/{source_id}/webhook-key:rotate",
    response_model=WebhookKeyRotation,
)
async def rotate_webhook_key(
    source_id: UUID,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> WebhookKeyRotation:
    return await service.rotate_webhook_key(
        scope=scope,
        source_id=source_id,
        expected_version=parse_etag(if_match),
        idempotency_key=idempotency_key,
    )


@router.post(
    "/monitor-sources/{source_id}/{command}",
    response_model=MonitorSourceDetail,
)
async def command_monitor_source(
    source_id: UUID,
    command: str,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> MonitorSourceDetail:
    result = await service.command_monitor_source(
        scope=scope,
        source_id=source_id,
        command=command,
        expected_version=parse_etag(if_match),
        idempotency_key=idempotency_key,
    )
    _etag(response, result.row_version)
    return result


@router.get(
    "/targets/{target_id}/monitor-bindings",
    response_model=tuple[MonitorBindingView, ...],
)
async def list_monitor_bindings(
    target_id: UUID, service: Service, scope: Scope
) -> tuple[MonitorBindingView, ...]:
    return await service.list_monitor_bindings(scope=scope, target_id=target_id)


@router.post(
    "/targets/{target_id}/monitor-bindings",
    response_model=MonitorBindingView,
    status_code=201,
)
async def create_monitor_binding(
    target_id: UUID,
    body: MonitorBindingCreate,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
) -> MonitorBindingView:
    result = await service.create_monitor_binding(
        scope=scope,
        target_id=target_id,
        request=body,
        idempotency_key=idempotency_key,
    )
    _etag(response, result.row_version)
    return result


@router.patch(
    "/targets/{target_id}/monitor-bindings/{binding_id}",
    response_model=MonitorBindingView,
)
async def patch_monitor_binding(
    target_id: UUID,
    binding_id: UUID,
    body: MonitorBindingPatch,
    response: Response,
    service: Service,
    scope: Scope,
    if_match: IfMatch = None,
) -> MonitorBindingView:
    result = await service.patch_monitor_binding(
        scope=scope,
        target_id=target_id,
        binding_id=binding_id,
        request=body,
        expected_version=parse_etag(if_match),
    )
    _etag(response, result.row_version)
    return result


@router.post(
    "/targets/{target_id}/monitor-bindings/{binding_id}/{command}",
    response_model=MonitorBindingView,
)
async def command_monitor_binding(
    target_id: UUID,
    binding_id: UUID,
    command: str,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> MonitorBindingView:
    result = await service.command_monitor_binding(
        scope=scope,
        target_id=target_id,
        binding_id=binding_id,
        command=command,
        expected_version=parse_etag(if_match),
        idempotency_key=idempotency_key,
    )
    _etag(response, result.row_version)
    return result


@router.post("/policies", response_model=PolicyDetail, status_code=201)
async def create_policy(
    body: PolicyCreate,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
) -> PolicyDetail:
    result = await service.create_policy(
        scope=scope,
        request=body,
        idempotency_key=idempotency_key,
    )
    _etag(response, result.row_version)
    return result


@router.get("/policies", response_model=PolicyPage)
async def list_policies(
    service: Service,
    scope: Scope,
    resource_status: str | None = Query(default=None, alias="status"),
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> PolicyPage:
    return await service.list_policies(
        scope=scope,
        status=resource_status,
        cursor=cursor,
        limit=limit,
    )


@router.get("/policies/{policy_id}", response_model=PolicyDetail)
async def get_policy(
    policy_id: UUID,
    response: Response,
    service: Service,
    scope: Scope,
) -> PolicyDetail:
    result = await service.get_policy(scope=scope, policy_id=policy_id)
    _etag(response, result.row_version)
    return result


@router.post("/policies/{policy_id}/{command}", response_model=PolicyDetail)
async def command_policy(
    policy_id: UUID,
    command: str,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> PolicyDetail:
    result = await service.command_policy(
        scope=scope,
        policy_id=policy_id,
        command=command,
        expected_version=parse_etag(if_match),
        idempotency_key=idempotency_key,
    )
    _etag(response, result.row_version)
    return result


@router.post(
    "/inspection-plans",
    response_model=InspectionPlanDetail,
    status_code=201,
)
async def create_inspection_plan(
    body: InspectionPlanCreate,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
) -> InspectionPlanDetail:
    result = await service.create_inspection_plan(
        scope=scope,
        request=body,
        idempotency_key=idempotency_key,
    )
    _etag(response, result.row_version)
    return result


@router.get("/inspection-plans", response_model=InspectionPlanPage)
async def list_inspection_plans(
    service: Service,
    scope: Scope,
    resource_status: str | None = Query(default=None, alias="status"),
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> InspectionPlanPage:
    return await service.list_inspection_plans(
        scope=scope,
        status=resource_status,
        cursor=cursor,
        limit=limit,
    )


@router.get(
    "/inspection-plans/{plan_id}", response_model=InspectionPlanDetail
)
async def get_inspection_plan(
    plan_id: UUID,
    response: Response,
    service: Service,
    scope: Scope,
) -> InspectionPlanDetail:
    result = await service.get_inspection_plan(scope=scope, plan_id=plan_id)
    _etag(response, result.row_version)
    return result


@router.patch(
    "/inspection-plans/{plan_id}", response_model=InspectionPlanDetail
)
async def patch_inspection_plan(
    plan_id: UUID,
    body: InspectionPlanPatch,
    response: Response,
    service: Service,
    scope: Scope,
    if_match: IfMatch = None,
) -> InspectionPlanDetail:
    result = await service.patch_inspection_plan(
        scope=scope,
        plan_id=plan_id,
        request=body,
        expected_version=parse_etag(if_match),
    )
    _etag(response, result.row_version)
    return result


async def _command_inspection_plan(
    plan_id: UUID,
    command: str,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> InspectionPlanDetail:
    result = await service.command_inspection_plan(
        scope=scope,
        plan_id=plan_id,
        command=command,
        expected_version=parse_etag(if_match),
        idempotency_key=idempotency_key,
    )
    _etag(response, result.row_version)
    return result


@router.post(
    "/inspection-plans/{plan_id}/activate",
    response_model=InspectionPlanDetail,
)
async def activate_inspection_plan(
    plan_id: UUID,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> InspectionPlanDetail:
    return await _command_inspection_plan(
        plan_id, "activate", response, service, scope, idempotency_key, if_match
    )


@router.post(
    "/inspection-plans/{plan_id}/pause",
    response_model=InspectionPlanDetail,
)
async def pause_inspection_plan(
    plan_id: UUID,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> InspectionPlanDetail:
    return await _command_inspection_plan(
        plan_id, "pause", response, service, scope, idempotency_key, if_match
    )


@router.post(
    "/inspection-plans/{plan_id}/disable",
    response_model=InspectionPlanDetail,
)
async def disable_inspection_plan(
    plan_id: UUID,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> InspectionPlanDetail:
    return await _command_inspection_plan(
        plan_id, "disable", response, service, scope, idempotency_key, if_match
    )


@router.get(
    "/inspection-plans/{plan_id}/targets",
    response_model=tuple[InspectionTargetView, ...],
)
async def list_inspection_targets(
    plan_id: UUID, service: Service, scope: Scope
) -> tuple[InspectionTargetView, ...]:
    return await service.list_inspection_targets(scope=scope, plan_id=plan_id)


@router.post(
    "/inspection-plans/{plan_id}/targets",
    response_model=InspectionTargetView,
    status_code=201,
)
async def add_inspection_target(
    plan_id: UUID,
    body: InspectionTargetCreate,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> InspectionTargetView:
    expected = parse_etag(if_match)
    result = await service.add_inspection_target(
        scope=scope,
        plan_id=plan_id,
        request=body,
        expected_plan_version=expected,
        idempotency_key=idempotency_key,
    )
    _etag(response, expected + 1)
    return result


@router.patch(
    "/inspection-plans/{plan_id}/targets/{plan_target_id}",
    response_model=InspectionTargetView,
)
async def patch_inspection_target(
    plan_id: UUID,
    plan_target_id: UUID,
    body: InspectionTargetPatch,
    response: Response,
    service: Service,
    scope: Scope,
    if_match: IfMatch = None,
) -> InspectionTargetView:
    expected = parse_etag(if_match)
    result = await service.patch_inspection_target(
        scope=scope,
        plan_id=plan_id,
        plan_target_id=plan_target_id,
        request=body,
        expected_plan_version=expected,
    )
    _etag(response, expected + 1)
    return result
