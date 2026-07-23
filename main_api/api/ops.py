"""Portal 可见的 AIOps 配置管理接口。"""

from __future__ import annotations

from typing import Annotated, Literal, TypeVar, cast
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from pydantic import BaseModel

from platform_clients.aiops import AIOpsManagementClient
from platform_core.contracts import PUBLIC_API_V1
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


router = APIRouter(prefix=f"{PUBLIC_API_V1}/ops", tags=["AIOps Config"])
IdempotencyKey = Annotated[str, Header(alias="Idempotency-Key")]
ModelT = TypeVar("ModelT", bound=BaseModel)


def require_if_match(
    value: str | None = Header(default=None, alias="If-Match"),
) -> str:
    if value is None:
        raise HTTPException(
            status_code=428,
            detail={
                "code": "PRECONDITION_REQUIRED",
                "message": "该操作必须提供 If-Match",
            },
        )
    return value


IfMatch = Annotated[str, Depends(require_if_match)]


def _client(request: Request) -> AIOpsManagementClient:
    return cast(AIOpsManagementClient, request.app.state.aiops_client)


def _validated(
    model_type: type[ModelT],
    payload: dict,
    response: Response | None = None,
) -> ModelT:
    result = model_type.model_validate(payload)
    row_version = getattr(result, "row_version", None)
    if response is not None and row_version is not None:
        response.headers["ETag"] = f'"rv-{int(row_version)}"'
    return result


@router.post("/targets", response_model=TargetDetail, status_code=201)
async def create_target(
    body: TargetCreate,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
) -> TargetDetail:
    payload = await _client(request).create_target(
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(TargetDetail, payload, response)


@router.get("/targets", response_model=TargetPage)
async def list_targets(
    request: Request,
    resource_status: str | None = Query(default=None, alias="status"),
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> TargetPage:
    payload = await _client(request).list_targets(
        status=resource_status,
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )
    return TargetPage.model_validate(payload)


@router.get("/targets/{target_id}", response_model=TargetDetail)
async def get_target(
    target_id: UUID, request: Request, response: Response
) -> TargetDetail:
    payload = await _client(request).get_target(
        target_id, auth_context=request.state.auth_context
    )
    return _validated(TargetDetail, payload, response)


@router.patch("/targets/{target_id}", response_model=TargetDetail)
async def patch_target(
    target_id: UUID,
    body: TargetPatch,
    request: Request,
    response: Response,
    if_match: IfMatch,
) -> TargetDetail:
    payload = await _client(request).patch_target(
        target_id,
        body.model_dump(mode="json", exclude_unset=True),
        if_match=if_match,
        auth_context=request.state.auth_context,
    )
    return _validated(TargetDetail, payload, response)


async def _target_command(
    *,
    target_id: UUID,
    command: str,
    request: Request,
    response: Response,
    if_match: str,
    idempotency_key: str,
) -> TargetDetail:
    payload = await _client(request).command_target(
        target_id,
        command,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(TargetDetail, payload, response)


@router.post("/targets/{target_id}/activate", response_model=TargetDetail)
async def activate_target(
    target_id: UUID,
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> TargetDetail:
    return await _target_command(
        target_id=target_id,
        command="activate",
        request=request,
        response=response,
        if_match=if_match,
        idempotency_key=idempotency_key,
    )


@router.post("/targets/{target_id}/maintenance", response_model=TargetDetail)
async def maintain_target(
    target_id: UUID,
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> TargetDetail:
    return await _target_command(
        target_id=target_id,
        command="maintenance",
        request=request,
        response=response,
        if_match=if_match,
        idempotency_key=idempotency_key,
    )


@router.post("/targets/{target_id}/disable", response_model=TargetDetail)
async def disable_target(
    target_id: UUID,
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> TargetDetail:
    return await _target_command(
        target_id=target_id,
        command="disable",
        request=request,
        response=response,
        if_match=if_match,
        idempotency_key=idempotency_key,
    )


@router.get(
    "/targets/{target_id}/agent-bindings",
    response_model=tuple[AgentBindingView, ...],
)
async def list_agent_bindings(
    target_id: UUID, request: Request
) -> tuple[AgentBindingView, ...]:
    payload = await _client(request).list_agent_bindings(
        target_id, auth_context=request.state.auth_context
    )
    return tuple(AgentBindingView.model_validate(item) for item in payload)


@router.post(
    "/targets/{target_id}/agent-bindings",
    response_model=AgentBindingView,
    status_code=201,
)
async def create_agent_binding(
    target_id: UUID,
    body: AgentBindingCreate,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
) -> AgentBindingView:
    payload = await _client(request).create_agent_binding(
        target_id,
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(AgentBindingView, payload, response)


@router.patch(
    "/targets/{target_id}/agent-bindings/{binding_id}",
    response_model=AgentBindingView,
)
async def patch_agent_binding(
    target_id: UUID,
    binding_id: UUID,
    body: AgentBindingPatch,
    request: Request,
    response: Response,
    if_match: IfMatch,
) -> AgentBindingView:
    payload = await _client(request).patch_agent_binding(
        target_id,
        binding_id,
        body.model_dump(mode="json", exclude_unset=True),
        if_match=if_match,
        auth_context=request.state.auth_context,
    )
    return _validated(AgentBindingView, payload, response)


@router.post(
    "/targets/{target_id}/agent-bindings/{binding_id}/{command}",
    response_model=AgentBindingView,
)
async def command_agent_binding(
    target_id: UUID,
    binding_id: UUID,
    command: Literal["revoke", "restore"],
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> AgentBindingView:
    payload = await _client(request).command_agent_binding(
        target_id,
        binding_id,
        command,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(AgentBindingView, payload, response)


@router.post(
    "/monitor-sources",
    response_model=MonitorSourceDetail,
    status_code=201,
)
async def create_monitor_source(
    body: MonitorSourceCreate,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
) -> MonitorSourceDetail:
    payload = await _client(request).create_monitor_source(
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(MonitorSourceDetail, payload, response)


@router.get("/monitor-sources", response_model=MonitorSourcePage)
async def list_monitor_sources(
    request: Request,
    resource_status: str | None = Query(default=None, alias="status"),
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> MonitorSourcePage:
    payload = await _client(request).list_monitor_sources(
        status=resource_status,
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )
    return MonitorSourcePage.model_validate(payload)


@router.get("/monitor-sources/{source_id}", response_model=MonitorSourceDetail)
async def get_monitor_source(
    source_id: UUID, request: Request, response: Response
) -> MonitorSourceDetail:
    payload = await _client(request).get_monitor_source(
        source_id, auth_context=request.state.auth_context
    )
    return _validated(MonitorSourceDetail, payload, response)


@router.patch(
    "/monitor-sources/{source_id}", response_model=MonitorSourceDetail
)
async def patch_monitor_source(
    source_id: UUID,
    body: MonitorSourcePatch,
    request: Request,
    response: Response,
    if_match: IfMatch,
) -> MonitorSourceDetail:
    payload = await _client(request).patch_monitor_source(
        source_id,
        body.model_dump(mode="json", exclude_unset=True),
        if_match=if_match,
        auth_context=request.state.auth_context,
    )
    return _validated(MonitorSourceDetail, payload, response)


@router.post(
    "/monitor-sources/{source_id}/health-checks",
    response_model=HealthCheckReceipt,
    status_code=status.HTTP_202_ACCEPTED,
)
async def request_monitor_health_check(
    source_id: UUID,
    request: Request,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> HealthCheckReceipt:
    payload = await _client(request).request_monitor_health_check(
        source_id,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return HealthCheckReceipt.model_validate(payload)


@router.post(
    "/monitor-sources/{source_id}/webhook-key:rotate",
    response_model=WebhookKeyRotation,
)
async def rotate_monitor_webhook_key(
    source_id: UUID,
    request: Request,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> WebhookKeyRotation:
    payload = await _client(request).rotate_monitor_webhook_key(
        source_id,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return WebhookKeyRotation.model_validate(payload)


@router.post(
    "/monitor-sources/{source_id}/{command}",
    response_model=MonitorSourceDetail,
)
async def command_monitor_source(
    source_id: UUID,
    command: Literal["enable", "disable"],
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> MonitorSourceDetail:
    payload = await _client(request).command_monitor_source(
        source_id,
        command,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(MonitorSourceDetail, payload, response)


@router.get(
    "/targets/{target_id}/monitor-bindings",
    response_model=tuple[MonitorBindingView, ...],
)
async def list_monitor_bindings(
    target_id: UUID, request: Request
) -> tuple[MonitorBindingView, ...]:
    payload = await _client(request).list_monitor_bindings(
        target_id, auth_context=request.state.auth_context
    )
    return tuple(MonitorBindingView.model_validate(item) for item in payload)


@router.post(
    "/targets/{target_id}/monitor-bindings",
    response_model=MonitorBindingView,
    status_code=201,
)
async def create_monitor_binding(
    target_id: UUID,
    body: MonitorBindingCreate,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
) -> MonitorBindingView:
    payload = await _client(request).create_monitor_binding(
        target_id,
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(MonitorBindingView, payload, response)


@router.patch(
    "/targets/{target_id}/monitor-bindings/{binding_id}",
    response_model=MonitorBindingView,
)
async def patch_monitor_binding(
    target_id: UUID,
    binding_id: UUID,
    body: MonitorBindingPatch,
    request: Request,
    response: Response,
    if_match: IfMatch,
) -> MonitorBindingView:
    payload = await _client(request).patch_monitor_binding(
        target_id,
        binding_id,
        body.model_dump(mode="json", exclude_unset=True),
        if_match=if_match,
        auth_context=request.state.auth_context,
    )
    return _validated(MonitorBindingView, payload, response)


@router.post(
    "/targets/{target_id}/monitor-bindings/{binding_id}/{command}",
    response_model=MonitorBindingView,
)
async def command_monitor_binding(
    target_id: UUID,
    binding_id: UUID,
    command: Literal["enable", "disable"],
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> MonitorBindingView:
    payload = await _client(request).command_monitor_binding(
        target_id,
        binding_id,
        command,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(MonitorBindingView, payload, response)


@router.post("/policies", response_model=PolicyDetail, status_code=201)
async def create_policy(
    body: PolicyCreate,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
) -> PolicyDetail:
    payload = await _client(request).create_policy(
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(PolicyDetail, payload, response)


@router.get("/policies", response_model=PolicyPage)
async def list_policies(
    request: Request,
    resource_status: str | None = Query(default=None, alias="status"),
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> PolicyPage:
    payload = await _client(request).list_policies(
        status=resource_status,
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )
    return PolicyPage.model_validate(payload)


@router.get("/policies/{policy_id}", response_model=PolicyDetail)
async def get_policy(
    policy_id: UUID, request: Request, response: Response
) -> PolicyDetail:
    payload = await _client(request).get_policy(
        policy_id, auth_context=request.state.auth_context
    )
    return _validated(PolicyDetail, payload, response)


@router.post("/policies/{policy_id}/{command}", response_model=PolicyDetail)
async def command_policy(
    policy_id: UUID,
    command: Literal["activate", "retire"],
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> PolicyDetail:
    payload = await _client(request).command_policy(
        policy_id,
        command,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(PolicyDetail, payload, response)


@router.post(
    "/inspection-plans",
    response_model=InspectionPlanDetail,
    status_code=201,
)
async def create_inspection_plan(
    body: InspectionPlanCreate,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
) -> InspectionPlanDetail:
    payload = await _client(request).create_inspection_plan(
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(InspectionPlanDetail, payload, response)


@router.get("/inspection-plans", response_model=InspectionPlanPage)
async def list_inspection_plans(
    request: Request,
    resource_status: str | None = Query(default=None, alias="status"),
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> InspectionPlanPage:
    payload = await _client(request).list_inspection_plans(
        status=resource_status,
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )
    return InspectionPlanPage.model_validate(payload)


@router.get(
    "/inspection-plans/{plan_id}", response_model=InspectionPlanDetail
)
async def get_inspection_plan(
    plan_id: UUID, request: Request, response: Response
) -> InspectionPlanDetail:
    payload = await _client(request).get_inspection_plan(
        plan_id, auth_context=request.state.auth_context
    )
    return _validated(InspectionPlanDetail, payload, response)


@router.patch(
    "/inspection-plans/{plan_id}", response_model=InspectionPlanDetail
)
async def patch_inspection_plan(
    plan_id: UUID,
    body: InspectionPlanPatch,
    request: Request,
    response: Response,
    if_match: IfMatch,
) -> InspectionPlanDetail:
    payload = await _client(request).patch_inspection_plan(
        plan_id,
        body.model_dump(mode="json", exclude_unset=True),
        if_match=if_match,
        auth_context=request.state.auth_context,
    )
    return _validated(InspectionPlanDetail, payload, response)


async def _inspection_plan_command(
    *,
    plan_id: UUID,
    command: str,
    request: Request,
    response: Response,
    if_match: str,
    idempotency_key: str,
) -> InspectionPlanDetail:
    payload = await _client(request).command_inspection_plan(
        plan_id,
        command,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(InspectionPlanDetail, payload, response)


@router.post(
    "/inspection-plans/{plan_id}/activate",
    response_model=InspectionPlanDetail,
)
async def activate_inspection_plan(
    plan_id: UUID,
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> InspectionPlanDetail:
    return await _inspection_plan_command(
        plan_id=plan_id,
        command="activate",
        request=request,
        response=response,
        if_match=if_match,
        idempotency_key=idempotency_key,
    )


@router.post(
    "/inspection-plans/{plan_id}/pause",
    response_model=InspectionPlanDetail,
)
async def pause_inspection_plan(
    plan_id: UUID,
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> InspectionPlanDetail:
    return await _inspection_plan_command(
        plan_id=plan_id,
        command="pause",
        request=request,
        response=response,
        if_match=if_match,
        idempotency_key=idempotency_key,
    )


@router.post(
    "/inspection-plans/{plan_id}/disable",
    response_model=InspectionPlanDetail,
)
async def disable_inspection_plan(
    plan_id: UUID,
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> InspectionPlanDetail:
    return await _inspection_plan_command(
        plan_id=plan_id,
        command="disable",
        request=request,
        response=response,
        if_match=if_match,
        idempotency_key=idempotency_key,
    )


@router.get(
    "/inspection-plans/{plan_id}/targets",
    response_model=tuple[InspectionTargetView, ...],
)
async def list_inspection_targets(
    plan_id: UUID, request: Request
) -> tuple[InspectionTargetView, ...]:
    payload = await _client(request).list_inspection_targets(
        plan_id, auth_context=request.state.auth_context
    )
    return tuple(InspectionTargetView.model_validate(item) for item in payload)


@router.post(
    "/inspection-plans/{plan_id}/targets",
    response_model=InspectionTargetView,
    status_code=201,
)
async def add_inspection_target(
    plan_id: UUID,
    body: InspectionTargetCreate,
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> InspectionTargetView:
    payload = await _client(request).add_inspection_target(
        plan_id,
        body.model_dump(mode="json"),
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    result = InspectionTargetView.model_validate(payload)
    current = int(if_match.strip('"')[3:])
    response.headers["ETag"] = f'"rv-{current + 1}"'
    return result


@router.patch(
    "/inspection-plans/{plan_id}/targets/{plan_target_id}",
    response_model=InspectionTargetView,
)
async def patch_inspection_target(
    plan_id: UUID,
    plan_target_id: UUID,
    body: InspectionTargetPatch,
    request: Request,
    response: Response,
    if_match: IfMatch,
) -> InspectionTargetView:
    payload = await _client(request).patch_inspection_target(
        plan_id,
        plan_target_id,
        body.model_dump(mode="json", exclude_unset=True),
        if_match=if_match,
        auth_context=request.state.auth_context,
    )
    result = InspectionTargetView.model_validate(payload)
    current = int(if_match.strip('"')[3:])
    response.headers["ETag"] = f'"rv-{current + 1}"'
    return result
