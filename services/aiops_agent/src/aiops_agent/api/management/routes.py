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
from aiops_agent.application.configuration.source_connection_test import (
    test_diagnostic_source_connection as run_source_connection_test,
)
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops import (
    AgentBindingCreate,
    AgentBindingPatch,
    AgentBindingView,
    ConnectivityCheckReceipt,
    InspectionPlanCreate,
    InspectionPlanDetail,
    InspectionPlanPage,
    InspectionPlanPatch,
    NotificationSubscriptionList,
    NotificationSubscriptionUpsert,
    NotificationSubscriptionView,
    SourceBindingCreate,
    SourceBindingPatch,
    SourceBindingView,
    DiagnosticSourceCreate,
    DiagnosticSourceConnectionTestResult,
    DiagnosticSourceDetail,
    DiagnosticSourcePage,
    DiagnosticSourcePatch,
    PolicyCreate,
    PolicyDetail,
    PolicyPage,
    TargetCreate,
    TargetConnectionTest,
    TargetConnectionTestResult,
    DatabaseCredentialInput,
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


@router.get(
    "/notification-subscriptions",
    response_model=NotificationSubscriptionList,
)
async def list_notification_subscriptions(
    service: Service,
    scope: Scope,
) -> NotificationSubscriptionList:
    return await service.list_notification_subscriptions(scope=scope)


@router.put(
    "/notification-subscriptions/targets/{target_id}",
    response_model=NotificationSubscriptionView,
)
async def upsert_notification_subscription(
    target_id: UUID,
    body: NotificationSubscriptionUpsert,
    response: Response,
    service: Service,
    scope: Scope,
    if_match: IfMatch = None,
) -> NotificationSubscriptionView:
    result = await service.upsert_notification_subscription(
        scope=scope,
        target_id=target_id,
        request=body,
        expected_version=(parse_etag(if_match) if if_match else None),
    )
    _etag(response, result.row_version)
    return result


@router.delete(
    "/notification-subscriptions/targets/{target_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def disable_notification_subscription(
    target_id: UUID,
    service: Service,
    scope: Scope,
    if_match: IfMatch = None,
) -> Response:
    await service.disable_notification_subscription(
        scope=scope,
        target_id=target_id,
        expected_version=parse_etag(if_match),
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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


@router.post(
    "/targets/test-connection",
    response_model=TargetConnectionTestResult,
)
async def test_target_connection(
    body: TargetConnectionTest,
    service: Service,
    scope: Scope,
) -> TargetConnectionTestResult:
    return await service.test_target_connection(scope=scope, request=body)


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


@router.post("/targets/{target_id}/diagnostic-credential:rotate", response_model=TargetDetail)
async def rotate_diagnostic_credential(target_id: UUID, body: DatabaseCredentialInput, response: Response, service: Service, scope: Scope, idempotency_key: IdempotencyKey, if_match: IfMatch = None) -> TargetDetail:
    result = await service.rotate_target_credential(scope=scope, target_id=target_id, credential_kind="DIAGNOSTIC", username=body.username, password=body.password, expected_version=parse_etag(if_match), idempotency_key=idempotency_key)
    _etag(response, result.row_version)
    return result


@router.post("/targets/{target_id}/execution-credential:rotate", response_model=TargetDetail)
async def rotate_execution_credential(target_id: UUID, body: DatabaseCredentialInput, response: Response, service: Service, scope: Scope, idempotency_key: IdempotencyKey, if_match: IfMatch = None) -> TargetDetail:
    result = await service.rotate_target_credential(scope=scope, target_id=target_id, credential_kind="EXECUTION", username=body.username, password=body.password, expected_version=parse_etag(if_match), idempotency_key=idempotency_key)
    _etag(response, result.row_version)
    return result


@router.post("/targets/{target_id}/execution-credential:remove", response_model=TargetDetail)
async def remove_execution_credential(target_id: UUID, response: Response, service: Service, scope: Scope, idempotency_key: IdempotencyKey, if_match: IfMatch = None) -> TargetDetail:
    result = await service.remove_execution_credential(scope=scope, target_id=target_id, expected_version=parse_etag(if_match), idempotency_key=idempotency_key)
    _etag(response, result.row_version)
    return result


@router.post("/targets/{target_id}/diagnostic-credential:remove", response_model=TargetDetail)
async def remove_diagnostic_credential(target_id: UUID, response: Response, service: Service, scope: Scope, idempotency_key: IdempotencyKey, if_match: IfMatch = None) -> TargetDetail:
    result = await service.remove_diagnostic_credential(scope=scope, target_id=target_id, expected_version=parse_etag(if_match), idempotency_key=idempotency_key)
    _etag(response, result.row_version)
    return result


@router.delete("/targets/{target_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_target(target_id: UUID, service: Service, scope: Scope, idempotency_key: IdempotencyKey, if_match: IfMatch = None) -> Response:
    await service.delete_target(scope=scope, target_id=target_id, expected_version=parse_etag(if_match), idempotency_key=idempotency_key)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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


@router.post("/targets/{target_id}/enable", response_model=TargetDetail)
async def enable_target(
    target_id: UUID,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> TargetDetail:
    return await _command_target(
        target_id, "enable", response, service, scope, idempotency_key, if_match
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


@router.post(
    "/targets/{target_id}/connectivity-checks",
    response_model=TargetDetail,
    status_code=status.HTTP_202_ACCEPTED,
)
async def request_target_connectivity_check(
    target_id: UUID,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> TargetDetail:
    result = await service.request_target_connectivity_check(
        scope=scope,
        target_id=target_id,
        expected_version=parse_etag(if_match),
        idempotency_key=idempotency_key,
    )
    _etag(response, result.row_version)
    return result


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
    "/diagnostic-sources",
    response_model=DiagnosticSourceDetail,
    status_code=201,
)
async def create_diagnostic_source(
    body: DiagnosticSourceCreate,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
) -> DiagnosticSourceDetail:
    result = await service.create_diagnostic_source(
        scope=scope,
        request=body,
        idempotency_key=idempotency_key,
    )
    _etag(response, result.row_version)
    return result


@router.post(
    "/diagnostic-sources/test-connection",
    response_model=DiagnosticSourceConnectionTestResult,
)
async def test_diagnostic_source_connection(
    body: DiagnosticSourceCreate,
    request: Request,
    scope: Scope,
) -> DiagnosticSourceConnectionTestResult:
    del scope
    return await run_source_connection_test(
        body,
        diagnostic_source_registry=(
            request.app.state.diagnostic_source_registry
        ),
    )


@router.get("/diagnostic-sources", response_model=DiagnosticSourcePage)
async def list_diagnostic_sources(
    service: Service,
    scope: Scope,
    resource_status: str | None = Query(default=None, alias="status"),
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> DiagnosticSourcePage:
    return await service.list_diagnostic_sources(
        scope=scope,
        status=resource_status,
        cursor=cursor,
        limit=limit,
    )


@router.get(
    "/diagnostic-sources/{source_id}", response_model=DiagnosticSourceDetail
)
async def get_diagnostic_source(
    source_id: UUID,
    response: Response,
    service: Service,
    scope: Scope,
) -> DiagnosticSourceDetail:
    result = await service.get_diagnostic_source(
        scope=scope, source_id=source_id
    )
    _etag(response, result.row_version)
    return result


@router.patch(
    "/diagnostic-sources/{source_id}", response_model=DiagnosticSourceDetail
)
async def patch_diagnostic_source(
    source_id: UUID,
    body: DiagnosticSourcePatch,
    response: Response,
    service: Service,
    scope: Scope,
    if_match: IfMatch = None,
) -> DiagnosticSourceDetail:
    result = await service.patch_diagnostic_source(
        scope=scope,
        source_id=source_id,
        request=body,
        expected_version=parse_etag(if_match),
    )
    _etag(response, result.row_version)
    return result


@router.delete(
    "/diagnostic-sources/{source_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_diagnostic_source(
    source_id: UUID,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> Response:
    await service.delete_diagnostic_source(
        scope=scope,
        source_id=source_id,
        expected_version=parse_etag(if_match),
        idempotency_key=idempotency_key,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/diagnostic-sources/{source_id}/connectivity-checks",
    response_model=ConnectivityCheckReceipt,
    status_code=status.HTTP_202_ACCEPTED,
)
async def request_diagnostic_source_connectivity_check(
    source_id: UUID,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> ConnectivityCheckReceipt:
    return await service.request_diagnostic_source_connectivity_check(
        scope=scope,
        source_id=source_id,
        expected_version=parse_etag(if_match),
        idempotency_key=idempotency_key,
    )


@router.post(
    "/diagnostic-sources/{source_id}/webhook-key:rotate",
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
    "/diagnostic-sources/{source_id}/{command}",
    response_model=DiagnosticSourceDetail,
)
async def command_diagnostic_source(
    source_id: UUID,
    command: str,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> DiagnosticSourceDetail:
    result = await service.command_diagnostic_source(
        scope=scope,
        source_id=source_id,
        command=command,
        expected_version=parse_etag(if_match),
        idempotency_key=idempotency_key,
    )
    _etag(response, result.row_version)
    return result


@router.get(
    "/targets/{target_id}/source-bindings",
    response_model=tuple[SourceBindingView, ...],
)
async def list_source_bindings(
    target_id: UUID, service: Service, scope: Scope
) -> tuple[SourceBindingView, ...]:
    return await service.list_source_bindings(scope=scope, target_id=target_id)


@router.post(
    "/targets/{target_id}/source-bindings",
    response_model=SourceBindingView,
    status_code=201,
)
async def create_source_binding(
    target_id: UUID,
    body: SourceBindingCreate,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
) -> SourceBindingView:
    result = await service.create_source_binding(
        scope=scope,
        target_id=target_id,
        request=body,
        idempotency_key=idempotency_key,
    )
    _etag(response, result.row_version)
    return result


@router.patch(
    "/targets/{target_id}/source-bindings/{binding_id}",
    response_model=SourceBindingView,
)
async def patch_source_binding(
    target_id: UUID,
    binding_id: UUID,
    body: SourceBindingPatch,
    response: Response,
    service: Service,
    scope: Scope,
    if_match: IfMatch = None,
) -> SourceBindingView:
    result = await service.patch_source_binding(
        scope=scope,
        target_id=target_id,
        binding_id=binding_id,
        request=body,
        expected_version=parse_etag(if_match),
    )
    _etag(response, result.row_version)
    return result


@router.post(
    "/targets/{target_id}/source-bindings/{binding_id}/{command}",
    response_model=SourceBindingView,
)
async def command_source_binding(
    target_id: UUID,
    binding_id: UUID,
    command: str,
    response: Response,
    service: Service,
    scope: Scope,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch = None,
) -> SourceBindingView:
    result = await service.command_source_binding(
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
