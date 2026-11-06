"""通知渠道治理 HTTP 合同；身份与权限由各平台认证层解析。"""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from platform_core.notifications.delivery import (
    BindingConfirmation, BindingCreate, ConnectionCreate, ConnectionUpdate, ExternalPolicyCreate, RetryRequest, DeliveryStatus,
)
from main_api.application.notification_delivery import ExternalDeliveryError
from main_api.api.notification_delivery_auth import external_context, external_entry_actor, external_capabilities

router = APIRouter(prefix="/api/v1/notification-delivery", tags=["Notification Delivery"])


@router.get("/capabilities")
async def capabilities(request: Request):
    return await external_capabilities(request)


@router.get("/catalog")
async def catalog(request: Request):
    await external_capabilities(request)
    from platform_core.notifications.delivery import EXTERNAL_EVENTS
    return {"events": EXTERNAL_EVENTS}


class VersionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_row_version: int = Field(ge=1)


class TestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    binding_id: UUID


class PolicyStateRequest(VersionRequest):
    enabled: bool


@router.put("/policies/{object_id}")
async def set_policy(object_id: UUID, body: PolicyStateRequest, request: Request, domain_id: str = Query(min_length=1)):
    return await _invoke(request, "set_policy_enabled", permission="channel", domain_id=domain_id,
        object_id=object_id, expected_version=body.expected_row_version, enabled=body.enabled)


@router.get("/entries/{entry_id}")
async def resolve_entry(entry_id: UUID, request: Request):
    actor_id = await external_entry_actor(request)
    try:
        return await request.app.state.external_notification_service.resolve_entry(actor_id=actor_id, entry_id=entry_id)
    except ExternalDeliveryError as exc:
        raise HTTPException(exc.status_code, {"code": exc.code, "message": "入口已过期、不可用或当前账号无权访问"}) from exc


@router.get("/me/bindings")
async def my_bindings(request: Request, domain_id: str = Query(min_length=1)):
    return await _invoke(request, "my_bindings", permission="self", domain_id=domain_id)


@router.post("/connections/{object_id}/test")
async def queue_test(object_id: UUID, body: TestRequest, request: Request, domain_id: str = Query(min_length=1)):
    return await _invoke(request, "queue_test", permission="channel", domain_id=domain_id,
        object_id=object_id, binding_id=body.binding_id)


async def _invoke(request, method, *, permission, domain_id, **kwargs):
    scope, actor_id = await external_context(request, permission=permission, domain_id=domain_id)
    try:
        return await getattr(request.app.state.external_notification_service, method)(
            scope=scope, actor_id=actor_id, **kwargs)
    except ExternalDeliveryError as exc:
        raise HTTPException(exc.status_code, {"code": exc.code, "message": "操作未完成，请核对状态与权限"}) from exc


async def _list(request, kind, permission, domain_id, limit):
    scope, _ = await external_context(request, permission=permission, domain_id=domain_id)
    return {"items": await request.app.state.external_notification_service.list(kind, scope=scope, limit=limit)}


@router.get("/connections")
async def connections(request: Request, domain_id: str = Query(min_length=1), limit: int = Query(100, ge=1, le=100)):
    return await _list(request, "connection", "channel", domain_id, limit)


@router.get("/targets")
async def targets(request: Request, domain_id: str = Query(min_length=1)):
    # 渠道管理员选试发/策略目标，绑定管理员选父连接；不扩大写权限。
    try:
        scope, _ = await external_context(request, permission="channel", domain_id=domain_id)
    except HTTPException as exc:
        if exc.status_code != 403:
            raise
        scope, _ = await external_context(request, permission="binding", domain_id=domain_id)
    service = request.app.state.external_notification_service
    return {"connections": await service.list("connection", scope=scope),
            "bindings": await service.list("binding", scope=scope)}


@router.post("/connections", status_code=201, openapi_extra={"x-sensitive-input": True})
async def create_connection(body: ConnectionCreate, request: Request, domain_id: str = Query(min_length=1)):
    return await _invoke(request, "create_connection", permission="channel", domain_id=domain_id, body=body)


@router.put("/connections/{object_id}", openapi_extra={"x-sensitive-input": True})
async def update_connection(object_id: UUID, body: ConnectionUpdate, request: Request, domain_id: str = Query(min_length=1)):
    return await _invoke(request, "update_connection", permission="channel", domain_id=domain_id, object_id=object_id, body=body)


@router.post("/connections/{object_id}/check")
async def check_connection(object_id: UUID, request: Request, domain_id: str = Query(min_length=1)):
    return await _invoke(request, "check_connection", permission="channel", domain_id=domain_id, object_id=object_id)


@router.get("/bindings")
async def bindings(request: Request, domain_id: str = Query(min_length=1), limit: int = Query(100, ge=1, le=100)):
    return await _list(request, "binding", "binding", domain_id, limit)


@router.post("/bindings", status_code=201)
async def create_binding(body: BindingCreate, request: Request, domain_id: str = Query(min_length=1)):
    return await _invoke(request, "create_binding", permission="binding", domain_id=domain_id, body=body)


@router.post("/bindings/{object_id}/disable")
async def disable_binding(object_id: UUID, body: VersionRequest, request: Request, domain_id: str = Query(min_length=1)):
    return await _invoke(request, "disable_binding", permission="binding", domain_id=domain_id,
        object_id=object_id, expected_version=body.expected_row_version)


@router.post("/bindings/{object_id}/challenge")
async def challenge(object_id: UUID, request: Request, domain_id: str = Query(min_length=1)):
    return await _invoke(request, "challenge", permission="binding", domain_id=domain_id, object_id=object_id)


@router.post("/bindings/{object_id}/confirm", openapi_extra={"x-sensitive-input": True})
async def confirm(object_id: UUID, body: BindingConfirmation, request: Request, domain_id: str = Query(min_length=1)):
    return await _invoke(request, "confirm", permission="self", domain_id=domain_id, object_id=object_id, body=body)


@router.get("/policies")
async def policies(request: Request, domain_id: str = Query(min_length=1), limit: int = Query(100, ge=1, le=100)):
    return await _list(request, "policy", "channel", domain_id, limit)


@router.post("/policies", status_code=201)
async def create_policy(body: ExternalPolicyCreate, request: Request, domain_id: str = Query(min_length=1)):
    return await _invoke(request, "create_policy", permission="channel", domain_id=domain_id, body=body)


@router.get("/jobs")
async def jobs(request: Request, domain_id: str = Query(min_length=1), limit: int = Query(100, ge=1, le=100),
               offset: int = Query(0, ge=0, le=1000000), status: DeliveryStatus | None = None):
    scope, _ = await external_context(request, permission="read", domain_id=domain_id)
    return {"items": await request.app.state.external_notification_service.list(
        "job", scope=scope, limit=limit, offset=offset, status=status)}


@router.get("/jobs/{job_id}/attempts")
async def attempts(job_id: UUID, request: Request, domain_id: str = Query(min_length=1)):
    scope, _ = await external_context(request, permission="read", domain_id=domain_id)
    return {"items": await request.app.state.external_notification_service.attempt_history(scope=scope, job_id=job_id)}


@router.post("/jobs/{job_id}/retry")
async def retry(job_id: UUID, body: RetryRequest, request: Request, domain_id: str = Query(min_length=1)):
    return await _invoke(request, "retry", permission="retry", domain_id=domain_id, job_id=job_id, body=body)


@router.get("/audit")
async def audit(request: Request, domain_id: str = Query(min_length=1), limit: int = Query(100, ge=1, le=100)):
    return await _list(request, "audit", "read", domain_id, limit)


@router.get("/health")
async def health(request: Request, domain_id: str = Query(min_length=1)):
    scope, _ = await external_context(request, permission="read", domain_id=domain_id)
    return await request.app.state.external_notification_service.health(scope=scope)


@router.get("/open/{entry_id}")
async def open_entry(entry_id: UUID, request: Request, domain_id: str = Query(min_length=1)):
    return await _invoke(request, "open_entry", permission="self", domain_id=domain_id, entry_id=entry_id)
