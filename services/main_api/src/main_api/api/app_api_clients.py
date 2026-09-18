"""App 管理员维护本 App 的第三方 API Client。"""

from datetime import datetime
from typing import Literal, cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from main_api.application import (
    AppApiKeyError,
    AppApiKeyService,
    authorize_app_request,
)
from platform_core.authorization import resolve_business_app_id
from platform_core.contracts import (
    IdentityEntryKind,
    PrincipalKind,
    PUBLIC_API_V1,
)
from platform_core.security import get_auth_context


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/apps/{{app_id}}/api-clients",
    tags=["App API Clients"],
)


def _canonical_app_id(value: str) -> str:
    try:
        return resolve_business_app_id(value)
    except ValueError as exc:
        raise HTTPException(404, {"code": "APP_NOT_FOUND"}) from exc


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppApiClientCreatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    subject_user_id: str = Field(min_length=1, max_length=256)
    scopes: tuple[str, ...] = Field(min_length=1, max_length=32)
    agent_ids: tuple[UUID, ...] = Field(min_length=1, max_length=128)
    expires_at: datetime
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10000)


class AppApiClientStatusPayload(_Payload):
    status: Literal["DISABLED"]


class AppApiCredentialRotatePayload(_Payload):
    expires_at: datetime


class AppApiCredentialView(_Payload):
    credential_id: UUID
    public_key_id: str
    status: str
    expires_at: datetime
    last_used_at: datetime | None = None
    created_at: datetime
    revoked_at: datetime | None = None


class AppApiClientView(_Payload):
    client_id: UUID
    app_id: str
    domain_id: int
    subject_user_id: str
    display_name: str
    status: str
    rate_limit_per_minute: int
    row_version: int
    created_by: str
    created_at: datetime
    updated_at: datetime
    scopes: list[str]
    agent_ids: list[UUID]
    credentials: list[AppApiCredentialView]


class AppApiClientCreatedView(AppApiClientView):
    credential_id: UUID
    api_key: str
    warning: str


class AppApiCredentialCreatedView(_Payload):
    client_id: UUID
    credential_id: UUID
    api_key: str
    expires_at: datetime
    warning: str


class AppApiClientListView(_Payload):
    items: list[AppApiClientView]


class AppApiScopeView(_Payload):
    scope_code: str
    required_permission: str


class AppApiScopeListView(_Payload):
    app_id: str
    items: list[AppApiScopeView]


def _service(request: Request) -> AppApiKeyService:
    return cast(AppApiKeyService, request.app.state.app_api_key_service)


async def _manager_context(request: Request, app_id: str) -> tuple[int, str]:
    context = get_auth_context(request)
    if (
        context.principal_kind != PrincipalKind.PORTAL
        or context.entry_kind != IdentityEntryKind.BUSINESS
        or context.api_key_id != "user-jwt"
    ):
        raise HTTPException(
            403,
            {
                "code": "APP_API_KEY_MANAGEMENT_USER_SESSION_REQUIRED",
                "message": "API Client 只能由 App 管理员用户会话维护",
            },
        )
    if context.app_id != app_id:
        raise HTTPException(403, {"code": "APP_CONTEXT_MISMATCH"})
    domain_id, actor_id, _ = await authorize_app_request(
        request,
        app_id=app_id,
        permission=f"{app_id}:api_key_manage",
    )
    return domain_id, actor_id


def _translate(exc: AppApiKeyError) -> HTTPException:
    return HTTPException(
        exc.status_code, {"code": exc.code, "message": str(exc)}
    )


async def _active_agent_ids(
    request: Request, *, app_id: str, domain_id: int
) -> frozenset[UUID]:
    """从业务 App 实时读取当前 Domain 的 ACTIVE Agent。"""

    context = request.state.auth_context
    if app_id == "knowledge_retrieval":
        rows = await request.app.state.knowledge_retrieval_app_client.list_agents(
            domain_id=domain_id,
            auth_context=context,
        )
    elif app_id == "km_asset":
        rows = await request.app.state.km_asset_client.list_agents(
            domain_id=domain_id,
            auth_context=context,
        )
    elif app_id == "aiops":
        rows = await request.app.state.aiops_client.list_private_agents(
            auth_context=context,
        )
    else:
        raise AppApiKeyError(
            "APP_API_KEY_UNSUPPORTED_APP",
            "该 App 尚未开放 API Client",
        )
    return frozenset(
        UUID(str(item["agent_id"]))
        for item in rows
        if item.get("status") == "ACTIVE" and item.get("agent_id")
    )


@router.get("/scopes", response_model=AppApiScopeListView)
async def list_app_api_scopes(app_id: str, request: Request):
    app_id = _canonical_app_id(app_id)
    await _manager_context(request, app_id)
    return {"app_id": app_id, "items": _service(request).scope_catalog(app_id)}


@router.get("", response_model=AppApiClientListView)
async def list_app_api_clients(app_id: str, request: Request):
    app_id = _canonical_app_id(app_id)
    domain_id, _ = await _manager_context(request, app_id)
    return {"items": await _service(request).list_clients(
        app_id=app_id, domain_id=domain_id
    )}


@router.post(
    "", status_code=status.HTTP_201_CREATED,
    response_model=AppApiClientCreatedView,
)
async def create_app_api_client(
    app_id: str, payload: AppApiClientCreatePayload, request: Request
):
    app_id = _canonical_app_id(app_id)
    domain_id, actor_id = await _manager_context(request, app_id)
    try:
        active_agent_ids = await _active_agent_ids(
            request,
            app_id=app_id,
            domain_id=domain_id,
        )
        return await _service(request).create_client(
            app_id=app_id,
            domain_id=domain_id,
            subject_user_id=payload.subject_user_id.strip(),
            display_name=payload.display_name.strip(),
            scopes=payload.scopes,
            agent_ids=payload.agent_ids,
            active_agent_ids=active_agent_ids,
            expires_at=payload.expires_at,
            rate_limit_per_minute=payload.rate_limit_per_minute,
            actor_id=actor_id,
        )
    except AppApiKeyError as exc:
        raise _translate(exc) from exc


@router.get("/{client_id}", response_model=AppApiClientView)
async def get_app_api_client(
    app_id: str, client_id: UUID, request: Request
):
    app_id = _canonical_app_id(app_id)
    domain_id, _ = await _manager_context(request, app_id)
    try:
        return await _service(request).get_client(
            app_id=app_id, domain_id=domain_id, client_id=client_id
        )
    except AppApiKeyError as exc:
        raise _translate(exc) from exc


@router.patch("/{client_id}", response_model=AppApiClientView)
async def update_app_api_client_status(
    app_id: str,
    client_id: UUID,
    payload: AppApiClientStatusPayload,
    request: Request,
):
    app_id = _canonical_app_id(app_id)
    domain_id, _ = await _manager_context(request, app_id)
    try:
        return await _service(request).set_status(
            app_id=app_id, domain_id=domain_id,
            client_id=client_id, status=payload.status
        )
    except AppApiKeyError as exc:
        raise _translate(exc) from exc


@router.post(
    "/{client_id}/rotate", response_model=AppApiCredentialCreatedView
)
async def rotate_app_api_credential(
    app_id: str,
    client_id: UUID,
    payload: AppApiCredentialRotatePayload,
    request: Request,
):
    app_id = _canonical_app_id(app_id)
    domain_id, actor_id = await _manager_context(request, app_id)
    try:
        return await _service(request).rotate(
            app_id=app_id,
            domain_id=domain_id,
            client_id=client_id,
            expires_at=payload.expires_at,
            actor_id=actor_id,
        )
    except AppApiKeyError as exc:
        raise _translate(exc) from exc


@router.post("/{client_id}/revoke", response_model=AppApiClientView)
async def revoke_app_api_client(
    app_id: str, client_id: UUID, request: Request
):
    """立即停用 API Client，并撤销其全部有效 Credential。"""
    app_id = _canonical_app_id(app_id)
    domain_id, _ = await _manager_context(request, app_id)
    try:
        return await _service(request).set_status(
            app_id=app_id, domain_id=domain_id,
            client_id=client_id, status="DISABLED"
        )
    except AppApiKeyError as exc:
        raise _translate(exc) from exc


__all__ = ["router"]
