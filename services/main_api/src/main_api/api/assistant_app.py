"""智能工作台 App 的公开 BFF 路由。"""

from datetime import date
from typing import Any, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from main_api.api.models import ModelCatalogItem, load_model_catalog
from main_api.application import (
    AccessControlService,
    AccessDeniedError,
    AppApiKeyError,
    UserAuthService,
    require_app_api_permission,
)
from platform_clients import AssistantAppClient, DataQueryClient, KnowledgeCoreClient
from platform_core.contracts import PUBLIC_API_V1, PrincipalKind
from platform_core.security import get_auth_context


router = APIRouter(prefix=f"{PUBLIC_API_V1}/apps/assistant", tags=["Assistant App"])

ASSISTANT_PORTAL_DOMAIN_NAME = "assistant_portal"
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "REJECTED"})


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssistantLoginPayload(_Payload):
    user_id: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=256)


class AssistantPasswordChangePayload(_Payload):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=12, max_length=256)

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, value: str) -> str:
        if not (
            any(char.islower() for char in value)
            and any(char.isupper() for char in value)
            and any(char.isdigit() for char in value)
            and any(not char.isalnum() for char in value)
        ):
            raise ValueError("新密码必须同时包含大小写字母、数字和特殊字符")
        return value


class AssistantAgentCreatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    knowledge_core_id: UUID | None = None
    data_model_ids: tuple[UUID, ...] = Field(default=(), max_length=100)
    models: dict[str, UUID] = Field(default_factory=dict)
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] = Field(default_factory=dict)
    status: Literal["DRAFT", "ACTIVE"] = "DRAFT"


class AssistantAgentUpdatePayload(_Payload):
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    knowledge_core_id: UUID | None = None
    data_model_ids: tuple[UUID, ...] = Field(default=None, max_length=100)
    models: dict[str, UUID] | None = None
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] | None = None
    status: Literal["DRAFT", "ACTIVE", "DISABLED", "ARCHIVED"] | None = None


class AssistantBindingUpsertPayload(_Payload):
    role: Literal["KNOWLEDGE", "X_SEARCH", "IMAGE_GENERATION"]
    model_id: UUID
    expected_row_version: int | None = Field(default=None, ge=1)


class AssistantResearchCreatePayload(_Payload):
    input: str = Field(min_length=1, max_length=32000)
    from_date: date | None = None
    to_date: date | None = None
    allowed_x_handles: tuple[str, ...] = Field(default=(), max_length=20)
    excluded_x_handles: tuple[str, ...] = Field(default=(), max_length=20)
    enable_image_understanding: bool = False
    enable_video_understanding: bool = False
    search_context_size: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"

    @model_validator(mode="after")
    def validate_filters(self) -> "AssistantResearchCreatePayload":
        if self.allowed_x_handles and self.excluded_x_handles:
            raise ValueError("allowed_x_handles 与 excluded_x_handles 不能同时设置")
        if self.from_date and self.to_date and self.from_date > self.to_date:
            raise ValueError("from_date 不能晚于 to_date")
        return self


class AssistantImageCreatePayload(_Payload):
    prompt: str = Field(min_length=1, max_length=32000)
    aspect_ratio: str = Field(default="1:1", pattern=r"^[1-9][0-9]?:[1-9][0-9]?$")
    count: int = Field(default=1, ge=1, le=4)


def _domain_actor(request: Request) -> tuple[int, str]:
    context = get_auth_context(request)
    if context.app_id and context.app_id != "assistant":
        raise HTTPException(403, {"code": "APP_CONTEXT_MISMATCH"})
    try:
        domain_id = int(context.domain_id or "")
    except ValueError as exc:
        raise HTTPException(403, {"code": "DOMAIN_CONTEXT_REQUIRED"}) from exc
    if domain_id < 1:
        raise HTTPException(403, {"code": "DOMAIN_CONTEXT_REQUIRED"})
    return domain_id, context.asserted_user_id or context.client_id


def _access(request: Request) -> AccessControlService:
    return cast(AccessControlService, request.app.state.access_control_service)


def _client(request: Request) -> AssistantAppClient:
    return cast(AssistantAppClient, request.app.state.assistant_app_client)


def _knowledge(request: Request) -> KnowledgeCoreClient:
    return cast(KnowledgeCoreClient, request.app.state.knowledge_core_client)


def _data_query(request: Request) -> DataQueryClient:
    return cast(DataQueryClient, request.app.state.data_query_client)


def _capabilities_from_bindings(bindings: list[dict[str, Any]]) -> dict[str, Any]:
    by_role = {item.get("role"): item for item in bindings}

    def summarize(role: str, flag: str | None) -> dict[str, bool]:
        item = by_role.get(role)
        bound = item is not None
        if item is not None and flag:
            verified = bool(item.get(flag))
        else:
            verified = bound and bool(item and item.get("ready"))
        ready = bool(item and item.get("ready"))
        return {"bound": bound, "verified": verified, "ready": ready}

    return {
        "knowledge": summarize("KNOWLEDGE", None),
        "x_search": summarize("X_SEARCH", "supports_x_search"),
        "image_generation": summarize("IMAGE_GENERATION", "supports_image_generation"),
    }


def _run_response(view: dict[str, Any]) -> JSONResponse:
    status_code = 200 if view.get("status") in TERMINAL_STATUSES else 202
    return JSONResponse(status_code=status_code, content=view)


def _require_any_app_permission(request: Request, *permissions: str) -> None:
    context = get_auth_context(request)
    if context.principal_kind != PrincipalKind.APP_API_CLIENT:
        return
    last_error: AppApiKeyError | None = None
    for permission in permissions:
        try:
            require_app_api_permission(request, permission)
            return
        except AppApiKeyError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error


async def _require(request: Request, permission: str):
    require_app_api_permission(request, permission)
    domain_id, actor_id = _domain_actor(request)
    try:
        snapshot = await _access(request).require(
            app_id="assistant", domain_id=domain_id, user_id=actor_id,
            permission_code=permission,
        )
    except AccessDeniedError as exc:
        raise HTTPException(403, {"code": "APP_PERMISSION_DENIED", "permission": permission}) from exc
    return domain_id, actor_id, snapshot


async def _snapshot(request: Request):
    domain_id, actor_id = _domain_actor(request)
    snapshot = await _access(request).snapshot(app_id="assistant", domain_id=domain_id, user_id=actor_id)
    return domain_id, actor_id, snapshot


async def _require_media_asset(request: Request, asset_id: UUID) -> tuple[int, str, dict[str, Any]]:
    _require_any_app_permission(request, "assistant:media_read", "assistant:image_generate")
    domain_id, actor_id, snapshot = await _snapshot(request)
    permissions = set(snapshot.permissions)
    can_read = "assistant:media_read" in permissions
    can_own = "assistant:image_generate" in permissions
    if not can_read and not can_own:
        raise HTTPException(403, {"code": "APP_PERMISSION_DENIED", "permission": "assistant:media_read"})
    asset = await _client(request).get_media_asset(
        asset_id=asset_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )
    if can_read or (can_own and asset.get("created_by") == actor_id):
        return domain_id, actor_id, asset
    raise HTTPException(403, {"code": "APP_PERMISSION_DENIED", "permission": "assistant:media_read"})


async def _require_active_knowledge_core(request: Request, *, domain_id: int, knowledge_core_id: UUID | None) -> None:
    if knowledge_core_id is None:
        raise HTTPException(422, {"code": "AGENT_KNOWLEDGE_CORE_REQUIRED", "message": "启用 Agent 前必须绑定一个可用 Knowledge Core"})
    core = await _knowledge(request).get_collection(
        domain_id=domain_id, collection_id=knowledge_core_id,
        auth_context=request.state.auth_context,
    )
    if core.get("status") != "ACTIVE":
        raise HTTPException(422, {"code": "AGENT_KNOWLEDGE_CORE_INACTIVE", "message": "绑定的 Knowledge Core 未处于 ACTIVE 状态"})


async def _require_active_data_binding(request: Request, *, agent: dict[str, Any]) -> None:
    raw_model_ids = agent.get("data_model_ids") or []
    if not raw_model_ids:
        return
    matched = await _data_query(request).management_has_active_agent_binding(
        consumer_app_id="assistant",
        agent_id=UUID(str(agent["agent_id"])),
        agent_version_id=UUID(str(agent["agent_version_id"])),
        semantic_model_ids={UUID(str(value)) for value in raw_model_ids},
        auth_context=request.state.auth_context,
    )
    if not matched:
        raise HTTPException(422, {"code": "APP_AGENT_QUERY_BINDING_REQUIRED", "message": "启用问数 Agent 前必须为当前版本配置有效查询绑定"})


@router.post("/auth/login")
async def login(payload: AssistantLoginPayload, request: Request):
    """使用平台用户凭据进入固定的智能工作台 Portal Domain。"""
    service = cast(UserAuthService, request.app.state.user_auth_service)
    return await service.login_for_domain_name(
        user_id=payload.user_id.strip(), password=payload.password,
        domain_name=ASSISTANT_PORTAL_DOMAIN_NAME, app_id="assistant",
    )


@router.post("/auth/password")
async def change_password(payload: AssistantPasswordChangePayload, request: Request):
    """修改智能工作台本地用户密码。"""
    service = cast(UserAuthService, request.app.state.user_auth_service)
    return await service.change_password(
        claims=request.state.user_token_claims,
        current_password=payload.current_password,
        new_password=payload.new_password,
    )


@router.get("/access")
async def get_access(request: Request):
    domain_id, actor_id, snapshot = await _snapshot(request)
    bindings = await _client(request).list_bindings(
        domain_id=domain_id, auth_context=request.state.auth_context,
    )
    return {
        "app_id": snapshot.app_id,
        "domain_id": snapshot.domain_id,
        "user_id": snapshot.user_id,
        "roles": snapshot.roles,
        "permissions": sorted(snapshot.permissions),
        "bindings": bindings,
        "capabilities": _capabilities_from_bindings(bindings if isinstance(bindings, list) else []),
    }


@router.get("/model-catalog", response_model=list[ModelCatalogItem])
async def list_assistant_model_catalog(request: Request):
    await _require(request, "assistant:model_binding_manage")
    return await load_model_catalog(request)


@router.get("/bindings")
async def list_bindings(request: Request):
    domain_id, _, _ = await _require(request, "assistant:model_binding_manage")
    return await _client(request).list_bindings(
        domain_id=domain_id, auth_context=request.state.auth_context,
    )


@router.put("/bindings")
async def upsert_binding(payload: AssistantBindingUpsertPayload, request: Request):
    domain_id, _, _ = await _require(request, "assistant:model_binding_manage")
    return await _client(request).upsert_binding(
        payload={"domain_id": domain_id, **payload.model_dump(mode="json")},
        auth_context=request.state.auth_context,
    )


@router.get("/x-search/runs")
async def list_research_runs(
    request: Request,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    domain_id, _, _ = await _require(request, "assistant:x_search")
    return await _client(request).list_research_runs(
        domain_id=domain_id, auth_context=request.state.auth_context, status=status, limit=limit,
    )


@router.post("/x-search/runs")
async def create_research_run(
    payload: AssistantResearchCreatePayload,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    domain_id, _, _ = await _require(request, "assistant:x_search")
    view = await _client(request).create_research_run(
        payload={"domain_id": domain_id, **payload.model_dump(mode="json")},
        auth_context=request.state.auth_context,
        idempotency_key=idempotency_key,
    )
    return _run_response(view)


@router.get("/x-search/runs/{run_id}")
async def get_research_run(run_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "assistant:x_search")
    return await _client(request).get_research_run(
        run_id=run_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )


@router.get("/x-search/runs/{run_id}/events")
async def list_research_events(run_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "assistant:x_search")
    return await _client(request).list_research_events(
        run_id=run_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )


@router.get("/x-search/runs/{run_id}/sources")
async def list_research_sources(run_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "assistant:x_search")
    return await _client(request).list_research_sources(
        run_id=run_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )


@router.get("/image-generations/runs")
async def list_image_runs(
    request: Request,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    domain_id, _, _ = await _require(request, "assistant:image_generate")
    return await _client(request).list_image_runs(
        domain_id=domain_id, auth_context=request.state.auth_context, status=status, limit=limit,
    )


@router.post("/image-generations/runs")
async def create_image_run(
    payload: AssistantImageCreatePayload,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    domain_id, _, _ = await _require(request, "assistant:image_generate")
    view = await _client(request).create_image_run(
        payload={"domain_id": domain_id, **payload.model_dump(mode="json")},
        auth_context=request.state.auth_context,
        idempotency_key=idempotency_key,
    )
    return _run_response(view)


@router.get("/image-generations/runs/{run_id}")
async def get_image_run(run_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "assistant:image_generate")
    return await _client(request).get_image_run(
        run_id=run_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )


@router.get("/image-generations/runs/{run_id}/events")
async def list_image_events(run_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "assistant:image_generate")
    return await _client(request).list_image_events(
        run_id=run_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )


@router.get("/media-assets")
async def list_media_assets(
    request: Request,
    run_id: UUID | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    domain_id, _, _ = await _require(request, "assistant:media_read")
    return await _client(request).list_media_assets(
        domain_id=domain_id, auth_context=request.state.auth_context,
        run_id=run_id, status=status, limit=limit,
    )


@router.get("/media-assets/{asset_id}")
async def get_media_asset(asset_id: UUID, request: Request):
    _, _, asset = await _require_media_asset(request, asset_id)
    return asset


@router.get("/media-assets/{asset_id}/content")
async def get_media_content(asset_id: UUID, request: Request):
    domain_id, _, _ = await _require_media_asset(request, asset_id)
    data, mime_type = await _client(request).get_media_content(
        asset_id=asset_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )
    return Response(
        content=data,
        media_type=mime_type,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/media-assets/{asset_id}:download-url")
async def create_media_download_url(asset_id: UUID, request: Request):
    domain_id, _, _ = await _require_media_asset(request, asset_id)
    return await _client(request).create_media_download_url(
        asset_id=asset_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )


@router.get("/runs")
async def list_runs(
    request: Request,
    kind: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    domain_id, _, _ = await _require(request, "assistant:run_read")
    return await _client(request).list_runs(
        domain_id=domain_id, auth_context=request.state.auth_context,
        kind=kind, status=status, limit=limit,
    )


@router.get("/knowledge-cores")
async def list_knowledge_cores(request: Request):
    domain_id, _, _ = await _require(request, "assistant:knowledge_core_manage")
    return await _knowledge(request).list_collections(domain_id=domain_id, auth_context=request.state.auth_context)


@router.get("/agents")
async def list_agents(request: Request):
    domain_id, _, _ = await _require(request, "assistant:knowledge_chat")
    return await _client(request).list_agents(domain_id=domain_id, auth_context=request.state.auth_context)


@router.post("/agents", status_code=status.HTTP_201_CREATED)
async def create_agent(payload: AssistantAgentCreatePayload, request: Request):
    domain_id, _, _ = await _require(request, "assistant:agent_manage")
    if payload.status == "ACTIVE":
        await _require_active_knowledge_core(request, domain_id=domain_id, knowledge_core_id=payload.knowledge_core_id)
        if payload.data_model_ids:
            raise HTTPException(422, {"code": "APP_AGENT_QUERY_BINDING_VERSION_REQUIRED", "message": "带问数能力的 Agent 必须先以草稿创建并配置有效查询绑定，再单独启用"})
    return await _client(request).create_agent(payload={"domain_id": domain_id, **payload.model_dump(mode="json")}, auth_context=request.state.auth_context)


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "assistant:knowledge_chat")
    return await _client(request).get_agent(agent_id=agent_id, domain_id=domain_id, auth_context=request.state.auth_context)


@router.patch("/agents/{agent_id}")
async def update_agent(
    agent_id: UUID,
    payload: AssistantAgentUpdatePayload,
    request: Request,
):
    domain_id, _, _ = await _require(request, "assistant:agent_manage")
    current = await _client(request).get_agent(agent_id=agent_id, domain_id=domain_id, auth_context=request.state.auth_context)
    values = payload.model_dump(mode="json", exclude_unset=True)
    status_value = values.get("status", current["status"])
    version_fields = {"knowledge_core_id", "data_model_ids", "models", "instruction", "config"}
    if status_value == "ACTIVE":
        await _require_active_knowledge_core(
            request, domain_id=domain_id,
            knowledge_core_id=(
                UUID(str(values["knowledge_core_id"]))
                if "knowledge_core_id" in values and values["knowledge_core_id"]
                else None
                if "knowledge_core_id" in values
                else UUID(str(current["knowledge_core_id"]))
                if current.get("knowledge_core_id")
                else None
            ),
        )
        data_model_ids = values.get("data_model_ids", current.get("data_model_ids") or [])
        if data_model_ids:
            if version_fields.intersection(values):
                raise HTTPException(422, {"code": "APP_AGENT_QUERY_BINDING_VERSION_REQUIRED", "message": "请先保存 Agent 草稿版本、创建该版本的查询绑定，再单独启用"})
            await _require_active_data_binding(request, agent=current)
    return await _client(request).update_agent(agent_id=agent_id, payload={"domain_id": domain_id, **values}, auth_context=request.state.auth_context)
