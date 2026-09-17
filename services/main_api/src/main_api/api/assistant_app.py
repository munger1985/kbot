"""智能工作台 App 的公开 BFF 路由。"""

import re
from datetime import date
from typing import Any, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, Response
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from main_api.api.models import ModelCatalogItem, load_model_catalog
from main_api.application import (
    AccessControlService,
    AccessDeniedError,
    AppApiKeyError,
    DomainConflictError,
    DomainLifecycleError,
    DomainManagementService,
    UserAuthService,
    require_app_api_permission,
)
from platform_clients import AssistantAppClient, AssistantAppClientError, DataQueryClient, KnowledgeCoreClient
from platform_core.contracts import PUBLIC_API_V1, PrincipalKind
from platform_core.contracts.data_query import SemanticModelDefinition
from platform_core.dictionary import (
    ModelCategory,
    coerce_model_category,
    is_enabled_model_status,
)
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


class AssistantDomainCreatePayload(_Payload):
    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)


class AssistantDomainUpdatePayload(_Payload):
    expected_row_version: int = Field(ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    status: Literal["ACTIVE", "DISABLED"] | None = None


class AssistantKnowledgeCoreCreatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    default_security_level: int = Field(default=1, ge=0, le=999)
    embedding: UUID
    visual_embedding: UUID | None = None


class AssistantKnowledgeCoreUpdatePayload(_Payload):
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    default_security_level: int | None = Field(default=None, ge=0, le=999)
    status: Literal["ACTIVE", "DISABLED"] | None = None
    embedding: UUID | None = None
    visual_embedding: UUID | None = None


class AssistantIntakeReviewPayload(_Payload):
    decision: Literal["APPROVE", "REJECT"]
    comment: str | None = Field(default=None, max_length=1000)


class AssistantKnowledgeCoreReprocessPayload(_Payload):
    document_version_id: UUID


class AssistantDataSourceEndpointPayload(_Payload):
    host: str = Field(
        min_length=1,
        max_length=253,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9.-]*$",
    )
    port: int = Field(ge=1, le=65535)
    database: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._$#-]{0,127}$",
    )
    allowed_schemas: tuple[str, ...] = Field(min_length=1, max_length=32)
    tls_enabled: bool = True

    @field_validator("allowed_schemas")
    @classmethod
    def validate_allowed_schemas(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        pattern = r"^[A-Za-z_][A-Za-z0-9_$#-]{0,127}$"
        normalized = tuple(item.strip() for item in value)
        if any(not re.fullmatch(pattern, item) for item in normalized):
            raise ValueError("allowed_schemas 包含非法数据库标识符")
        if len(set(normalized)) != len(normalized):
            raise ValueError("allowed_schemas 不能重复")
        return normalized


class AssistantDataSourceCredentialsPayload(_Payload):
    username: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=1024)


class AssistantDataSourceConnectionPayload(_Payload):
    source_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    endpoint: AssistantDataSourceEndpointPayload
    credentials: AssistantDataSourceCredentialsPayload


class AssistantDataSourceCreatePayload(AssistantDataSourceConnectionPayload):
    display_name: str = Field(min_length=1, max_length=256)
    auto_discover_schema: bool = True


class AssistantSchemaSelectionPayload(_Payload):
    object_ids: tuple[UUID, ...] = Field(min_length=1, max_length=5000)


class AssistantManualSchemaPayload(_Payload):
    ddl: str = Field(min_length=10, max_length=100_000)


class AssistantSemanticModelCandidatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    business_context: str | None = Field(default=None, max_length=4000)
    object_ids: tuple[UUID, ...] = Field(default=(), max_length=64)
    ai_model_id: UUID | None = None
    allow_ai_metadata: bool = False


class AssistantSemanticModelDraftUpdatePayload(_Payload):
    definition: SemanticModelDefinition
    expected_row_version: int = Field(ge=1)


class AssistantSemanticModelValidationPayload(_Payload):
    question: str = Field(min_length=2, max_length=2000)
    ai_model_id: UUID
    allow_ai_metadata: bool = False


class AssistantSemanticModelReviewPayload(_Payload):
    expected_row_version: int = Field(ge=1)


class AssistantSemanticModelPublishPayload(AssistantSemanticModelReviewPayload):
    schema_snapshot_id: UUID


class AssistantQueryBudgetPayload(_Payload):
    max_rows: int = Field(default=1000, ge=1, le=10000)
    max_result_bytes: int = Field(
        default=1_048_576, ge=1024, le=16_777_216
    )
    statement_timeout_seconds: int = Field(default=30, ge=1, le=300)
    max_concurrent_runs: int = Field(default=4, ge=1, le=64)


class AssistantPolicyBindingCreatePayload(_Payload):
    actor_ids: tuple[str, ...] = Field(default=(), max_length=1000)
    roles: tuple[str, ...] = Field(default=(), max_length=100)
    semantic_model_ids: tuple[UUID, ...] = Field(min_length=1, max_length=64)
    budget: AssistantQueryBudgetPayload = Field(
        default_factory=AssistantQueryBudgetPayload
    )


class AssistantAgentQueryBindingCreatePayload(_Payload):
    agent_id: UUID
    semantic_model_id: UUID
    policy_binding_id: UUID


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


def _domains(request: Request) -> DomainManagementService:
    service = getattr(request.app.state, "domain_management_service", None)
    if service is None:
        raise RuntimeError("Domain Management Service 尚未初始化")
    return cast(DomainManagementService, service)


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


async def _require_any(request: Request, *permissions: str):
    _require_any_app_permission(request, *permissions)
    domain_id, actor_id, snapshot = await _snapshot(request)
    matched = [item for item in permissions if item in set(snapshot.permissions)]
    if not matched:
        raise HTTPException(403, {"code": "APP_PERMISSION_DENIED", "permission": permissions[0]})
    return domain_id, actor_id, snapshot


def _auth_for_domain(request: Request, domain_id: int):
    context = request.state.auth_context
    if str(context.domain_id) == str(domain_id):
        return context
    return context.model_copy(update={"domain_id": str(domain_id)})


def _raise_domain_error(exc: DomainConflictError | DomainLifecycleError) -> None:
    if isinstance(exc, DomainConflictError):
        raise HTTPException(409, {"code": "DOMAIN_NAME_CONFLICT", "message": str(exc)}) from exc
    raise HTTPException(exc.status_code, {"code": exc.code, "message": str(exc)}) from exc


async def _assistant_collection_models(
    request: Request,
    *,
    embedding: UUID,
    visual_embedding: UUID | None,
) -> dict[str, str]:
    catalog = await load_model_catalog(request)
    by_id = {str(item.get("model_id")): item for item in catalog}
    requested = {"embedding": (embedding, ModelCategory.TXT_EMBEDDING)}
    if visual_embedding is not None:
        requested["visual_embedding"] = (visual_embedding, ModelCategory.IMG_EMBEDDING)
    models: dict[str, str] = {}
    for role, (model_id, expected_category) in requested.items():
        row = by_id.get(str(model_id))
        if row is None or not is_enabled_model_status(row.get("status")):
            raise HTTPException(
                422,
                {
                    "code": "KNOWLEDGE_CORE_MODEL_UNAVAILABLE",
                    "message": f"模型角色 {role} 绑定的模型未启用或不存在",
                },
            )
        if coerce_model_category(row.get("category")) != expected_category:
            raise HTTPException(
                422,
                {
                    "code": "KNOWLEDGE_CORE_MODEL_CATEGORY_INVALID",
                    "message": f"模型角色 {role} 的模型类别不正确",
                },
            )
        models[role] = str(model_id)
    return models


def _collection_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("collections", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


async def _domain_resource_conflicts(request: Request, *, domain_id: int) -> list[str]:
    auth_context = _auth_for_domain(request, domain_id)
    conflicts: list[str] = []
    collections = _collection_items(
        await _knowledge(request).list_collections(
            domain_id=domain_id, auth_context=auth_context
        )
    )
    live_collections = [
        item for item in collections if str(item.get("status") or "") not in {"DELETING"}
    ]
    if live_collections:
        conflicts.append(f"{len(live_collections)} 个 Knowledge Core")
    agents = await _client(request).list_agents(
        domain_id=domain_id, auth_context=auth_context
    )
    live_agents = [
        item
        for item in agents
        if isinstance(item, dict) and str(item.get("status") or "") not in {"ARCHIVED"}
    ]
    if live_agents:
        conflicts.append(f"{len(live_agents)} 个 Agent")
    runs = await _client(request).list_runs(
        domain_id=domain_id, auth_context=auth_context, limit=50
    )
    live_runs = [
        item
        for item in runs
        if isinstance(item, dict) and str(item.get("status") or "") not in TERMINAL_STATUSES
    ]
    if live_runs:
        conflicts.append(f"{len(live_runs)} 个进行中的 Run")
    assets = await _client(request).list_media_assets(
        domain_id=domain_id, auth_context=auth_context, limit=50
    )
    if assets:
        conflicts.append(f"{len(assets)} 个媒体资产")
    return conflicts


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
    bindings: list[dict[str, Any]] = []
    try:
        listed = await _client(request).list_bindings(
            domain_id=domain_id, auth_context=request.state.auth_context,
        )
        if isinstance(listed, list):
            bindings = listed
    except AssistantAppClientError as exc:
        logger.warning(
            "智能工作台 access 读取绑定失败，继续返回权限快照 code={} status={}",
            exc.code,
            exc.status_code,
        )
    return {
        "app_id": snapshot.app_id,
        "domain_id": snapshot.domain_id,
        "user_id": snapshot.user_id,
        "roles": snapshot.roles,
        "permissions": sorted(snapshot.permissions),
        "bindings": bindings,
        "capabilities": _capabilities_from_bindings(bindings),
    }


@router.get("/model-catalog", response_model=list[ModelCatalogItem])
async def list_assistant_model_catalog(request: Request):
    await _require_any(
        request,
        "assistant:model_binding_manage",
        "assistant:knowledge_core_manage",
        "assistant:data_model_manage",
    )
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


@router.delete("/x-search/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_research_run(run_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "assistant:x_search")
    await _client(request).delete_research_run(
        run_id=run_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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


@router.delete("/image-generations/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_image_run(run_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "assistant:image_generate")
    await _client(request).delete_image_run(
        run_id=run_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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


@router.get("/domains")
async def list_domains(request: Request):
    _, actor_id, _ = await _require(request, "assistant:domain_manage")
    return await _domains(request).list_for_app(app_id="assistant", user_id=actor_id)


@router.post("/domains", status_code=status.HTTP_201_CREATED)
async def create_domain(payload: AssistantDomainCreatePayload, request: Request):
    _, actor_id, _ = await _require(request, "assistant:domain_manage")
    try:
        return await _domains(request).create_for_app(
            app_id="assistant",
            name=payload.name,
            description=payload.description,
            actor_id=actor_id,
        )
    except (DomainConflictError, DomainLifecycleError) as exc:
        _raise_domain_error(exc)


@router.get("/domains/{target_domain_id}")
async def get_domain(target_domain_id: int, request: Request):
    _, actor_id, _ = await _require(request, "assistant:domain_manage")
    try:
        return await _domains(request).get_for_app(
            app_id="assistant", domain_id=target_domain_id, user_id=actor_id
        )
    except DomainLifecycleError as exc:
        _raise_domain_error(exc)


@router.patch("/domains/{target_domain_id}")
async def update_domain(
    target_domain_id: int,
    payload: AssistantDomainUpdatePayload,
    request: Request,
):
    _, actor_id, _ = await _require(request, "assistant:domain_manage")
    changes = payload.model_dump(exclude_unset=True)
    if changes.get("status") == "DISABLED":
        conflicts = await _domain_resource_conflicts(request, domain_id=target_domain_id)
        if conflicts:
            raise HTTPException(
                409,
                {
                    "code": "DOMAIN_IN_USE",
                    "message": "停用前需要先处理：" + "、".join(conflicts),
                },
            )
    try:
        return await _domains(request).update_for_app(
            app_id="assistant",
            domain_id=target_domain_id,
            user_id=actor_id,
            actor_id=actor_id,
            expected_row_version=payload.expected_row_version,
            name=payload.name,
            description=payload.description,
            description_set="description" in changes,
            status=payload.status,
        )
    except (DomainConflictError, DomainLifecycleError) as exc:
        _raise_domain_error(exc)


@router.delete("/domains/{target_domain_id}")
async def delete_domain(
    target_domain_id: int,
    request: Request,
    expected_row_version: int = Query(ge=1),
):
    _, actor_id, _ = await _require(request, "assistant:domain_manage")
    conflicts = await _domain_resource_conflicts(request, domain_id=target_domain_id)
    if conflicts:
        raise HTTPException(
            409,
            {
                "code": "DOMAIN_IN_USE",
                "message": "删除前需要先处理：" + "、".join(conflicts),
            },
        )
    try:
        return await _domains(request).disable_for_app(
            app_id="assistant",
            domain_id=target_domain_id,
            user_id=actor_id,
            actor_id=actor_id,
            expected_row_version=expected_row_version,
        )
    except (DomainConflictError, DomainLifecycleError) as exc:
        _raise_domain_error(exc)


@router.get("/knowledge-cores")
async def list_knowledge_cores(request: Request):
    domain_id, _, _ = await _require(request, "assistant:knowledge_core_manage")
    return await _knowledge(request).list_collections(
        domain_id=domain_id, auth_context=request.state.auth_context
    )


@router.post("/knowledge-cores", status_code=status.HTTP_201_CREATED)
async def create_knowledge_core(
    payload: AssistantKnowledgeCoreCreatePayload,
    request: Request,
):
    domain_id, _, _ = await _require(request, "assistant:knowledge_core_manage")
    models = await _assistant_collection_models(
        request,
        embedding=payload.embedding,
        visual_embedding=payload.visual_embedding,
    )
    return await _knowledge(request).create_collection(
        domain_id=domain_id,
        payload={
            "display_name": payload.display_name,
            "description": payload.description,
            "default_security_level": payload.default_security_level,
            "models": models,
            "metadata": {"owner_app_id": "assistant"},
        },
        auth_context=request.state.auth_context,
    )


@router.get("/knowledge-cores/{collection_id}")
async def get_knowledge_core(collection_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "assistant:knowledge_core_manage")
    return await _knowledge(request).get_collection(
        domain_id=domain_id,
        collection_id=collection_id,
        auth_context=request.state.auth_context,
    )


@router.get("/knowledge-cores/{collection_id}/processing")
async def list_knowledge_core_processing(collection_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "assistant:knowledge_core_manage")
    return await _knowledge(request).list_processing(
        domain_id=domain_id,
        collection_id=collection_id,
        page=1,
        page_size=100,
        auth_context=request.state.auth_context,
    )


@router.get("/knowledge-cores/{collection_id}/approvals")
async def list_knowledge_core_approvals(collection_id: UUID, request: Request):
    """列出当前 Knowledge Core 中等待审核的用户资料。"""
    domain_id, _, _ = await _require(request, "assistant:knowledge_core_manage")
    return await _knowledge(request).list_pending_approvals(
        domain_id=domain_id,
        collection_id=collection_id,
        auth_context=request.state.auth_context,
    )


@router.post(
    "/knowledge-cores/{collection_id}/bundles/{bundle_id}/revisions/"
    "{bundle_revision_id}/reprocess",
    status_code=status.HTTP_202_ACCEPTED,
)
async def reprocess_knowledge_core_file(
    collection_id: UUID,
    bundle_id: UUID,
    bundle_revision_id: UUID,
    payload: AssistantKnowledgeCoreReprocessPayload,
    request: Request,
):
    """重新调度一个失败文件的 KC 解析与后续索引流水线。"""
    domain_id, _, _ = await _require(request, "assistant:knowledge_core_manage")
    return await _knowledge(request).reprocess_revision(
        domain_id=domain_id,
        collection_id=collection_id,
        bundle_id=bundle_id,
        bundle_revision_id=bundle_revision_id,
        document_version_id=payload.document_version_id,
        auth_context=request.state.auth_context,
    )


@router.post("/knowledge-cores/{collection_id}/bundle-revisions/{bundle_revision_id}/approval")
async def review_knowledge_core_intake(
    collection_id: UUID,
    bundle_revision_id: UUID,
    payload: AssistantIntakeReviewPayload,
    request: Request,
):
    """审核用户资料，批准后由 KC 启动解析和索引流水线。"""
    domain_id, _, _ = await _require(request, "assistant:knowledge_core_manage")
    return await _knowledge(request).review_user_intake(
        domain_id=domain_id,
        collection_id=collection_id,
        bundle_revision_id=bundle_revision_id,
        decision=payload.decision,
        comment=payload.comment,
        auth_context=request.state.auth_context,
    )


@router.post("/knowledge-cores/{collection_id}/ingestions/user-files", status_code=status.HTTP_202_ACCEPTED)
async def upload_knowledge_core_files(collection_id: UUID, request: Request):
    """将 Assistant App 的用户文件流式转发至 KC，不在 Main API 落盘。"""
    domain_id, _, _ = await _require(request, "assistant:knowledge_core_manage")
    content_type = request.headers.get("Content-Type", "")
    idempotency_key = request.headers.get("Idempotency-Key", "").strip()
    if not content_type.lower().startswith("multipart/form-data") or not idempotency_key:
        raise HTTPException(
            status.HTTP_428_PRECONDITION_REQUIRED,
            {
                "code": "ASSISTANT_KC_UPLOAD_HEADERS_REQUIRED",
                "message": "缺少 multipart Content-Type 或 Idempotency-Key",
            },
        )
    upstream = await _knowledge(request).ingest_multipart(
        domain_id=domain_id,
        collection_id=collection_id,
        intake_kind="user-files",
        content_type=content_type,
        body=request.stream(),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return JSONResponse(status_code=upstream.status_code, content=upstream.payload)


@router.patch("/knowledge-cores/{collection_id}")
async def update_knowledge_core(
    collection_id: UUID,
    payload: AssistantKnowledgeCoreUpdatePayload,
    request: Request,
):
    domain_id, _, _ = await _require(request, "assistant:knowledge_core_manage")
    changes = payload.model_dump(exclude_unset=True)
    knowledge = _knowledge(request)
    auth_context = request.state.auth_context
    current = await knowledge.get_collection(
        domain_id=domain_id, collection_id=collection_id, auth_context=auth_context
    )
    if int(current.get("row_version") or 0) != payload.expected_row_version:
        raise HTTPException(
            409,
            {"code": "COLLECTION_VERSION_CONFLICT", "message": "Knowledge Core 已被其他请求修改"},
        )
    row_version = payload.expected_row_version
    profile_fields = {"display_name", "description", "default_security_level"}
    if profile_fields.intersection(changes):
        profile = {"expected_row_version": row_version}
        for key in profile_fields:
            if key in changes:
                profile[key] = changes[key]
        current = await knowledge.update_collection_profile(
            domain_id=domain_id,
            collection_id=collection_id,
            payload=profile,
            auth_context=auth_context,
        )
        row_version = int(current.get("row_version") or row_version)
    if "embedding" in changes or "visual_embedding" in changes:
        current_models = dict(current.get("models") or current.get("models_json") or {})
        embedding = payload.embedding or UUID(str(current_models["embedding"]))
        visual_embedding = (
            payload.visual_embedding
            if "visual_embedding" in changes
            else (UUID(str(current_models["visual_embedding"])) if current_models.get("visual_embedding") else None)
        )
        models = await _assistant_collection_models(
            request, embedding=embedding, visual_embedding=visual_embedding
        )
        for role, model_id in current_models.items():
            models.setdefault(role, str(model_id))
        current = await knowledge.update_collection_models(
            domain_id=domain_id,
            collection_id=collection_id,
            payload={"models": models, "expected_row_version": row_version},
            auth_context=auth_context,
        )
        row_version = int(current.get("row_version") or row_version)
    if payload.status is not None:
        current = await knowledge.change_collection_status(
            domain_id=domain_id,
            collection_id=collection_id,
            status=payload.status,
            auth_context=auth_context,
        )
    return current


@router.delete("/knowledge-cores/{collection_id}", status_code=status.HTTP_202_ACCEPTED)
async def delete_knowledge_core(collection_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "assistant:knowledge_core_manage")
    return await _knowledge(request).delete_collection(
        domain_id=domain_id,
        collection_id=collection_id,
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/connector-capabilities")
async def list_assistant_data_connector_capabilities(request: Request):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_capabilities(
        auth_context=request.state.auth_context
    )


@router.post("/data-models/data-sources/test-connection")
async def test_assistant_data_source(
    payload: AssistantDataSourceConnectionPayload,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_test_connection(
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/data-sources")
async def list_assistant_data_sources(
    request: Request,
    cursor: UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_list(
        resource="data-sources",
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )


@router.post(
    "/data-models/data-sources", status_code=status.HTTP_201_CREATED
)
async def create_assistant_data_source(
    payload: AssistantDataSourceCreatePayload,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_create(
        resource="data-sources",
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/data-sources/{data_source_id}")
async def get_assistant_data_source(
    data_source_id: UUID,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_get(
        resource="data-sources",
        resource_id=data_source_id,
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/data-sources/{data_source_id}/snapshots")
async def list_assistant_data_source_snapshots(
    data_source_id: UUID,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_action(
        method="GET",
        path=f"data-sources/{data_source_id}/snapshots",
        payload=None,
        auth_context=request.state.auth_context,
    )


@router.post(
    "/data-models/data-sources/{data_source_id}/snapshots",
    status_code=status.HTTP_202_ACCEPTED,
)
async def request_assistant_data_source_snapshot(
    data_source_id: UUID,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_request_snapshot(
        data_source_id=data_source_id,
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/snapshots/{snapshot_id}")
async def get_assistant_schema_snapshot(
    snapshot_id: UUID,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_action(
        method="GET",
        path=f"snapshots/{snapshot_id}",
        payload=None,
        auth_context=request.state.auth_context,
    )


@router.post("/data-models/snapshots/{snapshot_id}/selection")
async def select_assistant_schema_objects(
    snapshot_id: UUID,
    payload: AssistantSchemaSelectionPayload,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_action(
        method="POST",
        path=f"snapshots/{snapshot_id}/selection",
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.post(
    "/data-models/snapshots/{snapshot_id}/objects/{object_id}/retry"
)
async def retry_assistant_schema_object(
    snapshot_id: UUID,
    object_id: UUID,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_action(
        method="POST",
        path=f"snapshots/{snapshot_id}/objects/{object_id}/retry",
        payload={},
        auth_context=request.state.auth_context,
    )


@router.post(
    "/data-models/snapshots/{snapshot_id}/objects/{object_id}/manual-ddl"
)
async def supply_assistant_schema_object(
    snapshot_id: UUID,
    object_id: UUID,
    payload: AssistantManualSchemaPayload,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_action(
        method="POST",
        path=f"snapshots/{snapshot_id}/objects/{object_id}/manual-ddl",
        payload=payload.model_dump(),
        auth_context=request.state.auth_context,
    )


@router.post(
    "/data-models/snapshots/{snapshot_id}/semantic-model-draft",
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_assistant_semantic_model_draft(
    snapshot_id: UUID,
    payload: AssistantSemanticModelCandidatePayload,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_action(
        method="POST",
        path=f"snapshots/{snapshot_id}/semantic-model-draft",
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/generation-jobs/{generation_job_id}")
async def get_assistant_semantic_model_generation(
    generation_job_id: UUID,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_get(
        resource="semantic-model-generation-jobs",
        resource_id=generation_job_id,
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/policy-subjects")
async def list_assistant_policy_subjects(request: Request):
    domain_id, _, _ = await _require(
        request, "assistant:data_model_manage"
    )
    return await _access(request).list_policy_subjects(
        app_id="assistant", domain_id=domain_id
    )


@router.get("/data-models/policy-bindings")
async def list_assistant_policy_bindings(
    request: Request,
    cursor: UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_list(
        resource="policy-bindings",
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )


@router.post(
    "/data-models/policy-bindings", status_code=status.HTTP_201_CREATED
)
async def create_assistant_policy_binding(
    payload: AssistantPolicyBindingCreatePayload,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    if not payload.actor_ids and not payload.roles:
        raise HTTPException(
            422,
            {
                "code": "POLICY_SUBJECT_REQUIRED",
                "message": "至少选择一个用户或角色",
            },
        )
    return await _data_query(request).management_create(
        resource="policy-bindings",
        payload={
            "semantic_model_ids": [
                str(item) for item in payload.semantic_model_ids
            ],
            "subject_selector": {
                "actor_ids": list(payload.actor_ids),
                "roles": list(payload.roles),
            },
            "budget": payload.budget.model_dump(),
        },
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/agent-bindings")
async def list_assistant_agent_query_bindings(
    request: Request,
    cursor: UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_list(
        resource="agent-bindings",
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/agents")
async def list_assistant_agents_for_query_binding(request: Request):
    domain_id, _, _ = await _require(
        request, "assistant:data_model_manage"
    )
    return await _client(request).list_agents(
        domain_id=domain_id,
        auth_context=request.state.auth_context,
    )


@router.post(
    "/data-models/agent-bindings", status_code=status.HTTP_201_CREATED
)
async def create_assistant_agent_query_binding(
    payload: AssistantAgentQueryBindingCreatePayload,
    request: Request,
):
    domain_id, _, _ = await _require(
        request, "assistant:data_model_manage"
    )
    agent = await _client(request).get_agent(
        agent_id=payload.agent_id,
        domain_id=domain_id,
        auth_context=request.state.auth_context,
    )
    agent_version_id = agent.get("agent_version_id")
    if not agent_version_id:
        raise HTTPException(
            409,
            {
                "code": "AGENT_VERSION_MISSING",
                "message": "Assistant Agent 缺少当前版本，无法创建问数绑定",
            },
        )
    if agent.get("status") != "DRAFT":
        raise HTTPException(
            409,
            {
                "code": "AGENT_DRAFT_REQUIRED",
                "message": "请先将 Assistant Agent 保存为草稿，再创建问数绑定",
            },
        )
    return await _data_query(request).management_create(
        resource="agent-bindings",
        payload={
            "consumer_app_id": "assistant",
            "agent_id": str(payload.agent_id),
            "agent_version_id": str(agent_version_id),
            "semantic_model_id": str(payload.semantic_model_id),
            "policy_binding_id": str(payload.policy_binding_id),
        },
        auth_context=request.state.auth_context,
    )


@router.get("/data-models")
async def list_assistant_semantic_models(
    request: Request,
    cursor: UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_list(
        resource="semantic-models",
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/{semantic_model_id}")
async def get_assistant_semantic_model(
    semantic_model_id: UUID,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_get(
        resource="semantic-models",
        resource_id=semantic_model_id,
        auth_context=request.state.auth_context,
    )


@router.patch(
    "/data-models/{semantic_model_id}/versions/{semantic_model_version_id}"
)
async def update_assistant_semantic_model_draft(
    semantic_model_id: UUID,
    semantic_model_version_id: UUID,
    payload: AssistantSemanticModelDraftUpdatePayload,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_action(
        method="PATCH",
        path=(
            f"semantic-models/{semantic_model_id}/versions/"
            f"{semantic_model_version_id}"
        ),
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.post(
    "/data-models/{semantic_model_id}/versions/"
    "{semantic_model_version_id}/validations",
    status_code=status.HTTP_202_ACCEPTED,
)
async def validate_assistant_semantic_model(
    semantic_model_id: UUID,
    semantic_model_version_id: UUID,
    payload: AssistantSemanticModelValidationPayload,
    request: Request,
    idempotency_key: str = Header(
        alias="Idempotency-Key", min_length=8, max_length=128
    ),
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_action(
        method="POST",
        path=(
            f"semantic-models/{semantic_model_id}/versions/"
            f"{semantic_model_version_id}/validations"
        ),
        payload={
            **payload.model_dump(mode="json"),
            "idempotency_key": idempotency_key,
        },
        auth_context=request.state.auth_context,
    )


@router.get(
    "/data-models/{semantic_model_id}/versions/"
    "{semantic_model_version_id}/validations/{run_id}"
)
async def get_assistant_semantic_model_validation(
    semantic_model_id: UUID,
    semantic_model_version_id: UUID,
    run_id: UUID,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    return await _data_query(request).management_action(
        method="GET",
        path=(
            f"semantic-models/{semantic_model_id}/versions/"
            f"{semantic_model_version_id}/validations/{run_id}"
        ),
        payload=None,
        auth_context=request.state.auth_context,
    )


@router.post(
    "/data-models/{semantic_model_id}/versions/"
    "{semantic_model_version_id}/submit-review",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def submit_assistant_semantic_model_review(
    semantic_model_id: UUID,
    semantic_model_version_id: UUID,
    payload: AssistantSemanticModelReviewPayload,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    await _data_query(request).management_submit_model_review(
        semantic_model_id=semantic_model_id,
        semantic_model_version_id=semantic_model_version_id,
        payload=payload.model_dump(),
        auth_context=request.state.auth_context,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/data-models/{semantic_model_id}/versions/"
    "{semantic_model_version_id}/publish",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def publish_assistant_semantic_model(
    semantic_model_id: UUID,
    semantic_model_version_id: UUID,
    payload: AssistantSemanticModelPublishPayload,
    request: Request,
):
    await _require(request, "assistant:data_model_manage")
    await _data_query(request).management_publish_model(
        semantic_model_id=semantic_model_id,
        semantic_model_version_id=semantic_model_version_id,
        payload={
            "semantic_model_id": str(semantic_model_id),
            "semantic_model_version_id": str(semantic_model_version_id),
            **payload.model_dump(mode="json"),
        },
        auth_context=request.state.auth_context,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
