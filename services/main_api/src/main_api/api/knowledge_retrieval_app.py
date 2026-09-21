"""知识检索 App 的公开 BFF 路由。"""

import re
from datetime import date
from typing import Any, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from main_api.api.models import ModelCatalogItem, load_model_catalog
from main_api.application import (
    DomainConflictError,
    DomainLifecycleError,
    DomainManagementService,
    UserAuthService,
    authorize_app_request,
    authorize_app_request_any,
)
from platform_core.authorization import can_read_agent, filter_readable_agents
from platform_clients import (
    AgentRuntimeClient,
    KnowledgeRetrievalAppClient,
    KnowledgeRetrievalAppClientError,
    DataQueryClient,
    KnowledgeCoreClient,
)
from platform_core.contracts import (
    PUBLIC_API_V1,
    UpdateConversationRequest,
    is_grok_provider_model_name,
)
from platform_core.contracts.data_query import SemanticModelDefinition
from platform_core.dictionary import (
    ModelCategory,
    coerce_model_category,
    is_enabled_model_status,
)

from main_api.api.runs import (
    _DocumentReference,
    _document_locator,
    _effective_security_level,
    _event_stream,
    _parse_cursor,
    _preview_type,
    _reference_not_found,
)


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/apps/knowledge-retrieval",
    tags=["Knowledge Retrieval App"],
)

KNOWLEDGE_RETRIEVAL_PORTAL_DOMAIN_NAME = "knowledge_retrieval_portal"
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "REJECTED"})


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KnowledgeRetrievalLoginPayload(_Payload):
    user_id: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=256)


class KnowledgeRetrievalPasswordChangePayload(_Payload):
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


class KnowledgeRetrievalAgentCreatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    knowledge_core_id: UUID | None = None
    data_model_ids: tuple[UUID, ...] = Field(default=(), max_length=100)
    models: dict[str, UUID] = Field(default_factory=dict)
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] = Field(default_factory=dict)
    status: Literal["DRAFT", "ACTIVE"] = "DRAFT"


class KnowledgeRetrievalAgentUpdatePayload(_Payload):
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    knowledge_core_id: UUID | None = None
    data_model_ids: tuple[UUID, ...] = Field(default=None, max_length=100)
    models: dict[str, UUID] | None = None
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] | None = None
    status: Literal["DRAFT", "ACTIVE", "DISABLED", "ARCHIVED"] | None = None


class KnowledgeRetrievalConversationCreatePayload(_Payload):
    agent_id: UUID
    title: str | None = Field(default=None, min_length=1, max_length=512)
    retention_policy: str = Field(
        default="DEFAULT",
        pattern=r"^(DEFAULT|KEEP_FOREVER|DAYS_30|DAYS_90|DAYS_365)$",
    )


class KnowledgeRetrievalConversationTurnPayload(_Payload):
    input: str = Field(min_length=1, max_length=32000)
    expected_conversation_version: int = Field(ge=1)
    client_metadata: dict[str, Any] = Field(default_factory=dict)




class KnowledgeRetrievalResearchCreatePayload(_Payload):
    agent_id: UUID
    input: str = Field(min_length=1, max_length=32000)
    from_date: date | None = None
    to_date: date | None = None
    allowed_x_handles: tuple[str, ...] = Field(default=(), max_length=20)
    excluded_x_handles: tuple[str, ...] = Field(default=(), max_length=20)
    enable_image_understanding: bool = False
    enable_video_understanding: bool = False
    search_context_size: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"

    @model_validator(mode="after")
    def validate_filters(self) -> "KnowledgeRetrievalResearchCreatePayload":
        if self.allowed_x_handles and self.excluded_x_handles:
            raise ValueError("allowed_x_handles 与 excluded_x_handles 不能同时设置")
        if self.from_date and self.to_date and self.from_date > self.to_date:
            raise ValueError("from_date 不能晚于 to_date")
        return self




class KnowledgeRetrievalDomainCreatePayload(_Payload):
    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)


class KnowledgeRetrievalDomainUpdatePayload(_Payload):
    expected_row_version: int = Field(ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    status: Literal["ACTIVE", "DISABLED"] | None = None


class KnowledgeRetrievalKnowledgeCoreCreatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    default_security_level: int = Field(default=1, ge=0, le=999)
    embedding: UUID
    visual_embedding: UUID | None = None


class KnowledgeRetrievalKnowledgeCoreUpdatePayload(_Payload):
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    default_security_level: int | None = Field(default=None, ge=0, le=999)
    status: Literal["ACTIVE", "DISABLED"] | None = None
    embedding: UUID | None = None
    visual_embedding: UUID | None = None


class KnowledgeRetrievalIntakeReviewPayload(_Payload):
    decision: Literal["APPROVE", "REJECT"]
    comment: str | None = Field(default=None, max_length=1000)


class KnowledgeRetrievalKnowledgeCoreReprocessPayload(_Payload):
    document_version_id: UUID


class KnowledgeRetrievalDataSourceEndpointPayload(_Payload):
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


class KnowledgeRetrievalDataSourceCredentialsPayload(_Payload):
    username: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=1024)


class KnowledgeRetrievalDataSourceConnectionPayload(_Payload):
    source_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    endpoint: KnowledgeRetrievalDataSourceEndpointPayload
    credentials: KnowledgeRetrievalDataSourceCredentialsPayload


class KnowledgeRetrievalDataSourceCreatePayload(KnowledgeRetrievalDataSourceConnectionPayload):
    display_name: str = Field(min_length=1, max_length=256)
    auto_discover_schema: bool = True


class KnowledgeRetrievalSchemaSelectionPayload(_Payload):
    object_ids: tuple[UUID, ...] = Field(min_length=1, max_length=5000)


class KnowledgeRetrievalManualSchemaPayload(_Payload):
    ddl: str = Field(min_length=10, max_length=100_000)


class KnowledgeRetrievalSemanticModelCandidatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    business_context: str | None = Field(default=None, max_length=4000)
    object_ids: tuple[UUID, ...] = Field(default=(), max_length=64)
    ai_model_id: UUID | None = None
    allow_ai_metadata: bool = False


class KnowledgeRetrievalSemanticModelDraftUpdatePayload(_Payload):
    definition: SemanticModelDefinition
    expected_row_version: int = Field(ge=1)


class KnowledgeRetrievalSemanticModelValidationPayload(_Payload):
    question: str = Field(min_length=2, max_length=2000)
    ai_model_id: UUID
    allow_ai_metadata: bool = False


class KnowledgeRetrievalSemanticModelReviewPayload(_Payload):
    expected_row_version: int = Field(ge=1)


class KnowledgeRetrievalSemanticModelPublishPayload(KnowledgeRetrievalSemanticModelReviewPayload):
    schema_snapshot_id: UUID


def _client(request: Request) -> KnowledgeRetrievalAppClient:
    return cast(KnowledgeRetrievalAppClient, request.app.state.knowledge_retrieval_app_client)


def _knowledge(request: Request) -> KnowledgeCoreClient:
    return cast(KnowledgeCoreClient, request.app.state.knowledge_core_client)


def _data_query(request: Request) -> DataQueryClient:
    return cast(DataQueryClient, request.app.state.data_query_client)


def _runtime(request: Request) -> AgentRuntimeClient:
    return cast(AgentRuntimeClient, request.app.state.agent_runtime_client)


def _domains(request: Request) -> DomainManagementService:
    service = getattr(request.app.state, "domain_management_service", None)
    if service is None:
        raise RuntimeError("Domain Management Service 尚未初始化")
    return cast(DomainManagementService, service)




def _run_response(view: dict[str, Any]) -> JSONResponse:
    status_code = 200 if view.get("status") in TERMINAL_STATUSES else 202
    return JSONResponse(status_code=status_code, content=view)


async def _require(request: Request, permission: str):
    return await authorize_app_request(
        request,
        app_id="knowledge_retrieval",
        permission=permission,
    )


async def _require_any(request: Request, *permissions: str):
    return await authorize_app_request_any(
        request,
        app_id="knowledge_retrieval",
        permissions=tuple(permissions),
    )


def _auth_for_domain(request: Request, domain_id: int):
    context = request.state.auth_context
    if str(context.domain_id) == str(domain_id):
        return context
    return context.model_copy(update={"domain_id": str(domain_id)})


def _raise_domain_error(exc: DomainConflictError | DomainLifecycleError) -> None:
    if isinstance(exc, DomainConflictError):
        raise HTTPException(409, {"code": "DOMAIN_NAME_CONFLICT", "message": str(exc)}) from exc
    raise HTTPException(exc.status_code, {"code": exc.code, "message": str(exc)}) from exc


async def _knowledge_collection_models(
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


def _is_grok_model(row: dict[str, Any]) -> bool:
    return is_grok_provider_model_name(row.get("provider_model_name"))


async def _require_grok_model_id(request: Request, model_id: UUID) -> None:
    catalog = await load_model_catalog(request)
    row = next(
        (item for item in catalog if str(item.get("model_id")) == str(model_id)),
        None,
    )
    if row is None or coerce_model_category(row.get("category")) != ModelCategory.LLM:
        raise HTTPException(
            422,
            {"code": "X_SEARCH_MODEL_UNAVAILABLE", "message": "X Search LLM 不存在或未启用"},
        )
    if not _is_grok_model(row):
        raise HTTPException(
            422,
            {"code": "X_SEARCH_GROK_REQUIRED", "message": "X Search 只能绑定 Grok 模型"},
        )


async def _agent_x_search_model(
    request: Request, *, domain_id: int, agent_id: UUID, snapshot
) -> UUID:
    agent = await _client(request).get_agent(
        agent_id=agent_id,
        domain_id=domain_id,
        auth_context=request.state.auth_context,
    )
    if not can_read_agent(
        app_id="knowledge_retrieval",
        domain_id=domain_id,
        principal_kind=request.state.auth_context.principal_kind,
        permissions=snapshot.permissions,
        authorized_agent_ids=request.state.auth_context.authorized_agent_ids,
        agent=agent,
    ):
        raise HTTPException(404, {"code": "AGENT_NOT_FOUND"})
    if agent.get("status") != "ACTIVE":
        raise HTTPException(422, {"code": "AGENT_NOT_ACTIVE", "message": "Agent 未启用"})
    model_id = (agent.get("models") or {}).get("x_search_llm")
    if not model_id:
        raise HTTPException(
            422,
            {"code": "AGENT_X_SEARCH_MODEL_REQUIRED", "message": "Agent 尚未绑定 X Search Grok 模型"},
        )
    normalized = UUID(str(model_id))
    await _require_grok_model_id(request, normalized)
    return normalized


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




async def _require_active_knowledge_core(request: Request, *, domain_id: int, knowledge_core_id: UUID | None) -> None:
    if knowledge_core_id is None:
        raise HTTPException(422, {"code": "AGENT_KNOWLEDGE_CORE_REQUIRED", "message": "启用 Agent 前必须绑定一个可用 Knowledge Core"})
    core = await _knowledge(request).get_collection(
        domain_id=domain_id, collection_id=knowledge_core_id,
        auth_context=request.state.auth_context,
    )
    if core.get("status") != "ACTIVE":
        raise HTTPException(422, {"code": "AGENT_KNOWLEDGE_CORE_INACTIVE", "message": "绑定的 Knowledge Core 未处于 ACTIVE 状态"})


async def _require_published_data_models(
    request: Request, *, data_model_ids: tuple[UUID, ...] | list[Any],
) -> tuple[UUID, ...]:
    """确认 Agent 选择的问数模型均属于当前 Domain 且已有发布版本。"""
    normalized = tuple(UUID(str(value)) for value in data_model_ids)
    if len(normalized) != len(set(normalized)):
        raise HTTPException(
            422,
            {"code": "AGENT_DATA_MODEL_DUPLICATED", "message": "问数模型不能重复"},
        )
    if not normalized:
        return normalized
    response = await _data_query(request).management_list(
        resource="semantic-models",
        cursor=None,
        limit=200,
        auth_context=request.state.auth_context,
    )
    available = {
        UUID(str(item["semantic_model_id"]))
        for item in _collection_items(response)
        if item.get("semantic_model_id") and item.get("active_version") is not None
    }
    missing = [str(item) for item in normalized if item not in available]
    if missing:
        raise HTTPException(
            422,
            {
                "code": "AGENT_DATA_MODEL_NOT_PUBLISHED",
                "message": "Agent 只能绑定当前 Domain 已发布的问数模型",
                "semantic_model_ids": missing,
            },
        )
    return normalized


async def _sync_knowledge_data_models(
    request: Request, *, agent: dict[str, Any], data_model_ids: tuple[UUID, ...],
) -> None:
    await _data_query(request).management_sync_agent_bindings(
        consumer_app_id="knowledge_retrieval",
        agent_id=UUID(str(agent["agent_id"])),
        agent_version_id=UUID(str(agent["agent_version_id"])),
        semantic_model_ids=set(data_model_ids),
        auth_context=request.state.auth_context,
    )


async def _knowledge_execution_spec(
    request: Request, *, domain_id: int, agent_id: UUID
) -> dict[str, Any]:
    """读取 KnowledgeRetrieval Agent 冻结规格并修复其唯一 KC 检索授权。"""
    spec = await _client(request).execution_spec(
        agent_id=agent_id,
        domain_id=domain_id,
        auth_context=request.state.auth_context,
    )
    resource_context = spec.get("resource_context")
    collection_ids = (
        resource_context.get("collection_ids")
        if isinstance(resource_context, dict)
        else None
    )
    if not isinstance(collection_ids, list) or len(collection_ids) != 1:
        raise HTTPException(
            409,
            {
                "code": "KNOWLEDGE_RETRIEVAL_AGENT_COLLECTION_INVALID",
                "message": "Agent 执行规格必须且只能绑定一个 Knowledge Core",
            },
        )
    collection_id = UUID(str(collection_ids[0]))
    await _require_active_knowledge_core(
        request, domain_id=domain_id, knowledge_core_id=collection_id
    )
    await _knowledge(request).bind_collection(
        domain_id=domain_id,
        agent_id=agent_id,
        collection_id=collection_id,
        note="智能工作台 Agent 会话检索授权",
        auth_context=request.state.auth_context,
    )
    return spec


async def _knowledge_conversation(
    request: Request, *, domain_id: int, conversation_id: UUID
) -> dict[str, Any]:
    conversation = await _runtime(request).get_conversation(
        conversation_id=conversation_id,
        auth_context=request.state.auth_context,
    )
    await _client(request).get_agent(
        agent_id=UUID(str(conversation["agent_id"])),
        domain_id=domain_id,
        auth_context=request.state.auth_context,
    )
    return conversation


async def _knowledge_run(
    request: Request, *, domain_id: int, run_id: UUID
) -> dict[str, Any]:
    run = await _runtime(request).get_run(
        run_id=run_id, auth_context=request.state.auth_context
    )
    await _client(request).get_agent(
        agent_id=UUID(str(run["agent_id"])),
        domain_id=domain_id,
        auth_context=request.state.auth_context,
    )
    return run


async def _knowledge_document_reference(
    request: Request, *, domain_id: int, run_id: UUID, citation_label: str
) -> _DocumentReference:
    await _knowledge_run(request, domain_id=domain_id, run_id=run_id)
    artifact = await _runtime(request).get_result(
        run_id=run_id, auth_context=request.state.auth_context
    )
    payload = artifact.get("payload")
    references = payload.get("references") if isinstance(payload, dict) else None
    raw = next(
        (
            item
            for item in references or []
            if isinstance(item, dict)
            and item.get("reference_type") == "DOCUMENT"
            and item.get("citation_label") == citation_label
        ),
        None,
    )
    if raw is None:
        raise _reference_not_found()
    try:
        return _DocumentReference.model_validate(raw)
    except ValueError as exc:
        raise HTTPException(
            409,
            {
                "code": "DOCUMENT_REFERENCE_INVALID",
                "message": "Run 引用缺少不可变文档定位信息",
            },
        ) from exc


@router.post("/auth/login")
async def login(payload: KnowledgeRetrievalLoginPayload, request: Request):
    """使用平台用户凭据进入固定的智能工作台 Portal Domain。"""
    service = cast(UserAuthService, request.app.state.user_auth_service)
    return await service.login_for_domain_name(
        user_id=payload.user_id.strip(), password=payload.password,
        domain_name=KNOWLEDGE_RETRIEVAL_PORTAL_DOMAIN_NAME, app_id="knowledge_retrieval",
    )


@router.post("/auth/password")
async def change_password(payload: KnowledgeRetrievalPasswordChangePayload, request: Request):
    """修改智能工作台本地用户密码。"""
    service = cast(UserAuthService, request.app.state.user_auth_service)
    return await service.change_password(
        claims=request.state.user_token_claims,
        current_password=payload.current_password,
        new_password=payload.new_password,
    )


@router.get("/access")
async def get_access(request: Request):
    _, _, snapshot = await authorize_app_request(
        request, app_id="knowledge_retrieval", permission="knowledge_retrieval:use"
    )
    return {
        "app_id": snapshot.app_id,
        "domain_id": snapshot.domain_id,
        "user_id": snapshot.user_id,
        "roles": snapshot.roles,
        "permissions": sorted(snapshot.permissions),
    }


@router.get("/model-catalog", response_model=list[ModelCatalogItem])
async def list_knowledge_model_catalog(request: Request):
    await _require_any(
        request,
        "knowledge_retrieval:agent_manage",
        "knowledge_retrieval:x_search",
        "knowledge_retrieval:knowledge_core_manage",
        "knowledge_retrieval:data_model_manage",
    )
    return await load_model_catalog(request)






@router.get("/x-search/runs")
async def list_research_runs(
    request: Request,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:x_search")
    return await _client(request).list_research_runs(
        domain_id=domain_id, auth_context=request.state.auth_context, status=status, limit=limit,
    )


@router.post("/x-search/runs")
async def create_research_run(
    payload: KnowledgeRetrievalResearchCreatePayload,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    domain_id, _, snapshot = await _require(request, "knowledge_retrieval:x_search")
    model_id = await _agent_x_search_model(
        request, domain_id=domain_id, agent_id=payload.agent_id, snapshot=snapshot,
    )
    view = await _client(request).create_research_run(
        payload={
            "domain_id": domain_id,
            "model_id": str(model_id),
            **payload.model_dump(mode="json"),
        },
        auth_context=request.state.auth_context,
        idempotency_key=idempotency_key,
    )
    return _run_response(view)


@router.get("/x-search/runs/{run_id}")
async def get_research_run(run_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:x_search")
    return await _client(request).get_research_run(
        run_id=run_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )


@router.get("/x-search/runs/{run_id}/events")
async def list_research_events(run_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:x_search")
    return await _client(request).list_research_events(
        run_id=run_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )


@router.get("/x-search/runs/{run_id}/sources")
async def list_research_sources(run_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:x_search")
    return await _client(request).list_research_sources(
        run_id=run_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )


@router.delete("/x-search/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_research_run(run_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:x_search")
    await _client(request).delete_research_run(
        run_id=run_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)






















@router.get("/domains")
async def list_domains(request: Request):
    _, actor_id, _ = await _require(request, "knowledge_retrieval:domain_manage")
    return await _domains(request).list_for_app(app_id="knowledge_retrieval", user_id=actor_id)


@router.post("/domains", status_code=status.HTTP_201_CREATED)
async def create_domain(payload: KnowledgeRetrievalDomainCreatePayload, request: Request):
    _, actor_id, _ = await _require(request, "knowledge_retrieval:domain_manage")
    try:
        return await _domains(request).create_for_app(
            app_id="knowledge_retrieval",
            name=payload.name,
            description=payload.description,
            actor_id=actor_id,
        )
    except (DomainConflictError, DomainLifecycleError) as exc:
        _raise_domain_error(exc)


@router.get("/domains/{target_domain_id}")
async def get_domain(target_domain_id: int, request: Request):
    _, actor_id, _ = await _require(request, "knowledge_retrieval:domain_manage")
    try:
        return await _domains(request).get_for_app(
            app_id="knowledge_retrieval", domain_id=target_domain_id, user_id=actor_id
        )
    except DomainLifecycleError as exc:
        _raise_domain_error(exc)


@router.patch("/domains/{target_domain_id}")
async def update_domain(
    target_domain_id: int,
    payload: KnowledgeRetrievalDomainUpdatePayload,
    request: Request,
):
    _, actor_id, _ = await _require(request, "knowledge_retrieval:domain_manage")
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
            app_id="knowledge_retrieval",
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
    _, actor_id, _ = await _require(request, "knowledge_retrieval:domain_manage")
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
            app_id="knowledge_retrieval",
            domain_id=target_domain_id,
            user_id=actor_id,
            actor_id=actor_id,
            expected_row_version=expected_row_version,
        )
    except (DomainConflictError, DomainLifecycleError) as exc:
        _raise_domain_error(exc)


@router.get("/knowledge-cores")
async def list_knowledge_cores(request: Request):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_core_manage")
    return await _knowledge(request).list_collections(
        domain_id=domain_id, auth_context=request.state.auth_context
    )


@router.post("/knowledge-cores", status_code=status.HTTP_201_CREATED)
async def create_knowledge_core(
    payload: KnowledgeRetrievalKnowledgeCoreCreatePayload,
    request: Request,
):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_core_manage")
    models = await _knowledge_collection_models(
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
            "metadata": {"owner_app_id": "knowledge_retrieval"},
        },
        auth_context=request.state.auth_context,
    )


@router.get("/knowledge-cores/{collection_id}")
async def get_knowledge_core(collection_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_core_manage")
    return await _knowledge(request).get_collection(
        domain_id=domain_id,
        collection_id=collection_id,
        auth_context=request.state.auth_context,
    )


@router.get("/knowledge-cores/{collection_id}/processing")
async def list_knowledge_core_processing(collection_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_core_manage")
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
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_core_manage")
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
    payload: KnowledgeRetrievalKnowledgeCoreReprocessPayload,
    request: Request,
):
    """重新调度一个失败文件的 KC 解析与后续索引流水线。"""
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_core_manage")
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
    payload: KnowledgeRetrievalIntakeReviewPayload,
    request: Request,
):
    """审核用户资料，批准后由 KC 启动解析和索引流水线。"""
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_core_manage")
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
    """将 KnowledgeRetrieval App 的用户文件流式转发至 KC，不在 Main API 落盘。"""
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_core_manage")
    content_type = request.headers.get("Content-Type", "")
    idempotency_key = request.headers.get("Idempotency-Key", "").strip()
    if not content_type.lower().startswith("multipart/form-data") or not idempotency_key:
        raise HTTPException(
            status.HTTP_428_PRECONDITION_REQUIRED,
            {
                "code": "KNOWLEDGE_RETRIEVAL_KC_UPLOAD_HEADERS_REQUIRED",
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
    payload: KnowledgeRetrievalKnowledgeCoreUpdatePayload,
    request: Request,
):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_core_manage")
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
        models = await _knowledge_collection_models(
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
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_core_manage")
    return await _knowledge(request).delete_collection(
        domain_id=domain_id,
        collection_id=collection_id,
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/connector-capabilities")
async def list_knowledge_retrieval_data_connector_capabilities(request: Request):
    await _require(request, "knowledge_retrieval:data_model_manage")
    return await _data_query(request).management_capabilities(
        auth_context=request.state.auth_context
    )


@router.post("/data-models/data-sources/test-connection")
async def test_knowledge_retrieval_data_source(
    payload: KnowledgeRetrievalDataSourceConnectionPayload,
    request: Request,
):
    await _require(request, "knowledge_retrieval:data_model_manage")
    return await _data_query(request).management_test_connection(
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/data-sources")
async def list_knowledge_retrieval_data_sources(
    request: Request,
    cursor: UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    await _require(request, "knowledge_retrieval:data_model_manage")
    return await _data_query(request).management_list(
        resource="data-sources",
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )


@router.post(
    "/data-models/data-sources", status_code=status.HTTP_201_CREATED
)
async def create_knowledge_retrieval_data_source(
    payload: KnowledgeRetrievalDataSourceCreatePayload,
    request: Request,
):
    await _require(request, "knowledge_retrieval:data_model_manage")
    return await _data_query(request).management_create(
        resource="data-sources",
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/data-sources/{data_source_id}")
async def get_knowledge_retrieval_data_source(
    data_source_id: UUID,
    request: Request,
):
    await _require(request, "knowledge_retrieval:data_model_manage")
    return await _data_query(request).management_get(
        resource="data-sources",
        resource_id=data_source_id,
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/data-sources/{data_source_id}/snapshots")
async def list_knowledge_retrieval_data_source_snapshots(
    data_source_id: UUID,
    request: Request,
):
    await _require(request, "knowledge_retrieval:data_model_manage")
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
async def request_knowledge_retrieval_data_source_snapshot(
    data_source_id: UUID,
    request: Request,
):
    await _require(request, "knowledge_retrieval:data_model_manage")
    return await _data_query(request).management_request_snapshot(
        data_source_id=data_source_id,
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/snapshots/{snapshot_id}")
async def get_knowledge_retrieval_schema_snapshot(
    snapshot_id: UUID,
    request: Request,
):
    await _require(request, "knowledge_retrieval:data_model_manage")
    return await _data_query(request).management_action(
        method="GET",
        path=f"snapshots/{snapshot_id}",
        payload=None,
        auth_context=request.state.auth_context,
    )


@router.post("/data-models/snapshots/{snapshot_id}/selection")
async def select_knowledge_retrieval_schema_objects(
    snapshot_id: UUID,
    payload: KnowledgeRetrievalSchemaSelectionPayload,
    request: Request,
):
    await _require(request, "knowledge_retrieval:data_model_manage")
    return await _data_query(request).management_action(
        method="POST",
        path=f"snapshots/{snapshot_id}/selection",
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.post(
    "/data-models/snapshots/{snapshot_id}/objects/{object_id}/retry"
)
async def retry_knowledge_retrieval_schema_object(
    snapshot_id: UUID,
    object_id: UUID,
    request: Request,
):
    await _require(request, "knowledge_retrieval:data_model_manage")
    return await _data_query(request).management_action(
        method="POST",
        path=f"snapshots/{snapshot_id}/objects/{object_id}/retry",
        payload={},
        auth_context=request.state.auth_context,
    )


@router.post(
    "/data-models/snapshots/{snapshot_id}/objects/{object_id}/manual-ddl"
)
async def supply_knowledge_retrieval_schema_object(
    snapshot_id: UUID,
    object_id: UUID,
    payload: KnowledgeRetrievalManualSchemaPayload,
    request: Request,
):
    await _require(request, "knowledge_retrieval:data_model_manage")
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
async def generate_knowledge_retrieval_semantic_model_draft(
    snapshot_id: UUID,
    payload: KnowledgeRetrievalSemanticModelCandidatePayload,
    request: Request,
):
    await _require(request, "knowledge_retrieval:data_model_manage")
    return await _data_query(request).management_action(
        method="POST",
        path=f"snapshots/{snapshot_id}/semantic-model-draft",
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/generation-jobs/{generation_job_id}")
async def get_knowledge_retrieval_semantic_model_generation(
    generation_job_id: UUID,
    request: Request,
):
    await _require(request, "knowledge_retrieval:data_model_manage")
    return await _data_query(request).management_get(
        resource="semantic-model-generation-jobs",
        resource_id=generation_job_id,
        auth_context=request.state.auth_context,
    )


@router.get("/data-models")
async def list_knowledge_retrieval_semantic_models(
    request: Request,
    cursor: UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    await _require(request, "knowledge_retrieval:data_model_manage")
    return await _data_query(request).management_list(
        resource="semantic-models",
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )


@router.get("/data-models/{semantic_model_id}")
async def get_knowledge_retrieval_semantic_model(
    semantic_model_id: UUID,
    request: Request,
):
    await _require(request, "knowledge_retrieval:data_model_manage")
    return await _data_query(request).management_get(
        resource="semantic-models",
        resource_id=semantic_model_id,
        auth_context=request.state.auth_context,
    )


@router.patch(
    "/data-models/{semantic_model_id}/versions/{semantic_model_version_id}"
)
async def update_knowledge_retrieval_semantic_model_draft(
    semantic_model_id: UUID,
    semantic_model_version_id: UUID,
    payload: KnowledgeRetrievalSemanticModelDraftUpdatePayload,
    request: Request,
):
    await _require(request, "knowledge_retrieval:data_model_manage")
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
async def validate_knowledge_retrieval_semantic_model(
    semantic_model_id: UUID,
    semantic_model_version_id: UUID,
    payload: KnowledgeRetrievalSemanticModelValidationPayload,
    request: Request,
    idempotency_key: str = Header(
        alias="Idempotency-Key", min_length=8, max_length=128
    ),
):
    await _require(request, "knowledge_retrieval:data_model_manage")
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
async def get_knowledge_retrieval_semantic_model_validation(
    semantic_model_id: UUID,
    semantic_model_version_id: UUID,
    run_id: UUID,
    request: Request,
):
    await _require(request, "knowledge_retrieval:data_model_manage")
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
async def submit_knowledge_retrieval_semantic_model_review(
    semantic_model_id: UUID,
    semantic_model_version_id: UUID,
    payload: KnowledgeRetrievalSemanticModelReviewPayload,
    request: Request,
):
    await _require(request, "knowledge_retrieval:data_model_manage")
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
async def publish_knowledge_retrieval_semantic_model(
    semantic_model_id: UUID,
    semantic_model_version_id: UUID,
    payload: KnowledgeRetrievalSemanticModelPublishPayload,
    request: Request,
):
    await _require(request, "knowledge_retrieval:data_model_manage")
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
    domain_id, _, snapshot = await _require(request, "knowledge_retrieval:use")
    agents = await _client(request).list_agents(
        domain_id=domain_id, auth_context=request.state.auth_context
    )
    return filter_readable_agents(
        app_id="knowledge_retrieval",
        domain_id=domain_id,
        principal_kind=request.state.auth_context.principal_kind,
        permissions=snapshot.permissions,
        authorized_agent_ids=request.state.auth_context.authorized_agent_ids,
        agents=agents,
    )


@router.post("/conversations", status_code=status.HTTP_201_CREATED)
async def create_knowledge_conversation(
    payload: KnowledgeRetrievalConversationCreatePayload, request: Request
):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_chat")
    spec = await _knowledge_execution_spec(
        request, domain_id=domain_id, agent_id=payload.agent_id
    )
    return await _runtime(request).create_conversation(
        payload={**payload.model_dump(mode="json"), "execution_spec": spec},
        auth_context=request.state.auth_context,
    )


@router.get("/conversations")
async def list_knowledge_conversations(
    request: Request, limit: int = Query(default=50, ge=1, le=200)
):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_chat")
    agents = await _client(request).list_agents(
        domain_id=domain_id, auth_context=request.state.auth_context
    )
    agent_ids = {str(item.get("agent_id")) for item in agents}
    rows = await _runtime(request).list_conversations(
        limit=200, auth_context=request.state.auth_context
    )
    return [
        item for item in rows if str(item.get("agent_id")) in agent_ids
    ][:limit]


@router.get("/conversations/{conversation_id}")
async def get_knowledge_conversation(conversation_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_chat")
    return await _knowledge_conversation(
        request, domain_id=domain_id, conversation_id=conversation_id
    )


@router.patch("/conversations/{conversation_id}")
async def update_knowledge_conversation(
    conversation_id: UUID,
    payload: UpdateConversationRequest,
    request: Request,
):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_chat")
    await _knowledge_conversation(
        request, domain_id=domain_id, conversation_id=conversation_id
    )
    return await _runtime(request).update_conversation(
        conversation_id=conversation_id,
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.delete(
    "/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_knowledge_conversation(
    conversation_id: UUID,
    request: Request,
    expected_row_version: int = Query(ge=1),
):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_chat")
    await _knowledge_conversation(
        request, domain_id=domain_id, conversation_id=conversation_id
    )
    await _runtime(request).delete_conversation(
        conversation_id=conversation_id,
        expected_row_version=expected_row_version,
        auth_context=request.state.auth_context,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/conversations/{conversation_id}/turns",
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_knowledge_conversation_turn(
    conversation_id: UUID,
    payload: KnowledgeRetrievalConversationTurnPayload,
    request: Request,
    idempotency_key: str = Header(alias="Idempotency-Key"),
):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_chat")
    conversation = await _knowledge_conversation(
        request, domain_id=domain_id, conversation_id=conversation_id
    )
    agent_id = UUID(str(conversation["agent_id"]))
    spec = await _knowledge_execution_spec(
        request, domain_id=domain_id, agent_id=agent_id
    )
    resource_context = spec.get("resource_context") or {}
    collection_ids = list(resource_context.get("collection_ids") or [])
    security_level = await _effective_security_level(request)
    body = payload.model_dump(mode="json")
    body["execution_spec"] = spec
    body["collection_ids"] = collection_ids
    body["security_level"] = security_level
    return await _runtime(request).create_conversation_turn(
        conversation_id=conversation_id,
        payload=body,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )


@router.get("/conversations/{conversation_id}/turns")
async def list_knowledge_conversation_turns(
    conversation_id: UUID,
    request: Request,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=500),
):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_chat")
    await _knowledge_conversation(
        request, domain_id=domain_id, conversation_id=conversation_id
    )
    return await _runtime(request).list_conversation_turns(
        conversation_id=conversation_id,
        after=after,
        limit=limit,
        auth_context=request.state.auth_context,
    )


@router.get("/conversations/{conversation_id}/turns/{turn_id}/trace")
async def list_knowledge_retrieval_turn_trace(
    conversation_id: UUID,
    turn_id: UUID,
    request: Request,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_chat")
    await _knowledge_conversation(
        request, domain_id=domain_id, conversation_id=conversation_id
    )
    return await _runtime(request).list_turn_trace(
        conversation_id=conversation_id,
        turn_id=turn_id,
        after=after,
        limit=limit,
        auth_context=request.state.auth_context,
    )


@router.get("/runs/{run_id}")
async def get_knowledge_retrieval_agent_run(run_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_chat")
    return await _knowledge_run(request, domain_id=domain_id, run_id=run_id)


@router.get("/runs/{run_id}/result")
async def get_knowledge_retrieval_agent_run_result(run_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_chat")
    await _knowledge_run(request, domain_id=domain_id, run_id=run_id)
    return await _runtime(request).get_result(
        run_id=run_id, auth_context=request.state.auth_context
    )


@router.get("/runs/{run_id}/events")
async def stream_knowledge_retrieval_agent_run_events(
    run_id: UUID,
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_chat")
    cursor = _parse_cursor(last_event_id)
    summary = await _knowledge_run(request, domain_id=domain_id, run_id=run_id)
    if cursor > int(summary["event_cursor"]):
        raise HTTPException(
            400,
            {
                "code": "AGENT_EVENT_CURSOR_INVALID",
                "message": "Last-Event-ID 超过当前 Run 事件游标",
            },
        )
    return StreamingResponse(
        _event_stream(run_id=run_id, request=request, cursor=cursor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/runs/{run_id}/references/{citation_label}/preview")
async def get_knowledge_retrieval_reference_preview(
    run_id: UUID, citation_label: str, request: Request
):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_chat")
    reference = await _knowledge_document_reference(
        request,
        domain_id=domain_id,
        run_id=run_id,
        citation_label=citation_label,
    )
    preview = await _knowledge(request).get_bundle_revision_preview(
        domain_id=domain_id,
        collection_id=reference.collection_id,
        bundle_id=reference.bundle_id,
        bundle_revision_id=reference.bundle_revision_id,
        auth_context=request.state.auth_context,
    )
    source_file = next(
        (
            item
            for item in preview.get("files", [])
            if str(item.get("document_version_id"))
            == str(reference.document_version_id)
            and bool(item.get("preview_available"))
        ),
        None,
    )
    if source_file is None:
        raise _reference_not_found()
    mime_type = str(
        source_file.get("detected_mime_type")
        or source_file.get("declared_mime_type")
        or "application/octet-stream"
    ).split(";", 1)[0].strip().lower()
    page_no, page_end, bbox = _document_locator(reference)
    return {
        "reference_type": "DOCUMENT",
        "citation_label": reference.citation_label,
        "title": reference.title,
        "mime_type": mime_type,
        "preview_type": _preview_type(mime_type),
        "page_no": page_no,
        "page_end": page_end,
        "bbox": bbox,
        "content_url": (
            f"{PUBLIC_API_V1}/apps/knowledge-retrieval/runs/{run_id}/references/"
            f"{reference.citation_label}/content"
        ),
        "download_available": True,
    }


@router.get("/runs/{run_id}/references/{citation_label}/content")
async def stream_knowledge_retrieval_reference_content(
    run_id: UUID,
    citation_label: str,
    request: Request,
    range_header: str | None = Header(default=None, alias="Range"),
):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:knowledge_chat")
    reference = await _knowledge_document_reference(
        request,
        domain_id=domain_id,
        run_id=run_id,
        citation_label=citation_label,
    )
    upstream = await _knowledge(request).stream_source_file(
        domain_id=domain_id,
        collection_id=reference.collection_id,
        bundle_id=reference.bundle_id,
        bundle_revision_id=reference.bundle_revision_id,
        document_version_id=reference.document_version_id,
        range_header=range_header,
        auth_context=request.state.auth_context,
    )
    forwarded_headers = {
        name: upstream.headers[name]
        for name in (
            "accept-ranges",
            "cache-control",
            "content-disposition",
            "content-length",
            "content-range",
            "content-security-policy",
            "x-content-type-options",
        )
        if name in upstream.headers
    }
    return StreamingResponse(
        upstream.body,
        status_code=upstream.status_code,
        media_type=upstream.headers.get(
            "content-type", "application/octet-stream"
        ),
        headers=forwarded_headers,
    )


@router.get("/agents/options")
async def list_agent_management_options(request: Request):
    """返回 Agent 编辑器所需的安全资源目录，不暴露凭据和内部连接信息。"""
    domain_id, _, _ = await _require(request, "knowledge_retrieval:agent_manage")
    auth_context = request.state.auth_context
    collections = _collection_items(
        await _knowledge(request).list_collections(
            domain_id=domain_id,
            auth_context=auth_context,
        )
    )
    semantic_models = _collection_items(
        await _data_query(request).management_list(
            resource="semantic-models",
            cursor=None,
            limit=200,
            auth_context=auth_context,
        )
    )
    return {
        "knowledge_cores": [
            {
                "collection_id": item.get("collection_id"),
                "display_name": item.get("display_name"),
                "status": item.get("status"),
                "selectable": item.get("status") == "ACTIVE",
            }
            for item in collections
        ],
        "data_models": [
            {
                "semantic_model_id": item.get("semantic_model_id"),
                "display_name": item.get("display_name"),
                "description": item.get("description"),
                "active_version": item.get("active_version"),
                "selectable": item.get("active_version") is not None,
            }
            for item in semantic_models
        ],
        "models": await load_model_catalog(request),
    }


@router.post("/agents", status_code=status.HTTP_201_CREATED)
async def create_agent(payload: KnowledgeRetrievalAgentCreatePayload, request: Request):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:agent_manage")
    x_search_model_id = payload.models.get("x_search_llm")
    if x_search_model_id is not None:
        await _require_grok_model_id(request, x_search_model_id)
    data_model_ids = await _require_published_data_models(
        request, data_model_ids=payload.data_model_ids,
    )
    if payload.status == "ACTIVE":
        await _require_active_knowledge_core(request, domain_id=domain_id, knowledge_core_id=payload.knowledge_core_id)
    requested_status = payload.status
    values = payload.model_dump(mode="json")
    values["status"] = "DRAFT"
    agent = await _client(request).create_agent(
        payload={"domain_id": domain_id, **values},
        auth_context=request.state.auth_context,
    )
    await _sync_knowledge_data_models(
        request, agent=agent, data_model_ids=data_model_ids,
    )
    if requested_status == "ACTIVE":
        agent = await _client(request).update_agent(
            agent_id=UUID(str(agent["agent_id"])),
            payload={
                "domain_id": domain_id,
                "expected_row_version": int(agent["row_version"]),
                "status": "ACTIVE",
            },
            auth_context=request.state.auth_context,
        )
    return agent


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: UUID, request: Request):
    domain_id, _, snapshot = await _require(
        request, "knowledge_retrieval:use"
    )
    agent = await _client(request).get_agent(
        agent_id=agent_id,
        domain_id=domain_id,
        auth_context=request.state.auth_context,
    )
    if not can_read_agent(
        app_id="knowledge_retrieval",
        domain_id=domain_id,
        principal_kind=request.state.auth_context.principal_kind,
        permissions=snapshot.permissions,
        authorized_agent_ids=request.state.auth_context.authorized_agent_ids,
        agent=agent,
    ):
        raise HTTPException(404, {"code": "AGENT_NOT_FOUND"})
    return agent


@router.patch("/agents/{agent_id}")
async def update_agent(
    agent_id: UUID,
    payload: KnowledgeRetrievalAgentUpdatePayload,
    request: Request,
):
    domain_id, _, _ = await _require(request, "knowledge_retrieval:agent_manage")
    current = await _client(request).get_agent(agent_id=agent_id, domain_id=domain_id, auth_context=request.state.auth_context)
    values = payload.model_dump(mode="json", exclude_unset=True)
    effective_models = values.get("models", current.get("models") or {})
    x_search_model_id = effective_models.get("x_search_llm")
    if x_search_model_id is not None:
        await _require_grok_model_id(request, UUID(str(x_search_model_id)))
    status_value = values.get("status", current["status"])
    version_fields = {
        "knowledge_core_id", "data_model_ids", "models", "instruction", "config"
    }
    needs_sync = bool(version_fields.intersection(values)) or status_value == "ACTIVE"
    raw_data_model_ids = values.get(
        "data_model_ids", current.get("data_model_ids") or []
    )
    data_model_ids = (
        await _require_published_data_models(
            request, data_model_ids=raw_data_model_ids,
        )
        if needs_sync
        else tuple(UUID(str(value)) for value in raw_data_model_ids)
    )
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
    requested_status = status_value
    if requested_status == "ACTIVE":
        values["status"] = "DRAFT"
    agent = await _client(request).update_agent(
        agent_id=agent_id,
        payload={"domain_id": domain_id, **values},
        auth_context=request.state.auth_context,
    )
    if needs_sync:
        await _sync_knowledge_data_models(
            request, agent=agent, data_model_ids=data_model_ids,
        )
    if requested_status == "ACTIVE" and agent.get("status") != "ACTIVE":
        agent = await _client(request).update_agent(
            agent_id=agent_id,
            payload={
                "domain_id": domain_id,
                "expected_row_version": int(agent["row_version"]),
                "status": "ACTIVE",
            },
            auth_context=request.state.auth_context,
        )
    return agent


@router.delete("/agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def archive_agent(
    agent_id: UUID,
    request: Request,
    expected_row_version: int = Query(ge=1),
):
    """归档 Agent 并保留历史版本、运行记录及审计关系。"""
    domain_id, _, _ = await _require(request, "knowledge_retrieval:agent_manage")
    agent = await _client(request).update_agent(
        agent_id=agent_id,
        payload={
            "domain_id": domain_id,
            "expected_row_version": expected_row_version,
            "status": "ARCHIVED",
        },
        auth_context=request.state.auth_context,
    )
    await _sync_knowledge_data_models(
        request, agent=agent, data_model_ids=(),
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
