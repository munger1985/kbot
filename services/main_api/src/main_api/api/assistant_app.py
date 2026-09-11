"""智能工作台 App 的公开 BFF 路由。"""

from typing import Any, Literal, cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from main_api.application import AccessControlService, AccessDeniedError, require_app_api_permission
from platform_clients import AssistantAppClient, DataQueryClient, KnowledgeCoreClient
from platform_core.contracts import PUBLIC_API_V1
from platform_core.security import get_auth_context


router = APIRouter(prefix=f"{PUBLIC_API_V1}/apps/assistant", tags=["Assistant App"])


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentCreatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    knowledge_core_id: UUID | None = None
    data_model_ids: tuple[UUID, ...] = Field(default=(), max_length=100)
    models: dict[str, UUID] = Field(default_factory=dict)
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] = Field(default_factory=dict)
    status: Literal["DRAFT", "ACTIVE"] = "DRAFT"


class AgentUpdatePayload(_Payload):
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    knowledge_core_id: UUID | None = None
    data_model_ids: tuple[UUID, ...] = Field(default=None, max_length=100)
    models: dict[str, UUID] | None = None
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] | None = None
    status: Literal["DRAFT", "ACTIVE", "DISABLED", "ARCHIVED"] | None = None


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


@router.get("/access")
async def get_access(request: Request):
    domain_id, actor_id = _domain_actor(request)
    snapshot = await _access(request).snapshot(app_id="assistant", domain_id=domain_id, user_id=actor_id)
    return {"app_id": snapshot.app_id, "domain_id": snapshot.domain_id, "user_id": snapshot.user_id, "roles": snapshot.roles, "permissions": sorted(snapshot.permissions)}


@router.get("/knowledge-cores")
async def list_knowledge_cores(request: Request):
    domain_id, _, _ = await _require(request, "assistant:knowledge_core_manage")
    return await _knowledge(request).list_collections(domain_id=domain_id, auth_context=request.state.auth_context)


@router.get("/agents")
async def list_agents(request: Request):
    domain_id, _, _ = await _require(request, "assistant:knowledge_chat")
    return await _client(request).list_agents(domain_id=domain_id, auth_context=request.state.auth_context)


@router.post("/agents", status_code=status.HTTP_201_CREATED)
async def create_agent(payload: AgentCreatePayload, request: Request):
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
async def update_agent(agent_id: UUID, payload: AgentUpdatePayload, request: Request):
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
