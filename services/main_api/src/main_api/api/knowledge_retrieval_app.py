"""知识检索 App 的公开 BFF 路由。"""

from typing import Any, Literal, cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from main_api.application import (
    AccessConfigurationError,
    AccessControlService,
    AccessDeniedError,
)
from platform_clients import DataQueryClient, KnowledgeRetrievalAppClient
from platform_core.contracts import PUBLIC_API_V1
from platform_core.security import get_auth_context


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/apps/knowledge-retrieval",
    tags=["Knowledge Retrieval App"],
)


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KnowledgeMemberRolePayload(_Payload):
    display_name: str | None = Field(default=None, max_length=256)
    status: Literal["ACTIVE", "DISABLED"]


class KnowledgeAgentCreatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    enabled_capabilities: tuple[
        Literal["conversation", "document", "data_query"], ...
    ] = Field(min_length=1, max_length=3)
    models: dict[str, UUID] = Field(default_factory=dict)
    do_rerank: bool = False
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] = Field(default_factory=dict)
    status: Literal["DRAFT", "ACTIVE"] = "DRAFT"


class KnowledgeAgentUpdatePayload(_Payload):
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    enabled_capabilities: tuple[
        Literal["conversation", "document", "data_query"], ...
    ] | None = Field(default=None, min_length=1, max_length=3)
    models: dict[str, UUID] | None = None
    do_rerank: bool | None = None
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] | None = None
    status: Literal["DRAFT", "ACTIVE", "DISABLED", "ARCHIVED"] | None = None


def _domain_actor(request: Request) -> tuple[int, str]:
    context = get_auth_context(request)
    try:
        domain_id = int(context.domain_id or "")
    except ValueError as exc:
        raise HTTPException(403, {"code": "DOMAIN_CONTEXT_REQUIRED"}) from exc
    if domain_id < 1:
        raise HTTPException(403, {"code": "DOMAIN_CONTEXT_REQUIRED"})
    return domain_id, context.asserted_user_id or context.client_id


def _access(request: Request) -> AccessControlService:
    return cast(AccessControlService, request.app.state.access_control_service)


def _client(request: Request) -> KnowledgeRetrievalAppClient:
    return cast(
        KnowledgeRetrievalAppClient,
        request.app.state.knowledge_retrieval_app_client,
    )


def _data_query_client(request: Request) -> DataQueryClient:
    return cast(DataQueryClient, request.app.state.data_query_client)


def _uses_semantic_data_query(*, capabilities, config) -> bool:
    return (
        "data_query" in set(capabilities or ())
        and str((config or {}).get("data_query_mode") or "").upper() == "SEMANTIC"
    )


async def _require(request: Request, permission: str):
    domain_id, actor_id = _domain_actor(request)
    try:
        snapshot = await _access(request).require(
            app_id="knowledge_retrieval", domain_id=domain_id,
            user_id=actor_id, permission_code=permission,
        )
    except AccessDeniedError as exc:
        raise HTTPException(
            403, {"code": "APP_PERMISSION_DENIED", "permission": permission}
        ) from exc
    return domain_id, actor_id, snapshot


@router.get("/access")
async def get_access(request: Request):
    domain_id, actor_id = _domain_actor(request)
    snapshot = await _access(request).snapshot(
        app_id="knowledge_retrieval", domain_id=domain_id, user_id=actor_id
    )
    return {
        "app_id": snapshot.app_id, "domain_id": snapshot.domain_id,
        "user_id": snapshot.user_id, "roles": snapshot.roles,
        "permissions": sorted(snapshot.permissions),
    }


@router.get("/members")
async def list_members(request: Request):
    domain_id, _, _ = await _require(
        request, "knowledge_retrieval:member_manage"
    )
    return await _access(request).list_members(
        app_id="knowledge_retrieval", domain_id=domain_id
    )


@router.put("/members/{user_id}/roles/{role_code}")
async def set_member_role(
    user_id: str, role_code: str, payload: KnowledgeMemberRolePayload, request: Request
):
    domain_id, actor_id, _ = await _require(
        request, "knowledge_retrieval:member_manage"
    )
    try:
        return await _access(request).set_member_role(
            app_id="knowledge_retrieval", domain_id=domain_id,
            user_id=user_id, display_name=payload.display_name,
            role_code=role_code, status=payload.status, actor_id=actor_id,
        )
    except AccessConfigurationError as exc:
        raise HTTPException(
            422, {"code": "APP_MEMBER_CONFIGURATION_INVALID", "message": str(exc)}
        ) from exc


@router.get("/agents")
async def list_agents(request: Request):
    domain_id, _, snapshot = await _require(
        request, "knowledge_retrieval:use"
    )
    agents = await _client(request).list_agents(
        domain_id=domain_id, auth_context=request.state.auth_context
    )
    if "knowledge_retrieval:agent_manage" in snapshot.permissions:
        return agents
    return [
        item for item in agents if item.get("status") == "ACTIVE"
    ]


@router.post("/agents", status_code=status.HTTP_201_CREATED)
async def create_agent(payload: KnowledgeAgentCreatePayload, request: Request):
    domain_id, _, _ = await _require(
        request, "knowledge_retrieval:agent_manage"
    )
    if payload.status == "ACTIVE" and _uses_semantic_data_query(
        capabilities=payload.enabled_capabilities, config=payload.config,
    ):
        raise HTTPException(
            422,
            {
                "code": "APP_AGENT_QUERY_BINDING_REQUIRED",
                "message": "带问数能力的 Agent 必须先以草稿创建并配置有效查询绑定，再单独启用",
            },
        )
    return await _client(request).create_agent(
        payload={"domain_id": domain_id, **payload.model_dump(mode="json")},
        auth_context=request.state.auth_context,
    )


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: UUID, request: Request):
    domain_id, _, snapshot = await _require(
        request, "knowledge_retrieval:use"
    )
    agent = await _client(request).get_agent(
        agent_id=agent_id, domain_id=domain_id,
        auth_context=request.state.auth_context,
    )
    if (
        "knowledge_retrieval:agent_manage" not in snapshot.permissions
        and agent.get("status") != "ACTIVE"
    ):
        raise HTTPException(
            404,
            {"code": "AGENT_NOT_FOUND", "message": "Agent 不存在"},
        )
    return agent


@router.patch("/agents/{agent_id}")
async def update_agent(
    agent_id: UUID, payload: KnowledgeAgentUpdatePayload, request: Request
):
    domain_id, _, _ = await _require(
        request, "knowledge_retrieval:agent_manage"
    )
    current = await _client(request).get_agent(
        agent_id=agent_id,
        domain_id=domain_id,
        auth_context=request.state.auth_context,
    )
    values = payload.model_dump(mode="json", exclude_unset=True)
    capabilities = values.get(
        "enabled_capabilities", current.get("enabled_capabilities") or ()
    )
    config = values.get("config", current.get("config") or {})
    status_value = values.get("status", current.get("status"))
    if status_value == "ACTIVE" and _uses_semantic_data_query(
        capabilities=capabilities, config=config,
    ):
        version_fields = {
            "enabled_capabilities", "models", "do_rerank", "instruction", "config"
        }
        if version_fields.intersection(values):
            raise HTTPException(
                422,
                {
                    "code": "APP_AGENT_QUERY_BINDING_VERSION_REQUIRED",
                    "message": "请先保存 Agent 草稿版本、创建该版本的查询绑定，再单独启用",
                },
            )
        version_id = current.get("agent_version_id")
        if not version_id or not await _data_query_client(
            request
        ).management_has_active_agent_binding(
            consumer_app_id="knowledge_retrieval",
            agent_id=agent_id,
            agent_version_id=UUID(str(version_id)),
            semantic_model_ids=set(),
            auth_context=request.state.auth_context,
        ):
            raise HTTPException(
                422,
                {
                    "code": "APP_AGENT_QUERY_BINDING_REQUIRED",
                    "message": "启用问数 Agent 前必须为当前版本配置至少一个有效查询绑定",
                },
            )
    return await _client(request).update_agent(
        agent_id=agent_id,
        payload={
            "domain_id": domain_id,
            **values,
        },
        auth_context=request.state.auth_context,
    )
