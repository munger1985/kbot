"""知识检索 App 的公开 BFF 路由。"""

from typing import Any, Literal, cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from main_api.application import (
    authorize_app_request,
    require_app_api_agent,
)
from platform_core.authorization import can_read_agent, filter_readable_agents
from platform_clients import DataQueryClient, KnowledgeRetrievalAppClient
from platform_core.contracts import PUBLIC_API_V1


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/apps/knowledge-retrieval",
    tags=["Knowledge Retrieval App"],
)


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KnowledgeAgentCreatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    enabled_capabilities: tuple[
        Literal["conversation", "document", "data_query"], ...
    ] = Field(min_length=1, max_length=3)
    models: dict[str, UUID] = Field(default_factory=dict)
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
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] | None = None
    status: Literal["DRAFT", "ACTIVE", "DISABLED", "ARCHIVED"] | None = None


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
    return await authorize_app_request(
        request,
        app_id="knowledge_retrieval",
        permission=permission,
    )


@router.get("/access")
async def get_access(request: Request):
    _, _, snapshot = await authorize_app_request(
        request,
        app_id="knowledge_retrieval",
        permission="knowledge_retrieval:use",
    )
    return {
        "app_id": snapshot.app_id, "domain_id": snapshot.domain_id,
        "user_id": snapshot.user_id, "roles": snapshot.roles,
        "permissions": sorted(snapshot.permissions),
    }


@router.get("/agents")
async def list_agents(request: Request):
    domain_id, _, snapshot = await _require(
        request, "knowledge_retrieval:use"
    )
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
    require_app_api_agent(request, agent_id)
    agent = await _client(request).get_agent(
        agent_id=agent_id, domain_id=domain_id,
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
            "enabled_capabilities", "models", "instruction", "config"
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
