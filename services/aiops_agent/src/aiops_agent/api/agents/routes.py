"""AIOps 应用私有 Agent 管理 API。"""

from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from aiops_agent.api.dependencies import get_aiops_auth_context, require_service_scope
from aiops_agent.application.agents import (
    AIOpsAgentError,
    AgentImageCapabilities,
    AgentModelBindings,
    CreateAIOpsAgentCommand,
    UpdateAIOpsAgentCommand,
    UpsertAIOpsAgentGrantCommand,
)
from platform_core.contracts import AuthContext


router = APIRouter(prefix="/internal/v1/aiops/agents", tags=["AIOps Agents"])


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentCreateRequest(_Request):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    diagnostic_source_ids: tuple[UUID, ...] = Field(min_length=1, max_length=16)
    target_ids: tuple[UUID, ...] = Field(min_length=1, max_length=32)
    auto_alert_enabled: bool = True
    auto_observe_min_severity: Literal[
        "INFO", "WARNING", "HIGH", "CRITICAL"
    ] = "CRITICAL"
    alert_cooldown_minutes: int = Field(default=15, ge=0, le=1440)
    models: AgentModelBindings = Field(default_factory=AgentModelBindings)
    image_capabilities: AgentImageCapabilities = Field(
        default_factory=AgentImageCapabilities
    )
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] = Field(default_factory=dict)
    status: Literal["DRAFT", "ACTIVE"] = "DRAFT"


class AgentUpdateRequest(_Request):
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    diagnostic_source_ids: tuple[UUID, ...] | None = Field(
        default=None, min_length=1, max_length=16
    )
    target_ids: tuple[UUID, ...] | None = Field(
        default=None, min_length=1, max_length=32
    )
    auto_alert_enabled: bool | None = None
    auto_observe_min_severity: Literal[
        "INFO", "WARNING", "HIGH", "CRITICAL"
    ] | None = None
    alert_cooldown_minutes: int | None = Field(default=None, ge=0, le=1440)
    models: AgentModelBindings | None = None
    image_capabilities: AgentImageCapabilities | None = None
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] | None = None
    status: Literal["DRAFT", "ACTIVE", "DISABLED", "ARCHIVED"] | None = None


class GrantUpsertRequest(_Request):
    agent_id: UUID
    subject_type: Literal["USER", "ROLE"]
    subject_id: str = Field(min_length=1, max_length=256)
    status: Literal["ACTIVE", "DISABLED"] = "ACTIVE"


class GrantStatusRequest(_Request):
    status: Literal["ACTIVE", "DISABLED"]
    expected_row_version: int = Field(ge=1)


class AgentAuthorizeRequest(_Request):
    agent_id: UUID
    user_id: str = Field(min_length=1, max_length=256)
    role_codes: tuple[str, ...] = ()


def _scope(request: Request, context: AuthContext) -> tuple[int, str]:
    require_service_scope(request, "aiops.manage")
    if context.domain_id is None:
        raise HTTPException(
            403, {"code": "AIOPS_AGENT_DOMAIN_CONTEXT_REQUIRED"}
        )
    return int(context.domain_id), context.asserted_user_id or context.client_id


def _raise(exc: AIOpsAgentError):
    raise HTTPException(
        exc.status_code, {"code": exc.code, "message": exc.message}
    ) from exc


@router.get("")
async def list_agents(
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.agent_service.list(domain_id=domain_id)


@router.get("/model-references/{model_id}")
async def model_references(
    model_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    require_service_scope(request, "aiops.manage")
    return {
        "references": await request.app.state.agent_service.model_references(
            model_id=model_id
        )
    }


@router.get("/action-catalog/{target_id}")
async def action_catalog(
    target_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    try:
        return await request.app.state.agent_service.action_catalog(
            domain_id=domain_id, target_id=target_id
        )
    except AIOpsAgentError as exc:
        _raise(exc)


@router.post(":authorize")
async def authorize_agent(
    payload: AgentAuthorizeRequest,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    require_service_scope(request, "aiops.run")
    if context.domain_id is None:
        raise HTTPException(
            403, {"code": "AIOPS_AGENT_DOMAIN_CONTEXT_REQUIRED"}
        )
    try:
        return await request.app.state.agent_service.authorize(
            domain_id=int(context.domain_id),
            agent_id=payload.agent_id,
            user_id=payload.user_id,
            role_codes=payload.role_codes,
        )
    except AIOpsAgentError as exc:
        _raise(exc)


@router.get("/grants/list")
async def list_grants(
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.agent_service.list_grants(domain_id=domain_id)


@router.put("/grants")
async def upsert_grant(
    payload: GrantUpsertRequest,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    try:
        return await request.app.state.agent_service.upsert_grant(
            UpsertAIOpsAgentGrantCommand(
                domain_id=domain_id,
                actor_id=actor_id,
                **payload.model_dump(),
            )
        )
    except AIOpsAgentError as exc:
        _raise(exc)


@router.patch("/grants/{grant_id}")
async def update_grant(
    grant_id: UUID,
    payload: GrantStatusRequest,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    try:
        return await request.app.state.agent_service.update_grant_status(
            domain_id=domain_id,
            grant_id=grant_id,
            status=payload.status,
            expected_row_version=payload.expected_row_version,
            actor_id=actor_id,
        )
    except AIOpsAgentError as exc:
        _raise(exc)


@router.post("", status_code=201)
async def create_agent(
    payload: AgentCreateRequest,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    try:
        return await request.app.state.agent_service.create(
            CreateAIOpsAgentCommand(
                domain_id=domain_id,
                actor_id=actor_id,
                **payload.model_dump(),
            )
        )
    except AIOpsAgentError as exc:
        _raise(exc)


@router.get("/{agent_id}")
async def get_agent(
    agent_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    try:
        return await request.app.state.agent_service.get(
            domain_id=domain_id, agent_id=agent_id
        )
    except AIOpsAgentError as exc:
        _raise(exc)


@router.patch("/{agent_id}")
async def update_agent(
    agent_id: UUID,
    payload: AgentUpdateRequest,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    try:
        return await request.app.state.agent_service.update(
            UpdateAIOpsAgentCommand(
                domain_id=domain_id,
                agent_id=agent_id,
                actor_id=actor_id,
                **payload.model_dump(exclude_unset=True),
            )
        )
    except AIOpsAgentError as exc:
        _raise(exc)


@router.get("/{agent_id}/execution-spec")
async def execution_spec(
    agent_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    try:
        return await request.app.state.agent_service.execution_spec(
            domain_id=domain_id, agent_id=agent_id
        )
    except AIOpsAgentError as exc:
        _raise(exc)
