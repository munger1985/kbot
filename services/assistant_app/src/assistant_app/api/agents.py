"""智能工作台拥有的 Agent 内部 API。"""

from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from assistant_app.application import AgentApplicationError, AssistantAgentService, CreateAgentCommand, UpdateAgentCommand
from platform_core.contracts import AuthContext, ServiceIdentity


router = APIRouter(prefix="/internal/v1/assistant/agents", tags=["Assistant Agents"])


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentCreateRequest(_Request):
    domain_id: int = Field(ge=1)
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    knowledge_core_id: UUID | None = None
    data_model_ids: tuple[UUID, ...] = Field(default=(), max_length=100)
    models: dict[str, UUID] = Field(default_factory=dict)
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] = Field(default_factory=dict)
    status: Literal["DRAFT", "ACTIVE"] = "DRAFT"


class AgentUpdateRequest(_Request):
    domain_id: int = Field(ge=1)
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    knowledge_core_id: UUID | None = None
    data_model_ids: tuple[UUID, ...] = Field(default=None, max_length=100)
    models: dict[str, UUID] | None = None
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] | None = None
    status: Literal["DRAFT", "ACTIVE", "DISABLED", "ARCHIVED"] | None = None


def _service(request: Request) -> AssistantAgentService:
    return request.app.state.agent_service


def _context(request: Request, domain_id: int) -> str:
    identity = getattr(request.state, "service_identity", None)
    if not isinstance(identity, ServiceIdentity) or "assistant.manage" not in identity.scopes:
        raise HTTPException(403, {"code": "SERVICE_SCOPE_DENIED"})
    context = getattr(request.state, "auth_context", None)
    if not isinstance(context, AuthContext) or context.domain_id is None:
        raise HTTPException(403, {"code": "DOMAIN_CONTEXT_REQUIRED"})
    if int(context.domain_id) != domain_id:
        raise HTTPException(403, {"code": "DOMAIN_SCOPE_MISMATCH"})
    return context.asserted_user_id or context.client_id


def _raise(exc: AgentApplicationError) -> None:
    raise HTTPException(exc.status_code, {"code": exc.code, "message": exc.message}) from exc


@router.get("")
async def list_agents(domain_id: int, request: Request):
    _context(request, domain_id)
    return await _service(request).list(domain_id=domain_id)


@router.post("", status_code=201)
async def create_agent(payload: AgentCreateRequest, request: Request):
    try:
        return await _service(request).create(CreateAgentCommand(
            actor_id=_context(request, payload.domain_id), **payload.model_dump()
        ))
    except AgentApplicationError as exc:
        _raise(exc)


@router.get("/{agent_id}")
async def get_agent(agent_id: UUID, domain_id: int, request: Request):
    _context(request, domain_id)
    try:
        return await _service(request).get(domain_id=domain_id, agent_id=agent_id)
    except AgentApplicationError as exc:
        _raise(exc)


@router.patch("/{agent_id}")
async def update_agent(agent_id: UUID, payload: AgentUpdateRequest, request: Request):
    try:
        return await _service(request).update(UpdateAgentCommand(
            agent_id=agent_id,
            actor_id=_context(request, payload.domain_id),
            **payload.model_dump(exclude_unset=True),
        ))
    except AgentApplicationError as exc:
        _raise(exc)
