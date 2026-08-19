"""KM Asset Agent 内部 API。"""

from uuid import UUID

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from km_asset_app.api.assets import _context, _raise


router = APIRouter(prefix="/internal/v1/km-asset/agents", tags=["KM Asset Agents"])


class AgentCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain_id: int = Field(ge=1)
    source_id: UUID
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    models: dict[str, UUID] = Field(default_factory=dict)
    instruction: str | None = Field(default=None, max_length=32000)
    status: str = Field(default="DRAFT", pattern=r"^(DRAFT|ACTIVE)$")


class AgentActivateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain_id: int = Field(ge=1)
    expected_row_version: int = Field(ge=1)


class AgentUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain_id: int = Field(ge=1)
    expected_row_version: int = Field(ge=1)
    source_id: UUID
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    models: dict[str, UUID] = Field(default_factory=dict)
    instruction: str | None = Field(default=None, max_length=32000)


@router.get("")
async def list_agents(domain_id: int, request: Request):
    _context(request, domain_id)
    return await request.app.state.km_agent_service.list(domain_id=domain_id)


@router.post("", status_code=201)
async def create_agent(payload: AgentCreateRequest, request: Request):
    actor_id = _context(request, payload.domain_id)
    try:
        return await request.app.state.km_agent_service.create(actor_id=actor_id, **payload.model_dump())
    except Exception as exc:
        from km_asset_app.application import KmAssetApplicationError
        if isinstance(exc, KmAssetApplicationError):
            _raise(exc)
        raise


@router.get("/{agent_id}")
async def get_agent(agent_id: UUID, domain_id: int, request: Request):
    _context(request, domain_id)
    try:
        return await request.app.state.km_agent_service.get(domain_id=domain_id, agent_id=agent_id)
    except Exception as exc:
        from km_asset_app.application import KmAssetApplicationError
        if isinstance(exc, KmAssetApplicationError):
            _raise(exc)
        raise


@router.patch("/{agent_id}")
async def update_agent(
    agent_id: UUID, payload: AgentUpdateRequest, request: Request
):
    actor_id = _context(request, payload.domain_id)
    try:
        return await request.app.state.km_agent_service.update(
            agent_id=agent_id,
            actor_id=actor_id,
            **payload.model_dump(),
        )
    except Exception as exc:
        from km_asset_app.application import KmAssetApplicationError
        if isinstance(exc, KmAssetApplicationError):
            _raise(exc)
        raise


@router.post("/{agent_id}/activate")
async def activate_agent(agent_id: UUID, payload: AgentActivateRequest, request: Request):
    actor_id = _context(request, payload.domain_id)
    try:
        return await request.app.state.km_agent_service.activate(
            domain_id=payload.domain_id,
            agent_id=agent_id,
            expected_row_version=payload.expected_row_version,
            actor_id=actor_id,
        )
    except Exception as exc:
        from km_asset_app.application import KmAssetApplicationError
        if isinstance(exc, KmAssetApplicationError):
            _raise(exc)
        raise


@router.get("/{agent_id}/execution-spec")
async def execution_spec(agent_id: UUID, domain_id: int, request: Request):
    identity = request.state.service_identity
    scope = (
        "km_asset.manage"
        if "km_asset.manage" in identity.scopes
        else "km_asset.slack.dispatch"
    )
    actor_id = _context(request, domain_id, scope)
    try:
        return await request.app.state.km_agent_service.execution_spec(
            domain_id=domain_id,
            agent_id=agent_id,
            actor_id=actor_id,
        )
    except Exception as exc:
        from km_asset_app.application import KmAssetApplicationError
        if isinstance(exc, KmAssetApplicationError):
            _raise(exc)
        raise
