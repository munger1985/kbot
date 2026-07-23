"""Portal 可见的 Agent Definition 管理接口。"""

from typing import cast
from uuid import UUID

from fastapi import APIRouter, Request

from platform_clients import AgentRuntimeClient
from platform_core.contracts import (
    AgentDefinition,
    CreateAgentDefinitionRequest,
    PUBLIC_API_V1,
    UpdateAgentDefinitionRequest,
)


router = APIRouter(prefix=f"{PUBLIC_API_V1}/agents", tags=["Agents"])


def _client(request: Request) -> AgentRuntimeClient:
    return cast(
        AgentRuntimeClient,
        request.app.state.agent_runtime_client,
    )


@router.post("", status_code=201, response_model=AgentDefinition)
async def create_agent(
    payload: CreateAgentDefinitionRequest,
    request: Request,
) -> AgentDefinition:
    result = await _client(request).create_agent(
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )
    return AgentDefinition.model_validate(result)


@router.get("", response_model=list[AgentDefinition])
async def list_agents(request: Request) -> list[AgentDefinition]:
    rows = await _client(request).list_agents(
        auth_context=request.state.auth_context
    )
    return [AgentDefinition.model_validate(row) for row in rows]


@router.get("/{agent_id}", response_model=AgentDefinition)
async def get_agent(
    agent_id: UUID, request: Request
) -> AgentDefinition:
    result = await _client(request).get_agent(
        agent_id=agent_id,
        auth_context=request.state.auth_context,
    )
    return AgentDefinition.model_validate(result)


@router.patch("/{agent_id}", response_model=AgentDefinition)
async def update_agent(
    agent_id: UUID,
    payload: UpdateAgentDefinitionRequest,
    request: Request,
) -> AgentDefinition:
    result = await _client(request).update_agent(
        agent_id=agent_id,
        payload=payload.model_dump(
            mode="json", exclude_unset=True
        ),
        auth_context=request.state.auth_context,
    )
    return AgentDefinition.model_validate(result)
