"""Portal 可见的结构化问数辅助 API。"""

from typing import Any, cast

from fastapi import APIRouter, Request

from platform_clients import AgentRuntimeClient
from platform_core.contracts import PUBLIC_API_V1


router = APIRouter(prefix=f"{PUBLIC_API_V1}/data", tags=["Data Query"])


@router.get("/profiles")
async def list_data_profiles(request: Request) -> Any:
    client = cast(
        AgentRuntimeClient, request.app.state.agent_runtime_client
    )
    return await client.list_data_profiles(
        auth_context=request.state.auth_context
    )
