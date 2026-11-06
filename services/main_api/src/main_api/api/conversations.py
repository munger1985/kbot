"""知识检索 App 的长期记忆入口，不保留旧 Collection 会话 API。"""

from uuid import UUID
from fastapi import APIRouter, Depends, Query, Request, Response
from platform_core.contracts import MemoryItemView, PUBLIC_API_V1
from main_api.application import authorize_app_request, require_app_api_agent

async def _require_use(request: Request):
    await authorize_app_request(request, app_id="knowledge_retrieval", permission="knowledge_retrieval:use")

memory_router = APIRouter(prefix=f"{PUBLIC_API_V1}/apps/knowledge-retrieval/memories",
    tags=["Knowledge Retrieval Memories"], dependencies=[Depends(_require_use)])

def _client(request):
    return request.app.state.agent_runtime_client

@memory_router.get("", response_model=list[MemoryItemView])
async def list_memories(
    request: Request,
    agent_id: UUID = Query(),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[MemoryItemView]:
    require_app_api_agent(request, agent_id)
    rows = await _client(request).list_memories(
        agent_id=agent_id,
        limit=limit,
        auth_context=request.state.auth_context,
    )
    return [MemoryItemView.model_validate(row) for row in rows]


@memory_router.delete("/{memory_id}", status_code=204)
async def forget_memory(memory_id: UUID, request: Request) -> Response:
    await _client(request).forget_memory(
        memory_id=memory_id,
        auth_context=request.state.auth_context,
    )
    return Response(status_code=204)
