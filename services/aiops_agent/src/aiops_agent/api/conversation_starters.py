"""智能运维功能入口内部 API。"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from aiops_agent.api.dependencies import (
    get_aiops_auth_context,
    require_service_scope,
)
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops import ConversationStarterCatalogView


router = APIRouter(
    prefix="/internal/v1/aiops/conversation-starters",
    tags=["AIOps Conversation Starters"],
)


@router.get("", response_model=ConversationStarterCatalogView)
async def list_conversation_starters(
    request: Request,
    agent_id: UUID = Query(...),
    target_id: UUID = Query(...),
    context: AuthContext = Depends(get_aiops_auth_context),
):
    require_service_scope(request, "aiops.run")
    domain_id = int(context.domain_id or 0)
    if domain_id < 1:
        raise HTTPException(403, {"code": "AIOPS_DOMAIN_CONTEXT_REQUIRED"})
    return await request.app.state.conversation_starter_service.list(
        domain_id=domain_id,
        agent_id=agent_id,
        target_id=target_id,
    )
