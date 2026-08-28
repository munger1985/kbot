"""AIOps 专业 DBA Conversation 与 Turn 内部 API。"""

from uuid import UUID
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from aiops_agent.api.dependencies import (
    get_aiops_auth_context,
    require_service_scope,
)
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops import (
    ConversationUploadReceipt,
    ConversationStart,
    ConversationSummary,
    TurnCreate,
    TurnEventPage,
    TurnReceipt,
    TurnSummary,
    TurnView,
)


router = APIRouter(
    prefix="/internal/v1/aiops/conversations",
    tags=["AIOps Conversations"],
)


def _scope(request: Request, context: AuthContext) -> tuple[int, str]:
    require_service_scope(request, "aiops.run")
    if context.domain_id is None or int(context.domain_id) < 1:
        raise HTTPException(
            403,
            {"code": "AIOPS_DOMAIN_CONTEXT_REQUIRED"},
        )
    return int(context.domain_id), context.asserted_user_id or context.client_id


@router.post(
    "/uploads",
    status_code=201,
    response_model=ConversationUploadReceipt,
)
async def upload_conversation_input(
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    """接收有界原始文件流，返回只能由当前用户引用的上传 ID。"""
    domain_id, actor_id = _scope(request, context)
    file_name = unquote(request.headers.get("X-File-Name", "").strip())
    media_type = request.headers.get("Content-Type", "").strip()
    if not file_name or not media_type:
        raise HTTPException(
            422,
            {
                "code": "AIOPS_UPLOAD_METADATA_REQUIRED",
                "message": "上传必须提供 X-File-Name 和 Content-Type",
            },
        )
    try:
        stored = await request.app.state.conversation_upload_store.store(
            domain_id=domain_id,
            actor_id=actor_id,
            file_name=file_name,
            media_type=media_type,
            chunks=request.stream(),
        )
    except ValueError as exc:
        raise HTTPException(
            422,
            {"code": "AIOPS_UPLOAD_INVALID", "message": str(exc)},
        ) from exc
    return ConversationUploadReceipt(
        upload_id=stored.upload_id,
        file_name=stored.file_name,
        media_type=stored.media_type,
        byte_size=stored.byte_size,
        content_hash=stored.content_hash,
        expires_at=stored.expires_at,
    )


@router.post("", status_code=201, response_model=TurnReceipt)
async def start_conversation(
    payload: ConversationStart,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    """原子创建 Conversation 和第一条 Turn。"""
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.conversation_turn_service.start(
        domain_id=domain_id,
        actor_id=actor_id,
        trace_id=context.trace_id,
        conversation_create=payload.conversation,
        first_turn=payload.first_turn,
    )


@router.get("", response_model=list[ConversationSummary])
async def list_conversations(
    request: Request,
    agent_id: UUID | None = None,
    limit: int = Query(50, ge=1, le=50),
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.conversation_turn_service.list_conversations(
        domain_id=domain_id,
        actor_id=actor_id,
        agent_id=agent_id,
        limit=limit,
    )


@router.get("/{conversation_id}", response_model=ConversationSummary)
async def get_conversation(
    conversation_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.conversation_turn_service.get_conversation(
        domain_id=domain_id,
        conversation_id=conversation_id,
        actor_id=actor_id,
    )


@router.delete("/{conversation_id}", response_model=ConversationSummary)
async def archive_conversation(
    conversation_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    """从聊天历史移除会话，关联诊断和审计事实继续保留。"""
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.conversation_turn_service.archive_conversation(
        domain_id=domain_id,
        conversation_id=conversation_id,
        actor_id=actor_id,
    )


@router.post(
    "/{conversation_id}/turns",
    status_code=202,
    response_model=TurnReceipt,
)
async def create_turn(
    conversation_id: UUID,
    payload: TurnCreate,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.conversation_turn_service.create_turn(
        domain_id=domain_id,
        conversation_id=conversation_id,
        actor_id=actor_id,
        trace_id=context.trace_id,
        command=payload,
    )


@router.get(
    "/{conversation_id}/turns",
    response_model=list[TurnSummary],
)
async def list_turns(
    conversation_id: UUID,
    request: Request,
    after_turn_no: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.conversation_turn_service.list_turns(
        domain_id=domain_id,
        conversation_id=conversation_id,
        actor_id=actor_id,
        after_turn_no=after_turn_no,
        limit=limit,
    )


@router.get(
    "/{conversation_id}/turns/{turn_id}",
    response_model=TurnView,
)
async def get_turn(
    conversation_id: UUID,
    turn_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.conversation_turn_service.get_turn(
        domain_id=domain_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        actor_id=actor_id,
    )


@router.get(
    "/{conversation_id}/turns/{turn_id}/events",
    response_model=TurnEventPage,
)
async def list_turn_events(
    conversation_id: UUID,
    turn_id: UUID,
    request: Request,
    after_sequence: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=500),
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.conversation_turn_service.list_events(
        domain_id=domain_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        actor_id=actor_id,
        after_sequence=after_sequence,
        limit=limit,
    )


@router.post(
    "/{conversation_id}/turns/{turn_id}/cancel",
    response_model=TurnSummary,
)
async def cancel_turn(
    conversation_id: UUID,
    turn_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor_id = _scope(request, context)
    return await request.app.state.conversation_turn_service.cancel_turn(
        domain_id=domain_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        actor_id=actor_id,
        trace_id=context.trace_id,
    )
