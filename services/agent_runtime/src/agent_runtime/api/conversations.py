"""Conversation、Turn、公开轨迹和长期记忆的内部 API。"""

from __future__ import annotations

from typing import cast
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response

from agent_runtime.application import (
    AgentDefinitionNotFound,
    AgentRuntimeConflict,
    ConversationNotFound,
    ConversationService,
    ConversationTurnNotFound,
)
from platform_core.contracts import (
    ConversationTurnPage,
    ConversationTurnReceipt,
    ConversationView,
    CreateConversationRequest,
    CreateConversationTurnRequest,
    INTERNAL_API_V1,
    MemoryItemView,
    PublicTraceEvent,
    UpdateConversationRequest,
)


router = APIRouter(
    prefix=f"{INTERNAL_API_V1}/conversations",
    tags=["Agent Conversations"],
)
memory_router = APIRouter(
    prefix=f"{INTERNAL_API_V1}/memories",
    tags=["Agent Memories"],
)


def _service(request: Request) -> ConversationService:
    service = getattr(request.app.state, "conversation_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "CONVERSATION_SERVICE_NOT_READY",
                "message": "Conversation Service 尚未初始化",
            },
        )
    return cast(ConversationService, service)


def _identity(request: Request) -> tuple[int, str, str, str]:
    context = getattr(request.state, "auth_context", None)
    if context is None or not context.domain_id:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "DOMAIN_CONTEXT_REQUIRED",
                "message": "当前请求缺少受信 Domain 上下文",
            },
        )
    try:
        domain_id = int(context.domain_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_DOMAIN",
                "message": "Domain ID 必须是数字标识",
            },
        ) from exc
    return (
        domain_id,
        context.asserted_user_id or context.client_id,
        context.request_id,
        context.trace_id,
    )


def _raise_error(exc: Exception) -> None:
    if isinstance(
        exc,
        (
            ConversationNotFound,
            ConversationTurnNotFound,
            AgentDefinitionNotFound,
        ),
    ):
        status = 404
    elif isinstance(exc, AgentRuntimeConflict):
        status = 409
    else:
        raise exc
    raise HTTPException(
        status_code=status,
        detail={"code": exc.code, "message": str(exc)},
    )


@router.post("", status_code=201, response_model=ConversationView)
async def create_conversation(
    payload: CreateConversationRequest,
    request: Request,
) -> ConversationView:
    domain_id, actor_id, _, _ = _identity(request)
    try:
        return await _service(request).create(
            app_id=request.app.state.platform_app_id,
            domain_id=domain_id,
            actor_id=actor_id,
            agent_id=payload.agent_id,
            title=payload.title,
            retention_policy=payload.retention_policy,
        )
    except (AgentDefinitionNotFound, AgentRuntimeConflict) as exc:
        _raise_error(exc)


@router.get("", response_model=list[ConversationView])
async def list_conversations(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[ConversationView]:
    domain_id, actor_id, _, _ = _identity(request)
    return await _service(request).list(
        app_id=request.app.state.platform_app_id,
        domain_id=domain_id,
        actor_id=actor_id,
        limit=limit,
    )


@router.get("/{conversation_id}", response_model=ConversationView)
async def get_conversation(
    conversation_id: UUID,
    request: Request,
) -> ConversationView:
    domain_id, actor_id, _, _ = _identity(request)
    try:
        return await _service(request).get(
            conversation_id=conversation_id,
            app_id=request.app.state.platform_app_id,
            domain_id=domain_id,
            actor_id=actor_id,
        )
    except ConversationNotFound as exc:
        _raise_error(exc)


@router.patch("/{conversation_id}", response_model=ConversationView)
async def update_conversation(
    conversation_id: UUID,
    payload: UpdateConversationRequest,
    request: Request,
) -> ConversationView:
    domain_id, actor_id, _, _ = _identity(request)
    try:
        return await _service(request).update(
            conversation_id=conversation_id,
            app_id=request.app.state.platform_app_id,
            domain_id=domain_id,
            actor_id=actor_id,
            expected_row_version=payload.expected_row_version,
            title=payload.title,
            status=payload.status,
            retention_policy=payload.retention_policy,
        )
    except (ConversationNotFound, AgentRuntimeConflict) as exc:
        _raise_error(exc)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: UUID,
    request: Request,
    expected_row_version: int = Query(ge=1),
) -> Response:
    domain_id, actor_id, _, _ = _identity(request)
    try:
        await _service(request).delete(
            conversation_id=conversation_id,
            app_id=request.app.state.platform_app_id,
            domain_id=domain_id,
            actor_id=actor_id,
            expected_row_version=expected_row_version,
        )
    except (ConversationNotFound, AgentRuntimeConflict) as exc:
        _raise_error(exc)
    return Response(status_code=204)


@router.post(
    "/{conversation_id}/turns",
    status_code=202,
    response_model=ConversationTurnReceipt,
)
async def create_turn(
    conversation_id: UUID,
    payload: CreateConversationTurnRequest,
    request: Request,
    idempotency_key: str = Header(alias="Idempotency-Key"),
) -> ConversationTurnReceipt:
    domain_id, actor_id, request_id, trace_id = _identity(request)
    try:
        return await _service(request).create_turn(
            conversation_id=conversation_id,
            app_id=request.app.state.platform_app_id,
            domain_id=domain_id,
            actor_id=actor_id,
            request_id=request_id,
            trace_id=trace_id,
            auth_context=request.state.auth_context.model_dump(mode="json"),
            idempotency_key=idempotency_key,
            raw_input=payload.input,
            expected_conversation_version=(
                payload.expected_conversation_version
            ),
            collection_ids=payload.collection_ids,
            security_level=payload.security_level,
            client_metadata=payload.client_metadata,
            query_images=payload.images,
            budget=request.app.state.agent_runtime_budget,
        )
    except (
        AgentDefinitionNotFound,
        AgentRuntimeConflict,
        ConversationNotFound,
        ConversationTurnNotFound,
    ) as exc:
        _raise_error(exc)


@router.get(
    "/{conversation_id}/turns",
    response_model=ConversationTurnPage,
)
async def list_turns(
    conversation_id: UUID,
    request: Request,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> ConversationTurnPage:
    domain_id, actor_id, _, _ = _identity(request)
    try:
        return await _service(request).list_turns(
            conversation_id=conversation_id,
            app_id=request.app.state.platform_app_id,
            domain_id=domain_id,
            actor_id=actor_id,
            after_sequence=after,
            limit=limit,
        )
    except ConversationNotFound as exc:
        _raise_error(exc)


@router.get(
    "/{conversation_id}/turns/{turn_id}/trace",
    response_model=list[PublicTraceEvent],
)
async def list_turn_trace(
    conversation_id: UUID,
    turn_id: UUID,
    request: Request,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[PublicTraceEvent]:
    domain_id, actor_id, _, _ = _identity(request)
    try:
        return await _service(request).list_trace(
            conversation_id=conversation_id,
            turn_id=turn_id,
            app_id=request.app.state.platform_app_id,
            domain_id=domain_id,
            actor_id=actor_id,
            after_sequence=after,
            limit=limit,
        )
    except ConversationTurnNotFound as exc:
        _raise_error(exc)


@memory_router.get("", response_model=list[MemoryItemView])
async def list_memories(
    request: Request,
    agent_id: UUID = Query(),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[MemoryItemView]:
    domain_id, actor_id, _, _ = _identity(request)
    return await _service(request).list_memories(
        app_id=request.app.state.platform_app_id,
        domain_id=domain_id,
        actor_id=actor_id,
        agent_id=agent_id,
        limit=limit,
    )


@memory_router.delete("/{memory_id}", status_code=204)
async def forget_memory(
    memory_id: UUID,
    request: Request,
) -> Response:
    domain_id, actor_id, _, _ = _identity(request)
    try:
        await _service(request).forget_memory(
            memory_id=memory_id,
            app_id=request.app.state.platform_app_id,
            domain_id=domain_id,
            actor_id=actor_id,
        )
    except ConversationNotFound as exc:
        _raise_error(exc)
    return Response(status_code=204)
