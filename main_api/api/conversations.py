"""Conversation、Turn、历史轨迹和长期记忆的公开 API。"""

from typing import cast
import base64
import json
from uuid import UUID

from fastapi import (
    APIRouter,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)

from platform_clients import AgentRuntimeClient
from platform_core.contracts import (
    ConversationTurnPage,
    ConversationTurnReceipt,
    ConversationView,
    CreateConversationRequest,
    CreateConversationTurnRequest,
    ConversationQueryImage,
    MemoryItemView,
    PUBLIC_API_V1,
    PublicTraceEvent,
    UpdateConversationRequest,
)


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/conversations",
    tags=["Conversations"],
)
memory_router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/memories",
    tags=["Memories"],
)


def _client(request: Request) -> AgentRuntimeClient:
    return cast(AgentRuntimeClient, request.app.state.agent_runtime_client)


@router.post("", status_code=201, response_model=ConversationView)
async def create_conversation(
    payload: CreateConversationRequest,
    request: Request,
) -> ConversationView:
    result = await _client(request).create_conversation(
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )
    return ConversationView.model_validate(result)


@router.get("", response_model=list[ConversationView])
async def list_conversations(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[ConversationView]:
    rows = await _client(request).list_conversations(
        limit=limit,
        auth_context=request.state.auth_context,
    )
    return [ConversationView.model_validate(row) for row in rows]


@router.get("/{conversation_id}", response_model=ConversationView)
async def get_conversation(
    conversation_id: UUID,
    request: Request,
) -> ConversationView:
    result = await _client(request).get_conversation(
        conversation_id=conversation_id,
        auth_context=request.state.auth_context,
    )
    return ConversationView.model_validate(result)


@router.patch("/{conversation_id}", response_model=ConversationView)
async def update_conversation(
    conversation_id: UUID,
    payload: UpdateConversationRequest,
    request: Request,
) -> ConversationView:
    result = await _client(request).update_conversation(
        conversation_id=conversation_id,
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )
    return ConversationView.model_validate(result)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: UUID,
    request: Request,
    expected_row_version: int = Query(ge=1),
) -> Response:
    await _client(request).delete_conversation(
        conversation_id=conversation_id,
        expected_row_version=expected_row_version,
        auth_context=request.state.auth_context,
    )
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
    result = await _client(request).create_conversation_turn(
        conversation_id=conversation_id,
        payload=payload.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return ConversationTurnReceipt.model_validate(result)


@router.post(
    "/{conversation_id}/turns/multipart",
    status_code=202,
    response_model=ConversationTurnReceipt,
)
async def create_turn_with_images(
    conversation_id: UUID,
    request: Request,
    input: str = Form(min_length=1, max_length=32000),
    expected_conversation_version: int = Form(ge=1),
    collection_ids_json: str = Form(default="[]"),
    security_level: int = Form(default=0, ge=0, le=999),
    client_metadata_json: str = Form(default="{}"),
    images: list[UploadFile] = File(default_factory=list),
    idempotency_key: str = Header(alias="Idempotency-Key"),
) -> ConversationTurnReceipt:
    if not images or len(images) > 8:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_QUERY_IMAGES",
                "message": "查询图片数量必须在 1 到 8 之间",
            },
        )
    try:
        collection_ids = tuple(
            UUID(str(value)) for value in json.loads(collection_ids_json)
        )
        client_metadata = json.loads(client_metadata_json)
        encoded_images = []
        total = 0
        for image in images:
            content = await image.read(16 * 1024 * 1024 + 1)
            total += len(content)
            if (
                len(content) > 16 * 1024 * 1024
                or total > 32 * 1024 * 1024
            ):
                raise ValueError("查询图片大小超过限制")
            encoded_images.append(
                ConversationQueryImage(
                    file_name=image.filename or "query-image",
                    mime_type=image.content_type or "",
                    content_base64=base64.b64encode(content).decode("ascii"),
                )
            )
        payload = CreateConversationTurnRequest(
            input=input,
            expected_conversation_version=expected_conversation_version,
            collection_ids=collection_ids,
            security_level=security_level,
            client_metadata=client_metadata,
            images=tuple(encoded_images),
        )
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_QUERY_IMAGES", "message": str(exc)},
        ) from exc
    result = await _client(request).create_conversation_turn(
        conversation_id=conversation_id,
        payload=payload.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return ConversationTurnReceipt.model_validate(result)


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
    result = await _client(request).list_conversation_turns(
        conversation_id=conversation_id,
        after=after,
        limit=limit,
        auth_context=request.state.auth_context,
    )
    return ConversationTurnPage.model_validate(result)


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
    rows = await _client(request).list_turn_trace(
        conversation_id=conversation_id,
        turn_id=turn_id,
        after=after,
        limit=limit,
        auth_context=request.state.auth_context,
    )
    return [PublicTraceEvent.model_validate(row) for row in rows]


@memory_router.get("", response_model=list[MemoryItemView])
async def list_memories(
    request: Request,
    agent_id: UUID = Query(),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[MemoryItemView]:
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
