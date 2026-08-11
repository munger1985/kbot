"""Agent Run 的公开命令、结果和 SSE 事件接口。"""

from __future__ import annotations

import asyncio
import json
from time import monotonic
from typing import AsyncIterator, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from platform_clients import AgentRuntimeClient
from platform_core.contracts import (
    AgentArtifact,
    AgentRunReceipt,
    AgentRunSummary,
    PUBLIC_API_V1,
)
from main_api.application import AccessDeniedError
from platform_clients import KnowledgeRetrievalAppClient
from platform_core.security import get_auth_context


async def _require_use(request: Request) -> None:
    context = get_auth_context(request)
    try:
        await request.app.state.access_control_service.require(
            app_id="knowledge_retrieval",
            domain_id=int(context.domain_id or "0"),
            user_id=context.asserted_user_id or context.client_id,
            permission_code="knowledge_retrieval:use",
        )
    except AccessDeniedError as exc:
        raise HTTPException(
            403,
            {"code": "APP_PERMISSION_DENIED", "permission": "knowledge_retrieval:use"},
        ) from exc


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/apps/knowledge-retrieval/runs",
    tags=["Knowledge Retrieval Runs"],
    dependencies=[Depends(_require_use)],
)

_TERMINAL_EVENTS = {
    "RUN_COMPLETED",
    "RUN_FAILED",
    "RUN_CANCELLED",
    "RUN_EXPIRED",
}


class CancelRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_row_version: int = Field(ge=1)


class KnowledgeRunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_id: UUID
    input: str = Field(min_length=1, max_length=32000)
    collection_ids: tuple[UUID, ...] = ()
    security_level: int = Field(default=0, ge=0, le=999)
    client_metadata: dict = Field(default_factory=dict)


def _client(request: Request) -> AgentRuntimeClient:
    return cast(
        AgentRuntimeClient,
        request.app.state.agent_runtime_client,
    )


async def _authorized_spec(request: Request, agent_id: UUID) -> dict:
    context = get_auth_context(request)
    domain_id = int(context.domain_id or "0")
    actor_id = context.asserted_user_id or context.client_id
    snapshot = await request.app.state.access_control_service.snapshot(
        app_id="knowledge_retrieval", domain_id=domain_id, user_id=actor_id
    )
    client: KnowledgeRetrievalAppClient = (
        request.app.state.knowledge_retrieval_app_client
    )
    if "knowledge_retrieval:agent_manage" not in snapshot.permissions:
        await client.authorize(
            payload={
                "domain_id": domain_id,
                "agent_id": str(agent_id),
                "user_id": actor_id,
                "role_codes": list(snapshot.roles),
            },
            auth_context=context,
        )
    return await client.execution_spec(
        agent_id=agent_id, domain_id=domain_id, auth_context=context
    )


@router.post("", status_code=202, response_model=AgentRunReceipt)
async def create_run(
    payload: KnowledgeRunCreateRequest,
    request: Request,
    idempotency_key: str = Header(alias="Idempotency-Key"),
) -> AgentRunReceipt:
    spec = await _authorized_spec(request, payload.agent_id)
    result = await _client(request).create_run(
        payload={
            **payload.model_dump(mode="json"),
            "execution_spec": spec,
        },
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return AgentRunReceipt.model_validate(result)


@router.get("/{run_id}", response_model=AgentRunSummary)
async def get_run(
    run_id: UUID, request: Request
) -> AgentRunSummary:
    result = await _client(request).get_run(
        run_id=run_id,
        auth_context=request.state.auth_context,
    )
    return AgentRunSummary.model_validate(result)


@router.get("/{run_id}/result", response_model=AgentArtifact)
async def get_run_result(
    run_id: UUID, request: Request
) -> AgentArtifact:
    result = await _client(request).get_result(
        run_id=run_id,
        auth_context=request.state.auth_context,
    )
    return AgentArtifact.model_validate(result)


@router.post(
    "/{run_id}/cancel",
    status_code=202,
    response_model=AgentRunReceipt,
)
async def cancel_run(
    run_id: UUID,
    payload: CancelRunRequest,
    request: Request,
    idempotency_key: str = Header(alias="Idempotency-Key"),
) -> AgentRunReceipt:
    result = await _client(request).cancel_run(
        run_id=run_id,
        expected_row_version=payload.expected_row_version,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return AgentRunReceipt.model_validate(result)


@router.get("/{run_id}/events")
async def stream_run_events(
    run_id: UUID,
    request: Request,
    last_event_id: str | None = Header(
        default=None, alias="Last-Event-ID"
    ),
) -> StreamingResponse:
    cursor = _parse_cursor(last_event_id)
    summary = await _client(request).get_run(
        run_id=run_id,
        auth_context=request.state.auth_context,
    )
    latest = int(summary["event_cursor"])
    if cursor > latest:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "AGENT_EVENT_CURSOR_INVALID",
                "message": "Last-Event-ID 超过当前 Run 事件游标",
            },
        )
    return StreamingResponse(
        _event_stream(
            run_id=run_id,
            request=request,
            cursor=cursor,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _parse_cursor(value: str | None) -> int:
    if value is None or not value.strip():
        return 0
    try:
        parsed = int(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_EVENT_CURSOR",
                "message": "Last-Event-ID 必须是非负整数",
            },
        ) from exc
    if parsed < 0:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_EVENT_CURSOR",
                "message": "Last-Event-ID 不能为负数",
            },
        )
    return parsed


async def _event_stream(
    *,
    run_id: UUID,
    request: Request,
    cursor: int,
) -> AsyncIterator[str]:
    settings = request.app.state.main_api_settings
    poll_interval = settings.api.sse_poll_interval_seconds
    heartbeat_seconds = settings.api.sse_heartbeat_seconds
    batch_size = settings.api.sse_batch_size
    auth_context = request.state.auth_context
    last_output_at = monotonic()
    while not await request.is_disconnected():
        events = await _client(request).list_events(
            run_id=run_id,
            after_sequence=cursor,
            limit=batch_size,
            auth_context=auth_context,
        )
        for event in events:
            cursor = int(event["sequence_no"])
            yield _format_sse(event)
            last_output_at = monotonic()
            if event["event_type"] in _TERMINAL_EVENTS:
                yield (
                    "event: done\n"
                    f"data: {json.dumps({'sequence_no': cursor})}\n\n"
                )
                return
        if monotonic() - last_output_at >= heartbeat_seconds:
            yield ": heartbeat\n\n"
            last_output_at = monotonic()
        await asyncio.sleep(poll_interval)


def _format_sse(event: dict) -> str:
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return (
        f"id: {event['sequence_no']}\n"
        f"event: {event['event_type']}\n"
        f"data: {payload}\n\n"
    )
