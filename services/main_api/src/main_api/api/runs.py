"""Agent Run 的公开命令、结果和 SSE 事件接口。"""

from __future__ import annotations

import asyncio
import json
from time import monotonic
from typing import AsyncIterator, cast
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from platform_clients import AgentRuntimeClient
from platform_core.contracts import (
    AgentArtifact,
    AgentRunReceipt,
    AgentRunSummary,
    CreateAgentRunRequest,
    PUBLIC_API_V1,
)


router = APIRouter(prefix=f"{PUBLIC_API_V1}/runs", tags=["Agent Runs"])

_TERMINAL_EVENTS = {
    "RUN_COMPLETED",
    "RUN_FAILED",
    "RUN_CANCELLED",
    "RUN_EXPIRED",
}


class CancelRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_row_version: int = Field(ge=1)


def _client(request: Request) -> AgentRuntimeClient:
    return cast(
        AgentRuntimeClient,
        request.app.state.agent_runtime_client,
    )


@router.post("", status_code=202, response_model=AgentRunReceipt)
async def create_run(
    payload: CreateAgentRunRequest,
    request: Request,
    idempotency_key: str = Header(alias="Idempotency-Key"),
) -> AgentRunReceipt:
    result = await _client(request).create_run(
        payload=payload.model_dump(mode="json"),
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
