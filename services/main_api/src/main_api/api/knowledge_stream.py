"""知识 App 的 SSE 和证据下载展示辅助，不包含旧 Run/Collection 业务路由。"""

import asyncio
import json
from time import monotonic
from typing import AsyncIterator, Literal
from uuid import UUID
from fastapi import HTTPException, Request

_TERMINAL_EVENTS = {"RUN_COMPLETED", "RUN_FAILED", "RUN_CANCELLED", "RUN_EXPIRED"}

def _client(request):
    return request.app.state.agent_runtime_client

def _preview_type(
    mime_type: str,
) -> Literal["PDF", "IMAGE", "TEXT", "DOWNLOAD"]:
    if mime_type == "application/pdf":
        return "PDF"
    if mime_type in {"image/gif", "image/jpeg", "image/png", "image/webp"}:
        return "IMAGE"
    if mime_type == "text/plain":
        return "TEXT"
    return "DOWNLOAD"


def _reference_not_found() -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={
            "code": "DOCUMENT_REFERENCE_NOT_FOUND",
            "message": "引用不存在或当前用户无权访问",
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
