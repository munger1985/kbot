"""按 Domain 与稳定 Actor 标识隔离的站内通知 API。"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from main_api.application import NotificationCenterError, NotificationCenterService
from platform_core.contracts import PUBLIC_API_V1


router = APIRouter(prefix=PUBLIC_API_V1, tags=["Notifications"])


class NotificationReadUpdate(BaseModel):
    read: bool
    expected_row_version: int = Field(ge=1)


class NotificationReadMany(BaseModel):
    notification_ids: list[UUID] = Field(min_length=1, max_length=100)


class NotificationPreferenceUpdate(BaseModel):
    event_type: str = Field(min_length=3, max_length=160)
    enabled: bool


class OperationWatchUpdate(BaseModel):
    notify_terminal: bool = True


def _service(request: Request) -> NotificationCenterService:
    return request.app.state.notification_center_service


def _scope(request: Request) -> tuple[int, str]:
    context = request.state.auth_context
    actor_id = str(context.asserted_user_id or "").strip()
    if not actor_id or context.domain_id is None:
        raise HTTPException(
            403,
            {"code": "NOTIFICATION_CONTEXT_REQUIRED", "message": "通知需要 Domain 和 Actor 上下文"},
        )
    try:
        return int(context.domain_id), actor_id
    except ValueError as exc:
        raise HTTPException(
            422,
            {"code": "NOTIFICATION_DOMAIN_INVALID", "message": "Domain 标识格式无效"},
        ) from exc


def _raise(exc: NotificationCenterError) -> None:
    raise HTTPException(
        exc.status_code, {"code": exc.code, "message": exc.message}
    ) from exc


def _cursor(sequence: int) -> str:
    return base64.urlsafe_b64encode(f"v1:{sequence}".encode()).decode().rstrip("=")


def _decode_cursor(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        padded = value + "=" * (-len(value) % 4)
        version, sequence = base64.urlsafe_b64decode(padded).decode().split(":", 1)
        if version != "v1" or int(sequence) < 1:
            raise ValueError
        return int(sequence)
    except Exception as exc:
        raise HTTPException(
            422, {"code": "NOTIFICATION_CURSOR_INVALID", "message": "cursor 无效"}
        ) from exc


@router.get("/notifications/summary")
async def notification_summary(request: Request):
    domain_id, actor_id = _scope(request)
    return await _service(request).summary(domain_id=domain_id, actor_id=actor_id)


@router.get("/notifications/events")
async def notification_events(
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
):
    domain_id, actor_id = _scope(request)
    try:
        after_sequence = int(last_event_id or 0)
        if after_sequence < 0:
            raise ValueError
    except ValueError as exc:
        raise HTTPException(
            422, {"code": "NOTIFICATION_EVENT_ID_INVALID", "message": "Last-Event-ID 无效"}
        ) from exc
    settings = request.app.state.main_api_settings.notifications

    async def stream():
        cursor = after_sequence
        elapsed = 0.0
        while not await request.is_disconnected():
            rows = await _service(request).stream_events(
                domain_id=domain_id, actor_id=actor_id,
                after_sequence=cursor,
            )
            for row in rows:
                cursor = int(row["event_sequence"])
                payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                yield f"id: {cursor}\nevent: notification\ndata: {payload}\n\n"
                elapsed = 0.0
            if not rows and elapsed >= settings.sse_heartbeat_seconds:
                yield ": keep-alive\n\n"
                elapsed = 0.0
            await asyncio.sleep(settings.sse_poll_interval_seconds)
            elapsed += settings.sse_poll_interval_seconds

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/notifications")
async def list_notifications(
    request: Request,
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=30, ge=1, le=100),
):
    domain_id, actor_id = _scope(request)
    rows = await _service(request).list_notifications(
        domain_id=domain_id, actor_id=actor_id,
        limit=limit + 1, before_sequence=_decode_cursor(cursor),
    )
    items = rows[:limit]
    return {
        "items": items,
        "next_cursor": (
            _cursor(int(items[-1]["event_sequence"]))
            if len(rows) > limit and items else None
        ),
    }


@router.post("/notifications/read")
async def read_notifications(body: NotificationReadMany, request: Request):
    domain_id, actor_id = _scope(request)
    return await _service(request).mark_many_read(
        inbox_ids=body.notification_ids, domain_id=domain_id, actor_id=actor_id,
    )


@router.get("/notifications/preferences")
async def notification_preferences(request: Request):
    domain_id, actor_id = _scope(request)
    return {"items": await _service(request).preferences(
        domain_id=domain_id, actor_id=actor_id,
    )}


@router.put("/notifications/preferences")
async def set_notification_preference(
    body: NotificationPreferenceUpdate, request: Request,
):
    domain_id, actor_id = _scope(request)
    try:
        await _service(request).set_preference(
            domain_id=domain_id, actor_id=actor_id,
            event_type=body.event_type, enabled=body.enabled,
        )
    except ValueError as exc:
        raise HTTPException(
            422, {"code": "NOTIFICATION_EVENT_TYPE_INVALID", "message": str(exc)}
        ) from exc
    return {"event_type": body.event_type, "enabled": body.enabled}


@router.delete("/notifications/actor-data")
async def forget_notification_actor(request: Request):
    domain_id, actor_id = _scope(request)
    return await _service(request).forget_actor(
        domain_id=domain_id, actor_id=actor_id,
    )


@router.get("/notifications/quarantine")
async def notification_quarantine(
    request: Request, limit: int = Query(default=50, ge=1, le=100),
):
    domain_id, _ = _scope(request)
    return {"items": await _service(request).quarantine(
        domain_id=domain_id, limit=limit,
    )}


@router.post("/notifications/quarantine/{outbox_id}/retry")
async def retry_notification(outbox_id: UUID, request: Request):
    domain_id, _ = _scope(request)
    try:
        await _service(request).retry_quarantined(
            domain_id=domain_id, outbox_id=outbox_id,
        )
    except NotificationCenterError as exc:
        _raise(exc)
    return {"outbox_id": str(outbox_id), "status": "PENDING"}


@router.get("/notifications/{notification_id}")
async def get_notification(notification_id: UUID, request: Request):
    domain_id, actor_id = _scope(request)
    try:
        return await _service(request).get_notification(
            inbox_id=notification_id, domain_id=domain_id, actor_id=actor_id,
        )
    except NotificationCenterError as exc:
        _raise(exc)


@router.patch("/notifications/{notification_id}")
async def update_notification(
    notification_id: UUID, body: NotificationReadUpdate, request: Request,
):
    domain_id, actor_id = _scope(request)
    try:
        return await _service(request).set_read(
            inbox_id=notification_id, domain_id=domain_id, actor_id=actor_id,
            read=body.read, expected_row_version=body.expected_row_version,
        )
    except NotificationCenterError as exc:
        _raise(exc)


@router.get("/work-items")
async def list_work_items(
    request: Request,
    status: str = Query(default="OPEN", pattern="^(OPEN|COMPLETED|CANCELLED)$"),
    limit: int = Query(default=50, ge=1, le=100),
):
    domain_id, actor_id = _scope(request)
    return {"items": await _service(request).list_work_items(
        domain_id=domain_id, actor_id=actor_id, status=status, limit=limit,
    )}


@router.get("/background-operations")
async def list_background_operations(
    request: Request, limit: int = Query(default=50, ge=1, le=100),
):
    domain_id, actor_id = _scope(request)
    return {"items": await _service(request).list_operations(
        domain_id=domain_id, actor_id=actor_id, limit=limit,
    )}


@router.post("/background-operations/{operation_id}/watch")
async def watch_background_operation(
    operation_id: UUID, body: OperationWatchUpdate, request: Request,
):
    domain_id, actor_id = _scope(request)
    try:
        await _service(request).watch_operation(
            operation_id=operation_id, domain_id=domain_id,
            actor_id=actor_id, notify_terminal=body.notify_terminal,
        )
    except NotificationCenterError as exc:
        _raise(exc)
    return {"operation_id": str(operation_id), "watching": True}


@router.delete("/background-operations/{operation_id}/watch", status_code=204)
async def unwatch_background_operation(operation_id: UUID, request: Request):
    domain_id, actor_id = _scope(request)
    try:
        await _service(request).unwatch_operation(
            operation_id=operation_id, domain_id=domain_id, actor_id=actor_id,
        )
    except NotificationCenterError as exc:
        _raise(exc)
