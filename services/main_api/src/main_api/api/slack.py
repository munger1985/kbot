"""由 Slack 自身验签的 Events API 公开入口。"""

from __future__ import annotations

import asyncio
from time import monotonic
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict

from main_api.application import SlackIntakeService, SlackWebhookError


router = APIRouter(
    prefix="/api/v1/integrations/slack",
    tags=["Slack Integrations"],
)


class SlackEventReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    receipt_id: UUID | None
    accepted: bool
    duplicate: bool
    ignored: bool = False


class _RateLimiter:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._windows: dict[str, tuple[float, int]] = {}

    async def allow(self, key: str, limit: int) -> bool:
        now = monotonic()
        async with self._lock:
            started, count = self._windows.get(key, (now, 0))
            if now - started >= 60:
                started, count = now, 0
            if count >= limit:
                return False
            self._windows[key] = (started, count + 1)
            return True


_RATE_LIMITER = _RateLimiter()


@router.post(
    "/events",
    status_code=202,
    response_model=SlackEventReceipt,
    responses={
        200: {
            "description": "Slack URL verification challenge",
            "content": {"text/plain": {"schema": {"type": "string"}}},
        }
    },
)
async def receive_slack_event(request: Request):
    config = request.app.state.main_api_settings.integrations.slack
    source = request.client.host if request.client else "unknown"
    if not await _RATE_LIMITER.allow(source, config.requests_per_minute):
        raise HTTPException(
            status_code=429,
            detail={"code": "SLACK_RATE_LIMITED", "message": "Slack 请求频率超过限制"},
        )
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type != "application/json":
        raise HTTPException(
            status_code=415,
            detail={
                "code": "SLACK_CONTENT_TYPE_INVALID",
                "message": "Slack Events API 仅接受 application/json",
            },
        )
    body_buffer = bytearray()
    async for chunk in request.stream():
        body_buffer.extend(chunk)
        if len(body_buffer) > config.max_webhook_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "SLACK_PAYLOAD_TOO_LARGE",
                    "message": "Slack 请求正文超过限制",
                },
            )
    body = bytes(body_buffer)
    if not body:
        raise HTTPException(
            status_code=413,
            detail={"code": "SLACK_PAYLOAD_TOO_LARGE", "message": "Slack 请求正文不能为空"},
        )
    service: SlackIntakeService = request.app.state.slack_intake_service
    try:
        result = await service.receive(
            raw_body=body,
            headers={key.lower(): value for key, value in request.headers.items()},
        )
    except SlackWebhookError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    if result.challenge is not None:
        return Response(result.challenge, media_type="text/plain", status_code=200)
    return SlackEventReceipt(
        receipt_id=result.receipt_id,
        accepted=result.accepted,
        duplicate=result.duplicate,
        ignored=result.ignored,
    )
