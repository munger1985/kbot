"""Slack Events API 公开入口与 KM Asset 业务边界适配。"""

from __future__ import annotations

import asyncio
import base64
import hashlib
from datetime import UTC, datetime
from time import monotonic
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict

from platform_core.contracts import (
    AuthContext,
    PrincipalKind,
    SlackWebhookEnvelope,
    SlackWebhookReceipt,
)


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
_SLACK_SIGNATURE_HEADERS = frozenset(
    {
        "x-slack-request-timestamp",
        "x-slack-signature",
        "x-slack-retry-num",
        "x-slack-retry-reason",
    }
)


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
    config = request.app.state.main_api_settings.integrations
    source = request.client.host if request.client else "unknown"
    if not await _RATE_LIMITER.allow(
        source,
        config.slack_public_requests_per_minute,
    ):
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
        if len(body_buffer) > config.slack_public_max_webhook_bytes:
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
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    trace_id = request.headers.get("traceparent") or request_id
    envelope = SlackWebhookEnvelope(
        request_id=request_id,
        raw_body_base64=base64.b64encode(body).decode("ascii"),
        raw_body_hash=hashlib.sha256(body).hexdigest(),
        content_type=content_type,
        signature_headers={
            key.lower(): value
            for key, value in request.headers.items()
            if key.lower() in _SLACK_SIGNATURE_HEADERS
        },
        received_at=datetime.now(UTC),
    )
    context = AuthContext(
        principal_kind=PrincipalKind.SERVICE,
        client_id="slack-integration",
        calling_service=request.app.state.service_name,
        request_id=request_id,
        trace_id=trace_id,
    )
    payload = await request.app.state.km_asset_client.intake_slack_event(
        envelope=envelope,
        auth_context=context,
    )
    result = SlackWebhookReceipt.model_validate(payload)
    if result.challenge is not None:
        return Response(
            result.challenge,
            media_type="text/plain",
            status_code=200,
        )
    return SlackEventReceipt(
        receipt_id=result.receipt_id,
        accepted=result.accepted,
        duplicate=result.duplicate,
        ignored=result.ignored,
    )
