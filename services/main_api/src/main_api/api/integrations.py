"""不使用用户 Token 或 App API Key、由 Provider 自身验签的外部集成入口。"""

from __future__ import annotations

import base64
import asyncio
import hashlib
from datetime import UTC, datetime
from uuid import uuid4
from time import monotonic

from fastapi import APIRouter, HTTPException, Request

from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.contracts.aiops import (
    SignalEventEnvelope,
    SignalEventReceipt,
)


router = APIRouter(
    prefix="/api/v1/integrations/aiops/signals",
    tags=["AIOps Signal Integrations"],
)

_SIGNATURE_HEADERS = {
    "x-kbot-signature",
    "x-kbot-timestamp",
    "x-alertmanager-delivery-id",
    "x-zabbix-delivery-id",
}
_CONTENT_TYPES = {"application/json", "application/json; charset=utf-8"}


class _WebhookRateLimiter:
    """进程内第一道防护；多副本总限额仍由 API Gateway 强制。"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._windows: dict[str, tuple[float, int]] = {}

    async def allow(self, key_hash: str, limit: int) -> bool:
        now = monotonic()
        async with self._lock:
            started, count = self._windows.get(
                key_hash, (now, 0)
            )
            if now - started >= 60:
                started, count = now, 0
            if count >= limit:
                return False
            self._windows[key_hash] = (started, count + 1)
            if len(self._windows) > 10000:
                self._windows = {
                    key: value
                    for key, value in self._windows.items()
                    if now - value[0] < 60
                }
            return True


_RATE_LIMITER = _WebhookRateLimiter()


@router.post(
    "/{webhook_key}",
    response_model=SignalEventReceipt,
    status_code=202,
)
async def receive_signal_event(
    webhook_key: str, request: Request
) -> SignalEventReceipt:
    settings = request.app.state.main_api_settings
    content_type = request.headers.get("content-type", "").lower()
    if content_type not in _CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail={
                "code": "MONITOR_CONTENT_TYPE_INVALID",
                "message": "监控 Webhook 仅接受 application/json",
            },
        )
    if len(webhook_key) < 32 or len(webhook_key) > 256:
        raise HTTPException(status_code=404, detail="Webhook 路由不存在")
    webhook_key_hash = hashlib.sha256(
        webhook_key.encode("utf-8")
    ).hexdigest()
    if not await _RATE_LIMITER.allow(
        webhook_key_hash,
        settings.integrations.monitoring_requests_per_minute,
    ):
        raise HTTPException(
            status_code=429,
            detail={
                "code": "MONITOR_RATE_LIMITED",
                "message": "Webhook 请求频率超过限制",
            },
        )
    max_bytes = settings.integrations.monitoring_max_webhook_bytes
    body_buffer = bytearray()
    async for chunk in request.stream():
        body_buffer.extend(chunk)
        if len(body_buffer) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "MONITOR_PAYLOAD_TOO_LARGE",
                    "message": "Webhook 正文超过大小限制",
                },
            )
    body = bytes(body_buffer)
    if not body:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "MONITOR_PAYLOAD_TOO_LARGE",
                "message": "Webhook 正文不能为空",
            },
        )
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    trace_id = request.headers.get("traceparent") or request_id
    context = AuthContext(
        principal_kind=PrincipalKind.SERVICE,
        client_id="monitor-integration",
        calling_service=request.app.state.service_name,
        request_id=request_id,
        trace_id=trace_id,
    )
    envelope = SignalEventEnvelope(
        request_id=request_id,
        webhook_key_hash=webhook_key_hash,
        raw_body_base64=base64.b64encode(body).decode("ascii"),
        raw_body_hash=hashlib.sha256(body).hexdigest(),
        content_type=content_type,
        signature_headers={
            key.lower(): value
            for key, value in request.headers.items()
            if key.lower() in _SIGNATURE_HEADERS
        },
        received_at=datetime.now(UTC),
    )
    payload = await request.app.state.aiops_client.intake_signal_event(
        envelope, auth_context=context
    )
    return SignalEventReceipt(
        receipt_id=payload["inbox_id"],
        accepted=payload["accepted"],
        duplicate=payload["duplicate"],
        event_count=len(payload["signal_event_ids"]),
    )
