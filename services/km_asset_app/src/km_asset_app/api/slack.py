"""Main API 调用的 Slack 内部接入接口。"""

from __future__ import annotations

import base64
import binascii
import hashlib

from fastapi import APIRouter, HTTPException, Request

from km_asset_app.application import SlackIntakeService, SlackWebhookError
from platform_core.contracts import SlackWebhookEnvelope, SlackWebhookReceipt


router = APIRouter(
    prefix="/internal/v1/km-asset/integrations/slack",
    tags=["KM Asset Slack Integration"],
)


@router.post("/events", response_model=SlackWebhookReceipt)
async def intake_slack_event(
    envelope: SlackWebhookEnvelope,
    request: Request,
) -> SlackWebhookReceipt:
    """校验内部信封后，将原始 Slack 报文交给业务接入服务。"""
    try:
        raw_body = base64.b64decode(
            envelope.raw_body_base64,
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "SLACK_ENVELOPE_INVALID",
                "message": "Slack 内部报文正文编码无效",
            },
        ) from exc
    if hashlib.sha256(raw_body).hexdigest() != envelope.raw_body_hash:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "SLACK_ENVELOPE_HASH_MISMATCH",
                "message": "Slack 内部报文正文摘要不匹配",
            },
        )
    service: SlackIntakeService = request.app.state.slack_intake_service
    try:
        result = await service.receive(
            raw_body=raw_body,
            headers=envelope.signature_headers,
        )
    except SlackWebhookError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    return SlackWebhookReceipt(
        receipt_id=result.receipt_id,
        accepted=result.accepted,
        duplicate=result.duplicate,
        ignored=result.ignored,
        challenge=result.challenge,
    )
