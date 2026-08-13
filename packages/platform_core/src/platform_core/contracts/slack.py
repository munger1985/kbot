"""Slack 公共入口与 KM Asset 内部接入之间的 Wire 契约。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SlackWebhookEnvelope(BaseModel):
    """保真传递 Slack 原始正文和验签 Header。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "slack.webhook.internal.v1"
    request_id: str = Field(min_length=1, max_length=128)
    raw_body_base64: str = Field(min_length=1, max_length=30_000_000)
    raw_body_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_type: str = Field(min_length=1, max_length=256)
    signature_headers: dict[str, str] = Field(default_factory=dict)
    received_at: datetime


class SlackWebhookReceipt(BaseModel):
    """KM Asset 完成验签和 Inbox 落库后的内部回执。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "slack.webhook.internal.v1"
    receipt_id: UUID | None
    accepted: bool
    duplicate: bool
    ignored: bool = False
    challenge: str | None = None


__all__ = ["SlackWebhookEnvelope", "SlackWebhookReceipt"]
