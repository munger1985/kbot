"""通知 Outbox 的安全事件信封。"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .catalog import event_definition


_FORBIDDEN_KEYS = {
    "authorization", "cookie", "api_key", "password", "secret", "credential",
    "sql", "query_result", "file_content", "prompt", "messages", "raw_log",
}


class NotificationEnvelope(BaseModel):
    """只允许资源标识与安全摘要进入平台 Outbox。"""

    model_config = ConfigDict(extra="forbid")

    domain_id: int = Field(gt=0)
    event_type: str = Field(min_length=3, max_length=160)
    event_version: int = Field(default=1, ge=1, le=100)
    resource_type: str = Field(min_length=1, max_length=80)
    resource_id: str = Field(min_length=1, max_length=256)
    resource_name: str | None = Field(default=None, max_length=300)
    initiator_actor_id: str | None = Field(default=None, max_length=256)
    recipient_actor_ids: list[str] = Field(default_factory=list, max_length=50)
    summary: str = Field(min_length=1, max_length=1000)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str = Field(min_length=1, max_length=256)
    operation_id: str | None = Field(default=None, max_length=256)
    safe_data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("recipient_actor_ids")
    @classmethod
    def _recipients(cls, value: list[str]) -> list[str]:
        normalized = sorted({item.strip() for item in value if item.strip()})
        if any(len(item) > 256 for item in normalized):
            raise ValueError("recipient_actor_ids 包含过长标识")
        return normalized

    @model_validator(mode="after")
    def _safe(self) -> "NotificationEnvelope":
        event_definition(self.event_type)
        stack: list[Any] = [self.safe_data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, child in item.items():
                    if str(key).casefold() in _FORBIDDEN_KEYS:
                        raise ValueError("NOTIFICATION_PAYLOAD_SENSITIVE")
                    stack.append(child)
            elif isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, str) and len(item) > 2000:
                raise ValueError("NOTIFICATION_PAYLOAD_TOO_LARGE")
        return self
