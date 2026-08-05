"""跨服务共享的通知 Outbox ORM 映射。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UniversalTimestamp,
    UUIDv7Type,
)


class NotificationOutboxEntity(BaseEntity):
    __tablename__ = "KBOT_NOTIFICATION_OUTBOX"
    __table_args__ = (
        UniqueConstraint(
            "producer_service", "event_key", name="UK_NOTIFY_OUTBOX_EVENT"
        ),
    )

    outbox_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    producer_service: Mapped[str] = mapped_column(String(64), nullable=False)
    event_key: Mapped[str] = mapped_column(String(256), nullable=False)
    event_type: Mapped[str] = mapped_column(String(160), nullable=False)
    event_version: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")
    attempt_count: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False, default=0)
    available_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False, server_default=func.now()
    )
    lease_token: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    last_error_code: Mapped[str | None] = mapped_column(String(128))
    published_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False, server_default=func.now(),
        onupdate=func.now(),
    )
