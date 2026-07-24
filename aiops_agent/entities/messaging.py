"""AIOps Inbox/Outbox 可靠消息表的 SQLAlchemy 映射。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UniversalTimestamp,
    UUIDv7Type,
)


class InboxEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_INBOX"

    inbox_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    source_system: Mapped[str] = mapped_column(String(64), nullable=False)
    message_key: Mapped[str] = mapped_column(String(256), nullable=False)
    message_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    payload_uri: Mapped[str | None] = mapped_column(String(2048))
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="RECEIVED"
    )
    received_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(2000))
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    __mapper_args__ = {"version_id_col": row_version}


class OutboxEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_OUTBOX"

    outbox_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    aggregate_type: Mapped[str] = mapped_column(String(64), nullable=False)
    aggregate_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(256), nullable=False)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    payload_uri: Mapped[str | None] = mapped_column(String(2048))
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="PENDING"
    )
    available_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
    attempt_count: Mapped[int] = mapped_column(
        Numeric(8, 0), nullable=False, default=0
    )
    max_attempts: Mapped[int] = mapped_column(
        Numeric(8, 0), nullable=False, default=3
    )
    lease_owner: Mapped[str | None] = mapped_column(String(256))
    lease_token: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    lease_until: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    published_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(2000))
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
