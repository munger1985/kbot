"""Main API 拥有的通知、待办与后台任务投影实体。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Identity, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    UniversalTimestamp,
    UUIDv7Type,
)


class BackgroundOperationEntity(BaseEntity):
    __tablename__ = "KBOT_BACKGROUND_OPERATION"
    __table_args__ = (
        UniqueConstraint(
            "producer_service", "source_operation_id",
            name="UK_NOTIFY_OPERATION_SOURCE",
        ),
    )

    operation_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    producer_service: Mapped[str] = mapped_column(String(64), nullable=False)
    source_operation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    initiator_actor_id: Mapped[str | None] = mapped_column(String(256))
    resource_type: Mapped[str] = mapped_column(String(80), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(256), nullable=False)
    resource_name: Mapped[str | None] = mapped_column(String(300))
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    progress_current: Mapped[int | None] = mapped_column(Numeric(19, 0))
    progress_total: Mapped[int | None] = mapped_column(Numeric(19, 0))
    error_code: Mapped[str | None] = mapped_column(String(128))
    summary: Mapped[str] = mapped_column(String(1000), nullable=False)
    last_outbox_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    last_occurred_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class NotificationInboxEntity(BaseEntity):
    __tablename__ = "KBOT_NOTIFICATION_INBOX"
    __table_args__ = (
        UniqueConstraint("outbox_id", "recipient_actor_id", name="UK_NOTIFY_INBOX_RECIP"),
        UniqueConstraint("event_sequence", name="UK_NOTIFY_INBOX_SEQ"),
    )

    inbox_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    event_sequence: Mapped[int] = mapped_column(
        Numeric(19, 0), Identity(), nullable=False
    )
    outbox_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    recipient_actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    event_type: Mapped[str] = mapped_column(String(160), nullable=False)
    category: Mapped[str] = mapped_column(String(24), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    summary: Mapped[str] = mapped_column(String(1000), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(80), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(256), nullable=False)
    operation_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    read_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now()
    )


class NotificationPreferenceEntity(BaseEntity):
    __tablename__ = "KBOT_NOTIFICATION_PREF"

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True)
    actor_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(160), primary_key=True)
    enabled: Mapped[int] = mapped_column(Numeric(1, 0), default=1)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class WorkItemEntity(BaseEntity):
    __tablename__ = "KBOT_WORK_ITEM"

    work_item_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(80), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(256), nullable=False)
    action_type: Mapped[str] = mapped_column(String(160), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    summary: Mapped[str] = mapped_column(String(1000), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="OPEN")
    opened_outbox_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    resolved_outbox_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    last_occurred_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class OperationWatchEntity(BaseEntity):
    __tablename__ = "KBOT_OPERATION_WATCH"

    operation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True)
    actor_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    notify_terminal: Mapped[int] = mapped_column(Numeric(1, 0), default=1)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now()
    )


__all__ = [
    "BackgroundOperationEntity",
    "NotificationInboxEntity",
    "NotificationPreferenceEntity",
    "OperationWatchEntity",
    "WorkItemEntity",
]
