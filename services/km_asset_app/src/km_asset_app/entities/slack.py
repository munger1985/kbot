"""KM Asset App 拥有的 Slack 集成持久化实体。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Integer, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UUIDv7Type,
)


class SlackInboxEntity(BaseEntity):
    __tablename__ = "KBOT_KM_SLACK_INBOX"

    inbox_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    message_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    channel_id: Mapped[str] = mapped_column(String(64), nullable=False)
    slack_user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    event_ts: Mapped[str] = mapped_column(String(32), nullable=False)
    root_thread_ts: Mapped[str] = mapped_column(String(32), nullable=False)
    message_text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_body_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_payload_json: Mapped[dict] = mapped_column(
        OracleNativeJSON(), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="RECEIVED"
    )
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    conversation_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    turn_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    run_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    callback_sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(String(2000))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SlackThreadEntity(BaseEntity):
    __tablename__ = "KBOT_KM_SLACK_THREAD"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "channel_id",
            "root_thread_ts",
            "slack_user_id",
            name="UK_KM_SLACK_THREAD_SCOPE",
        ),
    )

    thread_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    channel_id: Mapped[str] = mapped_column(String(64), nullable=False)
    root_thread_ts: Mapped[str] = mapped_column(String(32), nullable=False)
    slack_user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    agent_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    conversation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SlackDeliveryEntity(BaseEntity):
    __tablename__ = "KBOT_KM_SLACK_DELIVERY"
    __table_args__ = (
        UniqueConstraint(
            "inbox_id", "delivery_type", name="UK_KM_SLACK_DELIVERY_KIND"
        ),
    )

    delivery_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True)
    inbox_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    channel_id: Mapped[str] = mapped_column(String(64), nullable=False)
    slack_user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    thread_ts: Mapped[str] = mapped_column(String(32), nullable=False)
    delivery_type: Mapped[str] = mapped_column(String(16), nullable=False)
    payload_json: Mapped[dict] = mapped_column(OracleNativeJSON(), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="PENDING"
    )
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    slack_message_ts: Mapped[str | None] = mapped_column(String(32))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(2000))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


__all__ = ["SlackDeliveryEntity", "SlackInboxEntity", "SlackThreadEntity"]
