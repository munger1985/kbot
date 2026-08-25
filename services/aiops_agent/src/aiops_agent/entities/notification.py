"""AIOps Target 主动分享订阅。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Index, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UniversalTimestamp,
    UUIDv7Type,
)


class NotificationSubscriptionEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_NOTIFICATION_SUBSCRIPTION"
    __table_args__ = (
        UniqueConstraint(
            "domain_id",
            "target_id",
            "recipient_actor_id",
            "channel",
            name="UK_OPS_NOTIFY_SUB_RECIPIENT",
        ),
        Index(
            "IX_OPS_NOTIFY_SUB_ROUTE",
            "target_id",
            "status",
            "minimum_severity",
        ),
    )

    subscription_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    recipient_actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    channel: Mapped[str] = mapped_column(
        String(16), nullable=False, default="IN_APP"
    )
    minimum_severity: Mapped[str] = mapped_column(
        String(16), nullable=False, default="HIGH"
    )
    stages_json: Mapped[list[str]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="ACTIVE"
    )
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    __mapper_args__ = {"version_id_col": row_version}
