"""Main API 自有的组合编排恢复 Receipt。"""

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


class CompositionReceiptEntity(BaseEntity):
    __tablename__ = "KBOT_COMPOSITION_RECEIPT"
    __table_args__ = (
        UniqueConstraint(
            "domain_id", "actor_id", "operation", "idempotency_key",
            name="UK_COMPOSITION_IDEMPOTENCY",
        ),
    )

    receipt_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    operation: Mapped[str] = mapped_column(String(80), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(80), nullable=False)
    resource_id: Mapped[str | None] = mapped_column(String(256))
    resource_version: Mapped[str | None] = mapped_column(String(128))
    verification_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False, default=dict
    )
    error_code: Mapped[str | None] = mapped_column(String(128))
    attempt_count: Mapped[int] = mapped_column(Numeric(10, 0), default=0)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(),
        onupdate=func.now(),
    )


__all__ = ["CompositionReceiptEntity"]
