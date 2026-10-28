"""控制面持久化的只读取证配额及执行租约。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.persistence.orm import BaseEntity, UniversalTimestamp, UUIDv7Type


class ResourcePoolEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_RESOURCE_POOL"

    pool_key: Mapped[str] = mapped_column(String(32), primary_key=True)
    global_limit: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False)
    target_limit: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False)
    source_limit: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False)
    cleanup_grace_seconds: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False)


class ResourceLeaseEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_RESOURCE_LEASE"

    lease_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True)
    pool_key: Mapped[str] = mapped_column(String(32), nullable=False)
    run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    source_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    release_token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    acquired_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), nullable=False)
    execution_deadline_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), nullable=False)
    held_until: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), nullable=False)
    released_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))

    __table_args__ = (
        Index("ix_ops_resource_lease_active", "pool_key", "held_until"),
        Index("ix_ops_resource_lease_target", "target_id", "held_until"),
        Index("ix_ops_resource_lease_source", "source_kind", "source_id", "held_until"),
    )
