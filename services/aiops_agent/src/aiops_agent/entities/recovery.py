"""Target 恢复目标与恢复演练的 SQLAlchemy 映射。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UniversalTimestamp,
    UUIDv7Type,
)


class RecoveryProfileEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_RECOVERY_PROFILE"

    recovery_profile_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    version_no: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    rpo_seconds: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    rto_seconds: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    required_drill_interval_days: Mapped[int] = mapped_column(
        Numeric(10, 0), nullable=False
    )
    required_assurance_level: Mapped[str] = mapped_column(
        String(32), nullable=False
    )
    rto_clock_basis: Mapped[str] = mapped_column(String(64), nullable=False)
    source_note: Mapped[str | None] = mapped_column(String(1000))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    effective_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    retired_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    confirmed_by: Mapped[str] = mapped_column(String(256), nullable=False)
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
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


class RecoveryDrillEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_RECOVERY_DRILL"

    drill_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    recovery_profile_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    recovery_profile_version: Mapped[int | None] = mapped_column(Numeric(19, 0))
    db_type: Mapped[str] = mapped_column(String(32), nullable=False)
    scenario: Mapped[str] = mapped_column(String(32), nullable=False)
    assurance_level: Mapped[str] = mapped_column(String(32), nullable=False)
    backup_source_type: Mapped[str] = mapped_column(String(48), nullable=False)
    environment: Mapped[str] = mapped_column(String(16), nullable=False)
    result: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    simulated_failure_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    recovered_through_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    service_validated_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    achieved_rpo_seconds: Mapped[int | None] = mapped_column(Numeric(19, 0))
    achieved_rto_seconds: Mapped[int | None] = mapped_column(Numeric(19, 0))
    recovery_marker_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    evidence_json: Mapped[list[dict[str, Any]]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    source_trust_level: Mapped[str] = mapped_column(String(32), nullable=False)
    source_ops_run_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    reviewed_by: Mapped[str | None] = mapped_column(String(256))
    reviewed_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    review_note: Mapped[str | None] = mapped_column(String(2000))
    notes: Mapped[str | None] = mapped_column(Text)
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
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
