"""AIOps Target、策略和 Agent 绑定的 SQLAlchemy 映射。"""

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


class TargetEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_TARGET"

    target_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    db_type: Mapped[str] = mapped_column(String(32), nullable=False)
    version_code: Mapped[str | None] = mapped_column(String(64))
    environment: Mapped[str] = mapped_column(String(16), nullable=False)
    db_role: Mapped[str] = mapped_column(
        String(16), nullable=False, default="UNKNOWN"
    )
    endpoint_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    readonly_connection_enabled: Mapped[bool] = mapped_column(
        Numeric(1, 0), nullable=False, default=False
    )
    controlled_change_enabled: Mapped[bool] = mapped_column(
        Numeric(1, 0), nullable=False, default=False
    )
    diagnostic_credential_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    execution_credential_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    security_level: Mapped[int] = mapped_column(
        Numeric(3, 0), nullable=False, default=1
    )
    capabilities_json: Mapped[dict[str, Any] | None] = mapped_column(
        OracleNativeJSON
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="DISABLED"
    )
    connectivity_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="UNKNOWN"
    )
    observed_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="UNKNOWN"
    )
    last_observed_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    connectivity_check_request_id: Mapped[UUID | None] = mapped_column(
        UUIDv7Type()
    )
    connectivity_check_requested_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    last_connectivity_check_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    last_connectivity_success_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    last_error_code: Mapped[str | None] = mapped_column(String(128))
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    connectivity_version: Mapped[int] = mapped_column(
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


class PolicyEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_POLICY"

    policy_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    policy_key: Mapped[str] = mapped_column(String(128), nullable=False)
    version_no: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    rules_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    policy_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    effective_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    retired_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
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


class TargetBindingEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_TARGET_BINDING"

    binding_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    agent_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    allow_mutation: Mapped[bool] = mapped_column(
        Numeric(1, 0), nullable=False, default=False
    )
    policy_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    allowed_actions_json: Mapped[list[str] | None] = mapped_column(OracleNativeJSON)
    change_window_json: Mapped[dict[str, Any] | None] = mapped_column(
        OracleNativeJSON
    )
    max_daily_executions: Mapped[int | None] = mapped_column(Numeric(10, 0))
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="ACTIVE"
    )
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
