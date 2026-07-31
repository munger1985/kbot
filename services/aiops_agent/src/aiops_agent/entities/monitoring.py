"""AIOps 监控源、外部事件和告警的 SQLAlchemy 映射。"""

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


class MonitorSourceEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_MONITOR_SOURCE"

    monitor_source_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    endpoint: Mapped[str | None] = mapped_column(String(2048))
    secret_ref: Mapped[str | None] = mapped_column(String(1024))
    webhook_secret_ref: Mapped[str | None] = mapped_column(String(1024))
    tls_profile_ref: Mapped[str | None] = mapped_column(String(1024))
    webhook_key_hash: Mapped[str | None] = mapped_column(String(64))
    previous_webhook_key_hash: Mapped[str | None] = mapped_column(String(64))
    previous_webhook_key_expires_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    capabilities_json: Mapped[dict[str, Any] | None] = mapped_column(
        OracleNativeJSON
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="DISABLED"
    )
    health_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="UNKNOWN"
    )
    health_check_request_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    health_check_requested_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    last_health_check_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    last_error_code: Mapped[str | None] = mapped_column(String(128))
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    health_version: Mapped[int] = mapped_column(
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


class TargetMonitorEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_TARGET_MONITOR"

    target_monitor_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    monitor_source_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), nullable=False
    )
    external_target_key: Mapped[str] = mapped_column(
        String(256), nullable=False
    )
    role: Mapped[str] = mapped_column(
        String(16), nullable=False, default="PRIMARY"
    )
    priority: Mapped[int] = mapped_column(
        Numeric(8, 0), nullable=False, default=100
    )
    metric_scope_json: Mapped[dict[str, Any] | None] = mapped_column(
        OracleNativeJSON
    )
    mapping_overrides_json: Mapped[dict[str, Any] | None] = mapped_column(
        OracleNativeJSON
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="ACTIVE"
    )
    health_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="UNKNOWN"
    )
    last_health_check_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    last_error_code: Mapped[str | None] = mapped_column(String(128))
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    health_version: Mapped[int] = mapped_column(
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


class OpsEventEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_EVENT"

    event_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    monitor_source_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), nullable=False
    )
    alert_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    source_inbox_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    source_event_key: Mapped[str] = mapped_column(String(256), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    event_status: Mapped[str] = mapped_column(String(16), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    received_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    payload_uri: Mapped[str | None] = mapped_column(String(2048))
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    normalizer_version: Mapped[str] = mapped_column(String(64), nullable=False)
    processing_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="RECEIVED"
    )
    trace_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class OpsAlertEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_ALERT"

    alert_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    summary: Mapped[str] = mapped_column(String(1000), nullable=False)
    correlation_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    first_seen_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    event_count: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
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
