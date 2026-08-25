"""AIOps 诊断源、信号事件和故障情境的 SQLAlchemy 映射。"""

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


class DiagnosticSourceEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_DIAGNOSTIC_SOURCE"

    diagnostic_source_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    adapter_id: Mapped[str] = mapped_column(String(128), nullable=False)
    adapter_version: Mapped[str] = mapped_column(String(64), nullable=False)
    endpoint: Mapped[str | None] = mapped_column(String(2048))
    auth_credential_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    webhook_credential_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    tls_profile_ref: Mapped[str | None] = mapped_column(String(1024))
    webhook_key_hash: Mapped[str | None] = mapped_column(String(64))
    previous_webhook_key_hash: Mapped[str | None] = mapped_column(String(64))
    previous_webhook_key_expires_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    declared_capabilities_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False, default=dict
    )
    discovered_capabilities_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    config_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False, default=dict
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
    last_success_at: Mapped[datetime | None] = mapped_column(
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


class TargetSourceBindingEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_TARGET_SOURCE_BINDING"

    target_source_binding_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    diagnostic_source_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), nullable=False
    )
    source_locator_key: Mapped[str] = mapped_column(String(512), nullable=False)
    source_locator_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    role: Mapped[str] = mapped_column(
        String(16), nullable=False, default="PRIMARY"
    )
    priority: Mapped[int] = mapped_column(
        Numeric(8, 0), nullable=False, default=100
    )
    capability_scope_json: Mapped[dict[str, Any] | None] = mapped_column(
        OracleNativeJSON
    )
    mapping_overrides_json: Mapped[dict[str, Any] | None] = mapped_column(
        OracleNativeJSON
    )
    query_budget_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
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


class SignalEventEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_SIGNAL_EVENT"

    signal_event_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    target_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    diagnostic_source_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), nullable=False
    )
    source_binding_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    source_inbox_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    source_event_key: Mapped[str] = mapped_column(String(512), nullable=False)
    signal_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    event_class: Mapped[str] = mapped_column(String(128), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    normalized_status: Mapped[str] = mapped_column(String(16), nullable=False)
    source_status: Mapped[str | None] = mapped_column(String(128))
    summary: Mapped[str | None] = mapped_column(String(1000))
    occurred_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    received_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    source_updated_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    dedup_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    payload_uri: Mapped[str | None] = mapped_column(String(2048))
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_locator_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    normalizer_version: Mapped[str] = mapped_column(String(64), nullable=False)
    processing_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="RECEIVED"
    )
    trace_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class SituationEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_SITUATION"

    situation_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    situation_type: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    summary: Mapped[str | None] = mapped_column(String(2000))
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    correlation_key: Mapped[str] = mapped_column(String(256), nullable=False)
    correlation_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    correlation_version: Mapped[str] = mapped_column(String(64), nullable=False)
    correlation_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    first_observed_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    last_observed_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    acknowledged_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    acknowledged_by: Mapped[str | None] = mapped_column(String(256))
    resolved_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    closed_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
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


class SituationEventEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_SITUATION_EVENT"

    situation_event_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    situation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    signal_event_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    relation_type: Mapped[str] = mapped_column(String(16), nullable=False)
    correlation_method: Mapped[str] = mapped_column(String(32), nullable=False)
    correlation_score: Mapped[float | None] = mapped_column(Numeric(8, 6))
    correlation_detail_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    attached_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
    attached_by: Mapped[str] = mapped_column(String(256), nullable=False)
