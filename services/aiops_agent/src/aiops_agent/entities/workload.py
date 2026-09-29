"""数据库无关的工作负载快照、语句、指标和活动采样 Oracle 映射。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Index, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UniversalTimestamp,
    UUIDv7Type,
)


class WorkloadSnapshotEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_WORKLOAD_SNAPSHOT"
    __table_args__ = (
        Index(
            "IX_OPS_WORKLOAD_SCOPE_TIME",
            "domain_id",
            "target_id",
            "collected_at",
            "workload_snapshot_id",
        ),
        Index(
            "IX_OPS_WORKLOAD_EXPIRY",
            "expires_at",
            "workload_snapshot_id",
        ),
        Index(
            "IX_OPS_WORKLOAD_SCHEDULE",
            "target_id",
            "scheduled_for",
        ),
    )

    workload_snapshot_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    scheduled_for: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    collected_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    window_seconds: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False)
    database_type: Mapped[str] = mapped_column(String(16), nullable=False)
    instance_identity_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    instance_started_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    database_version: Mapped[str] = mapped_column(String(64), nullable=False)
    capability_probe_version: Mapped[str] = mapped_column(String(64), nullable=False)
    catalog_version: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    continuity_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    coverage_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    instance_metrics_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    replication_metrics_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(128))
    byte_size: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class WorkloadStatementEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_WORKLOAD_STATEMENT"
    __table_args__ = (
        Index(
            "IX_OPS_WORKLOAD_STMT_SUBJECT",
            "domain_id",
            "target_id",
            "statement_identity_type",
            "statement_identity_value",
            "workload_statement_id",
        ),
        Index("IX_OPS_WORKLOAD_STMT_SNAPSHOT", "workload_snapshot_id"),
        Index("IX_OPS_WORKLOAD_STMT_TARGET", "target_id"),
    )

    workload_statement_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    workload_snapshot_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    statement_identity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    statement_identity_value: Mapped[str] = mapped_column(String(128), nullable=False)
    database_name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    user_identifier: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    top_level: Mapped[int] = mapped_column(Numeric(2, 0), nullable=False, default=-1)
    normalized_statement: Mapped[str | None] = mapped_column(Text)
    execution_count: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    total_duration_microseconds: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    mean_duration_microseconds: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    max_duration_microseconds: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    rows_processed: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    database_metrics_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    rank_dimension: Mapped[str] = mapped_column(String(32), nullable=False)
    rank_no: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False)
    quality_status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class WorkloadMetricEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_WORKLOAD_METRIC"
    __table_args__ = (
        Index(
            "IX_OPS_WORKLOAD_METRIC_FAMILY",
            "domain_id",
            "target_id",
            "metric_family",
            "workload_metric_id",
        ),
        Index("IX_OPS_WORKLOAD_METRIC_SNAP", "workload_snapshot_id"),
        Index("IX_OPS_WORKLOAD_METRIC_TARGET", "target_id"),
    )

    workload_metric_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    workload_snapshot_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    metric_family: Mapped[str] = mapped_column(String(32), nullable=False)
    metric_code: Mapped[str] = mapped_column(String(128), nullable=False)
    dimension_key: Mapped[str] = mapped_column(String(256), nullable=False)
    dimension_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    counter_value: Mapped[float | None] = mapped_column(Numeric(30, 6))
    gauge_value: Mapped[float | None] = mapped_column(Numeric(30, 6))
    unit: Mapped[str] = mapped_column(String(32), nullable=False)
    quality_status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class ActivitySampleEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_ACTIVITY_SAMPLE"
    __table_args__ = (
        Index(
            "IX_OPS_ACTIVITY_SCOPE_TIME",
            "domain_id",
            "target_id",
            "sampled_at",
            "activity_sample_id",
        ),
        Index(
            "IX_OPS_ACTIVITY_EXPIRY",
            "expires_at",
            "activity_sample_id",
        ),
        Index("IX_OPS_ACTIVITY_TARGET", "target_id"),
    )

    activity_sample_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    database_type: Mapped[str] = mapped_column(String(16), nullable=False)
    sampled_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    instance_identity_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    session_identifier: Mapped[str | None] = mapped_column(String(128))
    worker_identifier: Mapped[str | None] = mapped_column(String(128))
    statement_identity_type: Mapped[str | None] = mapped_column(String(32))
    statement_identity_value: Mapped[str | None] = mapped_column(String(128))
    database_name: Mapped[str | None] = mapped_column(String(128))
    schema_name: Mapped[str | None] = mapped_column(String(128))
    username_hash: Mapped[str | None] = mapped_column(String(64))
    client_hash: Mapped[str | None] = mapped_column(String(64))
    command_name: Mapped[str | None] = mapped_column(String(64))
    session_state: Mapped[str | None] = mapped_column(String(256))
    stage_name: Mapped[str | None] = mapped_column(String(256))
    wait_name: Mapped[str | None] = mapped_column(String(256))
    database_sample_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    transaction_active: Mapped[int] = mapped_column(Numeric(1, 0), nullable=False, default=0)
    lock_waiting: Mapped[int] = mapped_column(Numeric(1, 0), nullable=False, default=0)
    sample_weight: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False, default=1)
    quality_status: Mapped[str] = mapped_column(String(16), nullable=False)
    byte_size: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=0)
    expires_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
