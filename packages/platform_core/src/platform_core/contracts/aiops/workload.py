"""数据库无关的工作负载、报告和活动采样 Wire 契约。"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .types import AIOpsContract, JsonObject, UUIDv7, UtcDatetime


WORKLOAD_SNAPSHOT_SCHEMA_VERSION = "AIOPS_WORKLOAD_SNAPSHOT.v2"
WORKLOAD_REPORT_SCHEMA_VERSION = "AIOPS_WORKLOAD_REPORT.v2"
WORKLOAD_DIFF_REPORT_SCHEMA_VERSION = "AIOPS_WORKLOAD_DIFF_REPORT.v2"
ACTIVITY_REPORT_SCHEMA_VERSION = "AIOPS_ACTIVITY_REPORT.v2"
SQL_HEALTHCHECK_REPORT_SCHEMA_VERSION = "AIOPS_SQL_HEALTHCHECK_REPORT.v2"
PGBADGER_MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class WorkloadDatabaseType(StrEnum):
    MYSQL = "MYSQL"
    POSTGRESQL = "POSTGRESQL"


class ReportOrigin(StrEnum):
    ORACLE_NATIVE = "ORACLE_NATIVE"
    AIOPS_GENERATED = "AIOPS_GENERATED"
    EXTERNAL_IMPORTED = "EXTERNAL_IMPORTED"


class WorkloadReportType(StrEnum):
    MYSQL_WORKLOAD = "MYSQL_WORKLOAD"
    MYSQL_WORKLOAD_DIFF = "MYSQL_WORKLOAD_DIFF"
    MYSQL_ACTIVITY = "MYSQL_ACTIVITY"
    MYSQL_SQL_HEALTHCHECK = "MYSQL_SQL_HEALTHCHECK"
    POSTGRESQL_WORKLOAD = "POSTGRESQL_WORKLOAD"
    POSTGRESQL_WORKLOAD_DIFF = "POSTGRESQL_WORKLOAD_DIFF"
    POSTGRESQL_ACTIVITY = "POSTGRESQL_ACTIVITY"
    POSTGRESQL_SQL_HEALTHCHECK = "POSTGRESQL_SQL_HEALTHCHECK"
    POSTGRESQL_PGBADGER = "POSTGRESQL_PGBADGER"


class StatementIdentityType(StrEnum):
    MYSQL_DIGEST = "MYSQL_DIGEST"
    POSTGRESQL_QUERY_ID = "POSTGRESQL_QUERY_ID"


class CollectionQuality(StrEnum):
    AVAILABLE = "AVAILABLE"
    DISABLED = "DISABLED"
    DENIED = "DENIED"
    TIMEOUT = "TIMEOUT"
    PARTIAL = "PARTIAL"
    RESET = "RESET"
    EVICTED = "EVICTED"


class WorkloadStatementSnapshot(AIOpsContract):
    statement_identity_type: StatementIdentityType
    statement_identity_value: str = Field(min_length=1, max_length=128)
    database_name: str = Field(default="", max_length=128)
    user_identifier: str = Field(default="", max_length=128)
    top_level: bool | None = None
    normalized_statement: str | None = Field(default=None, max_length=4000)
    execution_count: int = Field(ge=0, le=2**63 - 1)
    total_duration_microseconds: int = Field(ge=0, le=2**63 - 1)
    mean_duration_microseconds: int = Field(ge=0, le=2**63 - 1)
    max_duration_microseconds: int = Field(ge=0, le=2**63 - 1)
    rows_processed: int = Field(ge=0, le=2**63 - 1)
    database_metrics: JsonObject = Field(default_factory=dict)
    rank_dimension: str = Field(default="TOTAL_DURATION", max_length=32)
    rank_no: int = Field(ge=1)
    quality_status: CollectionQuality = CollectionQuality.AVAILABLE

    @model_validator(mode="after")
    def validate_identity(self) -> "WorkloadStatementSnapshot":
        value = self.statement_identity_value
        if self.statement_identity_type == StatementIdentityType.MYSQL_DIGEST:
            if len(value) != 64 or any(c not in "0123456789ABCDEF" for c in value):
                raise ValueError("MySQL Digest 必须是64位大写十六进制字符串")
        else:
            if not _signed_bigint_string(value):
                raise ValueError("PostgreSQL query_id 必须是有符号bigint十进制字符串")
        return self


class WorkloadMetricSnapshot(AIOpsContract):
    metric_family: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,31}$")
    metric_code: str = Field(pattern=r"^[a-z][a-z0-9_.]{0,127}$")
    dimension_key: str = Field(min_length=1, max_length=256)
    dimensions: JsonObject = Field(default_factory=dict)
    counter_value: float | None = Field(default=None, ge=0)
    gauge_value: float | None = Field(default=None)
    unit: str = Field(min_length=1, max_length=32)
    quality_status: CollectionQuality

    @model_validator(mode="after")
    def validate_value(self) -> "WorkloadMetricSnapshot":
        if self.counter_value is None and self.gauge_value is None:
            raise ValueError("工作负载指标必须包含counter或gauge值")
        return self


class WorkloadSnapshotCreate(AIOpsContract):
    schema_version: str = WORKLOAD_SNAPSHOT_SCHEMA_VERSION
    target_id: UUIDv7
    database_type: WorkloadDatabaseType
    scheduled_for: UtcDatetime
    collected_at: UtcDatetime
    window_seconds: int = Field(ge=60, le=3600)
    instance_identity: JsonObject
    instance_started_at: UtcDatetime
    database_version: str = Field(min_length=1, max_length=64)
    capability_probe_version: str = Field(min_length=1, max_length=64)
    catalog_version: str = Field(min_length=1, max_length=64)
    continuity: JsonObject = Field(default_factory=dict)
    coverage: dict[str, CollectionQuality]
    instance_metrics: dict[str, int | float]
    replication_metrics: JsonObject = Field(default_factory=dict)
    statements: tuple[WorkloadStatementSnapshot, ...] = Field(default=(), max_length=500)
    metrics: tuple[WorkloadMetricSnapshot, ...] = Field(default=(), max_length=5000)
    status: Literal["READY", "PARTIAL", "FAILED"] = "READY"
    error_code: str | None = Field(default=None, max_length=128)
    retention_days: int = Field(default=30, ge=1, le=365)


class ReportWindow(AIOpsContract):
    target_id: UUIDv7
    period_start: UtcDatetime
    period_end: UtcDatetime

    @model_validator(mode="after")
    def validate_window(self) -> "ReportWindow":
        if self.period_start >= self.period_end:
            raise ValueError("报告起始时间必须早于结束时间")
        return self


class WorkloadDiffRequest(AIOpsContract):
    target_id: UUIDv7
    baseline_start: UtcDatetime
    baseline_end: UtcDatetime
    after_start: UtcDatetime
    after_end: UtcDatetime

    @model_validator(mode="after")
    def validate_windows(self) -> "WorkloadDiffRequest":
        if not self.baseline_start < self.baseline_end <= self.after_start < self.after_end:
            raise ValueError("对比窗口必须按时间先后且不重叠")
        return self


class ActivitySamplerPolicy(AIOpsContract):
    enabled: bool = False
    interval_seconds: int = Field(default=5, ge=1, le=30)
    retention_days: int = Field(default=7, ge=1, le=30)
    query_timeout_seconds: int = Field(default=2, ge=1, le=10)
    max_rows_per_sample: int = Field(default=200, ge=1, le=1000)
    max_daily_bytes: int = Field(default=100 * 1024 * 1024, ge=1024 * 1024, le=10 * 1024 * 1024 * 1024)


class WorkloadPolicy(AIOpsContract):
    enabled: bool = False
    interval_minutes: int = Field(default=5, ge=1, le=60)
    retention_days: int = Field(default=30, ge=1, le=365)
    top_statement_limit: int = Field(default=100, ge=10, le=500)
    round_timeout_seconds: int = Field(default=30, ge=5, le=120)


class ActivitySample(AIOpsContract):
    session_identifier: str | None = Field(default=None, max_length=128)
    worker_identifier: str | None = Field(default=None, max_length=128)
    statement_identity_type: StatementIdentityType | None = None
    statement_identity_value: str | None = Field(default=None, max_length=128)
    database_name: str | None = Field(default=None, max_length=128)
    schema_name: str | None = Field(default=None, max_length=128)
    username: str | None = Field(default=None, max_length=256)
    client: str | None = Field(default=None, max_length=512)
    command_name: str | None = Field(default=None, max_length=64)
    session_state: str | None = Field(default=None, max_length=256)
    stage_name: str | None = Field(default=None, max_length=256)
    wait_name: str | None = Field(default=None, max_length=256)
    database_sample: JsonObject = Field(default_factory=dict)
    transaction_active: bool = False
    lock_waiting: bool = False
    sample_weight: int = Field(default=1, ge=1, le=1000)
    quality_status: CollectionQuality = CollectionQuality.AVAILABLE

    @model_validator(mode="after")
    def validate_statement_identity(self) -> "ActivitySample":
        identity_type = self.statement_identity_type
        identity_value = self.statement_identity_value
        if (identity_type is None) != (identity_value is None):
            raise ValueError("活动样本的语句身份类型和值必须同时提供")
        if identity_type == StatementIdentityType.MYSQL_DIGEST:
            if (
                len(identity_value or "") != 64
                or any(c not in "0123456789ABCDEF" for c in identity_value or "")
            ):
                raise ValueError("活动样本中的MySQL Digest无效")
        if identity_type == StatementIdentityType.POSTGRESQL_QUERY_ID and not (
            _signed_bigint_string(identity_value or "")
        ):
            raise ValueError("活动样本中的PostgreSQL query_id无效")
        return self


class ActivitySampleBatch(AIOpsContract):
    target_id: UUIDv7
    database_type: WorkloadDatabaseType
    sampled_at: UtcDatetime
    instance_identity: JsonObject
    retention_days: int = Field(default=7, ge=1, le=30)
    samples: tuple[ActivitySample, ...] = Field(max_length=1000)


class ReportArtifactView(AIOpsContract):
    artifact_id: UUIDv7
    report_type: WorkloadReportType
    report_origin: ReportOrigin
    title: str = Field(min_length=1, max_length=512)
    period_start: UtcDatetime
    period_end: UtcDatetime
    format: str = Field(pattern=r"^(JSON|HTML|PDF)$")
    content_type: str = Field(min_length=1, max_length=128)
    file_name: str = Field(min_length=1, max_length=256)
    byte_size: int = Field(ge=0)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    download_url: str = Field(min_length=1, max_length=2048)


def report_type_for(database_type: WorkloadDatabaseType, kind: str) -> WorkloadReportType:
    return WorkloadReportType(f"{database_type}_{kind}")


def _signed_bigint_string(value: str) -> bool:
    if not value or value in {"-", "+"}:
        return False
    if value.startswith("+"):
        return False
    digits = value[1:] if value.startswith("-") else value
    if not digits.isdigit() or (len(digits) > 1 and digits.startswith("0")):
        return False
    try:
        number = int(value)
    except ValueError:
        return False
    return -(2**63) <= number <= 2**63 - 1
