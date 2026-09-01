"""AIOps 巡检计划、调度 Fire 和报告的 SQLAlchemy 映射。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Computed, Index, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UniversalTimestamp,
    UUIDv7Type,
)


class InspectionReportTemplateEntity(BaseEntity):
    """Domain 私有的巡检报告展示模板。"""

    __tablename__ = "KBOT_OPS_REPORT_TEMPLATE"
    __table_args__ = (
        UniqueConstraint("domain_id", "display_name", name="UK_OPS_REPORT_TEMPLATE_NAME"),
        Index("IX_OPS_REPORT_TEMPLATE_SCOPE", "domain_id", "status"),
    )
    template_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    current_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    __mapper_args__ = {"version_id_col": row_version}


class InspectionReportTemplateVersionEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_REPORT_TEMPLATE_VER"
    __table_args__ = (
        UniqueConstraint("template_id", "version_no", name="UK_OPS_REPORT_TEMPLATE_VER"),
        UniqueConstraint("template_id", "content_hash", name="UK_OPS_REPORT_TEMPLATE_HASH"),
    )
    template_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    template_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    version_no: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    definition_json: Mapped[dict] = mapped_column(OracleNativeJSON, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)


class InspectionPlanEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_INSPECTION_PLAN"

    inspection_plan_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    agent_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    schedule_type: Mapped[str] = mapped_column(String(16), nullable=False)
    cron_expression: Mapped[str] = mapped_column(String(256), nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    template_id: Mapped[str] = mapped_column(String(128), nullable=False)
    template_version: Mapped[str] = mapped_column(String(64), nullable=False)
    timeout_seconds: Mapped[int] = mapped_column(
        Numeric(10, 0), nullable=False
    )
    overlap_policy: Mapped[str] = mapped_column(String(16), nullable=False)
    misfire_policy: Mapped[str] = mapped_column(String(16), nullable=False)
    schedule_resolver_version: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="ACTIVE"
    )
    next_run_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    last_run_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    last_scheduled_for: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    lease_owner: Mapped[str | None] = mapped_column(String(256))
    lease_token: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    lease_until: Mapped[datetime | None] = mapped_column(
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


class InspectionFireEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_INSPECTION_FIRE"

    inspection_fire_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    inspection_plan_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), nullable=False
    )
    scheduled_for: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    scheduled_for_utc: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=False),
        Computed("SYS_EXTRACT_UTC(SCHEDULED_FOR)"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    plan_row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False
    )
    template_id: Mapped[str] = mapped_column(String(128), nullable=False)
    template_version: Mapped[str] = mapped_column(String(64), nullable=False)
    schedule_resolver_version: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    plan_snapshot_json: Mapped[dict] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    resolution_json: Mapped[dict | None] = mapped_column(OracleNativeJSON)
    target_count: Mapped[int] = mapped_column(
        Numeric(10, 0), nullable=False, default=0
    )
    run_count: Mapped[int] = mapped_column(
        Numeric(10, 0), nullable=False, default=0
    )
    completed_count: Mapped[int] = mapped_column(
        Numeric(10, 0), nullable=False, default=0
    )
    failed_count: Mapped[int] = mapped_column(
        Numeric(10, 0), nullable=False, default=0
    )
    skip_reason: Mapped[str | None] = mapped_column(String(256))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(2000))
    started_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
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
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    __mapper_args__ = {"version_id_col": row_version}


class ReportEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_REPORT"

    report_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    ops_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    report_key: Mapped[str] = mapped_column(String(128), nullable=False)
    report_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False
    )
    supersedes_report_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    is_current: Mapped[int] = mapped_column(
        Numeric(1, 0), nullable=False, default=0
    )
    report_type: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    period_start: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    period_end: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    baseline_start: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    baseline_end: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    after_start: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    after_end: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    result: Mapped[str | None] = mapped_column(String(16))
    template_id: Mapped[str] = mapped_column(String(128), nullable=False)
    template_version: Mapped[str] = mapped_column(String(64), nullable=False)
    generated_by_task_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), nullable=False
    )
    content_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    content_hash: Mapped[str | None] = mapped_column(String(64))
    summary: Mapped[str | None] = mapped_column(String(2000))
    security_level: Mapped[int] = mapped_column(Numeric(3, 0), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
