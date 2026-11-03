"""与诊断 Run 分离的持久化输出任务及逐次执行记录。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class OpsOutputJobEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_OUTPUT_JOB"

    job_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    job_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    source_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    source_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    conversation_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    anchor_run_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    idempotency_key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_snapshot_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="QUEUED")
    stage: Mapped[str] = mapped_column(String(24), nullable=False, default="ACCEPTED")
    attempt_count: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False, default=0)
    available_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), nullable=False)
    lease_owner: Mapped[str | None] = mapped_column(String(256))
    lease_token: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    lease_until: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    result_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    processing_ref_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(2000))
    failure_stage: Mapped[str | None] = mapped_column(String(24))
    retryable: Mapped[bool] = mapped_column(Numeric(1, 0), nullable=False, default=0)
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    __mapper_args__ = {"version_id_col": row_version}
    __table_args__ = (
        Index("ix_ops_output_job_queue", "status", "available_at", "lease_until"),
        Index("ix_ops_output_job_actor", "domain_id", "actor_id", "job_id"),
        Index("ix_ops_output_job_source", "domain_id", "source_kind", "source_id"),
        Index("ix_ops_output_job_idempotency", "domain_id", "actor_id", "idempotency_key_hash"),
        Index("ix_ops_output_job_conversation", "conversation_id", "job_id"),
    )


class OpsOutputJobAttemptEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_OUTPUT_JOB_ATTEMPT"

    attempt_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    job_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_OUTPUT_JOB.job_id"), nullable=False)
    attempt_no: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False)
    lease_token: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    lease_owner: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    stage: Mapped[str] = mapped_column(String(24), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(2000))
    __table_args__ = (Index("ix_ops_output_attempt_job", "job_id", "attempt_no"),)
