"""AIOps Run、Task、Artifact 和事件流的 SQLAlchemy 映射。"""

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


class OpsRunEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_RUN"

    ops_run_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, index=True)
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    agent_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    parent_agent_run_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    parent_delegation_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    trigger_type: Mapped[str] = mapped_column(String(16), nullable=False)
    trigger_event_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    trigger_alert_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    inspection_fire_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    source_proposal_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    source_result_artifact_id: Mapped[UUID | None] = mapped_column(
        UUIDv7Type()
    )
    actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    original_request: Mapped[str | None] = mapped_column(Text)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="CREATED"
    )
    plan_snapshot_json: Mapped[dict[str, Any] | None] = mapped_column(
        OracleNativeJSON
    )
    policy_snapshot_json: Mapped[dict[str, Any] | None] = mapped_column(
        OracleNativeJSON
    )
    root_cause_level: Mapped[str | None] = mapped_column(String(16))
    final_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    deadline_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    cancel_requested_by: Mapped[str | None] = mapped_column(String(256))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(2000))
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    __mapper_args__ = {"version_id_col": row_version}


class OpsTaskEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_TASK"

    ops_task_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    ops_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    parent_task_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    task_key: Mapped[str] = mapped_column(String(128), nullable=False)
    task_type: Mapped[str] = mapped_column(String(24), nullable=False)
    handler_id: Mapped[str] = mapped_column(String(128), nullable=False)
    handler_version: Mapped[str] = mapped_column(String(64), nullable=False)
    input_schema_version: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    output_schema_version: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    depends_on_json: Mapped[list[str] | None] = mapped_column(OracleNativeJSON)
    input_artifacts_json: Mapped[list[str] | None] = mapped_column(OracleNativeJSON)
    output_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    priority: Mapped[int] = mapped_column(
        Numeric(8, 0), nullable=False, default=100
    )
    available_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
    attempt_count: Mapped[int] = mapped_column(
        Numeric(8, 0), nullable=False, default=0
    )
    max_attempts: Mapped[int] = mapped_column(
        Numeric(8, 0), nullable=False, default=3
    )
    timeout_seconds: Mapped[int] = mapped_column(
        Numeric(10, 0), nullable=False
    )
    lease_owner: Mapped[str | None] = mapped_column(String(256))
    lease_token: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    lease_until: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    started_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(2000))
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


class OpsArtifactEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_ARTIFACT"

    artifact_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    ops_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    ops_task_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    artifact_key: Mapped[str] = mapped_column(String(256), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[Any | None] = mapped_column(OracleNativeJSON)
    payload_uri: Mapped[str | None] = mapped_column(String(2048))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    byte_size: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    provenance_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    trust_level: Mapped[str] = mapped_column(String(24), nullable=False)
    security_level: Mapped[int] = mapped_column(Numeric(3, 0), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class OpsRunEventEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_RUN_EVENT"

    ops_run_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True
    )
    ops_task_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    sequence_no: Mapped[int] = mapped_column(
        Numeric(19, 0), primary_key=True
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    event_key: Mapped[str | None] = mapped_column(String(128))
    visibility: Mapped[str] = mapped_column(String(16), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
