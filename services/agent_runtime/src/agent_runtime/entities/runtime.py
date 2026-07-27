"""KBOT_AGENT_* 表的 SQLAlchemy 映射。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UniversalTimestamp,
    UUIDv7Type,
)


class AgentRunEntity(BaseEntity):
    __tablename__ = "KBOT_AGENT_RUN"

    run_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    agent_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    parent_run_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    original_input: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="CREATED"
    )
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    policy_snapshot_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    config_snapshot_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    budget_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    deadline_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    final_task_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    result_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now()
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
    )
    __mapper_args__ = {"version_id_col": row_version}


class AgentTaskEntity(BaseEntity):
    __tablename__ = "KBOT_AGENT_TASK"

    task_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    parent_task_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    task_key: Mapped[str] = mapped_column(String(64), nullable=False)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    execution_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    specialist: Mapped[str | None] = mapped_column(String(64))
    skill_id: Mapped[str | None] = mapped_column(String(128))
    skill_version: Mapped[str | None] = mapped_column(String(64))
    delegate_service: Mapped[str | None] = mapped_column(String(128))
    delegate_capability: Mapped[str | None] = mapped_column(String(128))
    execution_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    completion_requirement: Mapped[str] = mapped_column(
        String(16), nullable=False, default="REQUIRED"
    )
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="PENDING"
    )
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    depends_on_json: Mapped[list[str]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    input_artifacts_json: Mapped[list[str]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    expected_outputs_json: Mapped[list[str]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    required_scopes_json: Mapped[list[str]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    output_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1
    )
    timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    lease_owner: Mapped[str | None] = mapped_column(String(256))
    lease_token: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    lease_until: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    next_retry_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now()
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
    )
    __mapper_args__ = {"version_id_col": row_version}


class AgentArtifactEntity(BaseEntity):
    __tablename__ = "KBOT_AGENT_ARTIFACT"

    artifact_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    task_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    producer: Mapped[str] = mapped_column(String(128), nullable=False)
    producer_version: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict[str, Any] | list[Any] | None] = mapped_column(
        OracleNativeJSON
    )
    storage_uri: Mapped[str | None] = mapped_column(String(2048))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    provenance_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    security_level: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now()
    )


class AgentRunEventEntity(BaseEntity):
    __tablename__ = "KBOT_AGENT_RUN_EVENT"

    run_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True
    )
    sequence_no: Mapped[int] = mapped_column(
        Numeric(19, 0), primary_key=True
    )
    task_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    event_key: Mapped[str | None] = mapped_column(String(256))
    artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    event_payload_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    actor_type: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now()
    )


class AgentDelegationEntity(BaseEntity):
    __tablename__ = "KBOT_AGENT_DELEGATION"

    delegation_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    parent_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    parent_task_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    target_service: Mapped[str] = mapped_column(String(128), nullable=False)
    target_capability: Mapped[str] = mapped_column(String(128), nullable=False)
    child_run_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="CREATED"
    )
    last_child_event_sequence: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=0
    )
    next_poll_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    result_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=3
    )
    lease_owner: Mapped[str | None] = mapped_column(String(256))
    lease_token: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    lease_until: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    __mapper_args__ = {"version_id_col": row_version}
