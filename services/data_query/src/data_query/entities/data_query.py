"""KBOT_DQ_* 表的 SQLAlchemy 映射。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, Index, Integer, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UniversalTimestamp,
    UUIDv7Type,
)


class _VersionedEntity(BaseEntity):
    __abstract__ = True

    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    __mapper_args__ = {"version_id_col": row_version}


class DataSourceEntity(_VersionedEntity):
    __tablename__ = "KBOT_DQ_DATA_SOURCE"
    __table_args__ = (
        UniqueConstraint("domain_id", "display_name", name="uq_dq_source_name"),
        UniqueConstraint("data_source_id", "credential_id", name="uq_dq_source_cred"),
    )

    data_source_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="DRAFT", index=True)
    current_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    configuration_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    configuration_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    credential_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    capabilities_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False, default=dict)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)


class SchemaSnapshotEntity(_VersionedEntity):
    __tablename__ = "KBOT_DQ_SCHEMA_SNAPSHOT"

    schema_snapshot_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    data_source_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    source_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="REQUESTED", index=True)
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    connector_type: Mapped[str] = mapped_column(String(32), nullable=False)
    connector_version: Mapped[str] = mapped_column(String(64), nullable=False)
    capabilities_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    objects_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))
    requested_by: Mapped[str] = mapped_column(String(256), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))


class SchemaSnapshotObjectEntity(_VersionedEntity):
    """一次结构采集中的单个表或视图任务。

    快照批次负责生命周期，对象行负责选择、进度、局部失败、重试与人工补录。
    """

    __tablename__ = "KBOT_DQ_SNAPSHOT_OBJECT"
    __table_args__ = (
        UniqueConstraint(
            "schema_snapshot_id", "schema_name", "object_name",
            name="uq_KBOT_DQ_snapshot_object",
        ),
        Index("ix_KBOT_DQ_snapshot_object_status", "schema_snapshot_id", "status"),
    )

    schema_snapshot_object_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    schema_snapshot_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    schema_name: Mapped[str] = mapped_column(String(128), nullable=False)
    object_name: Mapped[str] = mapped_column(String(128), nullable=False)
    object_type: Mapped[str] = mapped_column(String(24), nullable=False)
    selected: Mapped[int] = mapped_column(Numeric(1, 0), nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="DISCOVERED", index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_source: Mapped[str] = mapped_column(String(16), nullable=False, default="AUTO")
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))
    started_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))


class SemanticModelEntity(_VersionedEntity):
    __tablename__ = "KBOT_DQ_SEMANTIC_MODEL"

    semantic_model_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000))
    active_version: Mapped[int | None] = mapped_column(Integer)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)


class SemanticModelVersionEntity(_VersionedEntity):
    __tablename__ = "KBOT_DQ_MODEL_VERSION"
    __table_args__ = (
        UniqueConstraint("semantic_model_id", "version_no", name="uq_KBOT_DQ_model_version"),
    )

    semantic_model_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    semantic_model_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    data_source_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    schema_snapshot_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="DRAFT", index=True)
    definition_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    definition_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    review_comment: Mapped[str | None] = mapped_column(Text)
    submitted_by: Mapped[str | None] = mapped_column(String(256))
    reviewed_by: Mapped[str | None] = mapped_column(String(256))
    published_by: Mapped[str | None] = mapped_column(String(256))
    published_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))


class SemanticModelGenerationJobEntity(_VersionedEntity):
    """可恢复的语义模型生成作业。"""

    __tablename__ = "KBOT_DQ_MODEL_GEN_JOB"
    __table_args__ = (
        Index("ix_dq_model_generation_claim", "status", "created_at"),
        CheckConstraint("status IN ('QUEUED','RUNNING','SUCCEEDED','FAILED')", name="ck_dq_model_generation_status"),
    )

    generation_job_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, index=True)
    schema_snapshot_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    requested_by: Mapped[str] = mapped_column(String(256), nullable=False)
    request_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="QUEUED", index=True)
    semantic_model_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    semantic_model_version_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    error_code: Mapped[str | None] = mapped_column(String(128))
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_owner: Mapped[str | None] = mapped_column(String(256))
    lease_token: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    lease_until: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))


class PolicyBindingEntity(_VersionedEntity):
    __tablename__ = "KBOT_DQ_POLICY"

    policy_binding_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, index=True)
    subject_selector_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    semantic_model_ids_json: Mapped[list[str]] = mapped_column(OracleNativeJSON, nullable=False)
    policy_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    policy_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="ACTIVE", index=True)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)


class AgentBindingEntity(_VersionedEntity):
    __tablename__ = "KBOT_DQ_AGENT_BINDING"
    __table_args__ = (UniqueConstraint(
        "domain_id", "consumer_app_id", "agent_id", "agent_version_id",
        "semantic_model_id", "policy_binding_id",
        name="uq_KBOT_DQ_agent_binding",
    ),)

    agent_binding_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, index=True)
    consumer_app_id: Mapped[str] = mapped_column(String(128), nullable=False)
    agent_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    agent_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    semantic_model_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    policy_binding_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="ACTIVE", index=True)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)


class VerifiedQueryEntity(_VersionedEntity):
    __tablename__ = "KBOT_DQ_VERIFIED_QUERY"
    __table_args__ = (UniqueConstraint("semantic_model_version_id", "question_hash", name="uq_KBOT_DQ_verified_question"),)

    verified_query_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    semantic_model_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    query_plan_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    assertion_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="DRAFT", index=True)
    verified_by: Mapped[str | None] = mapped_column(String(256))
    verified_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))


class DataQueryRunEntity(_VersionedEntity):
    __tablename__ = "KBOT_DQ_RUN"
    __table_args__ = (UniqueConstraint("domain_id", "actor_id", "idempotency_key", name="uq_KBOT_DQ_run_idempotency"),)

    data_query_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, index=True)
    actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    consumer_app_id: Mapped[str] = mapped_column(String(128), nullable=False)
    agent_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    agent_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    parent_agent_run_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), index=True)
    parent_agent_task_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    original_question: Mapped[str] = mapped_column(Text, nullable=False)
    standalone_query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="CREATED", index=True)
    plan_snapshot_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    policy_snapshot_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    semantic_model_snapshot_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    deadline_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    cancel_requested_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))


class DataQueryExecutionEntity(_VersionedEntity):
    __tablename__ = "KBOT_DQ_EXECUTION"
    __table_args__ = (UniqueConstraint("data_query_run_id", "attempt_no", name="uq_KBOT_DQ_execution_attempt"),)

    data_query_execution_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, index=True)
    data_query_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="CREATED", index=True)
    connector_type: Mapped[str] = mapped_column(String(32), nullable=False)
    connector_version: Mapped[str] = mapped_column(String(64), nullable=False)
    query_plan_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    compiled_query_hash: Mapped[str | None] = mapped_column(String(64))
    preflight_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    execution_summary_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    lease_owner: Mapped[str | None] = mapped_column(String(256))
    lease_token: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    lease_until: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True), index=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))


class DataQueryResultEntity(_VersionedEntity):
    __tablename__ = "KBOT_DQ_RESULT"
    __table_args__ = (UniqueConstraint("data_query_run_id", name="uq_KBOT_DQ_result_run"),)

    data_query_result_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, index=True)
    data_query_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    columns_json: Mapped[list[dict[str, Any]]] = mapped_column(OracleNativeJSON, nullable=False)
    preview_rows_json: Mapped[list[dict[str, Any]]] = mapped_column(OracleNativeJSON, nullable=False)
    row_count: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    observed_row_count: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    truncated: Mapped[int] = mapped_column(Numeric(1, 0), nullable=False, default=0)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    storage_uri: Mapped[str | None] = mapped_column(String(2048))
    byte_size: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    available_until: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False, index=True
    )
    purged_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))


class DataQueryEventEntity(BaseEntity):
    __tablename__ = "KBOT_DQ_EVENT"
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, index=True)
    data_query_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True)
    sequence_no: Mapped[int] = mapped_column(Numeric(19, 0), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    event_key: Mapped[str | None] = mapped_column(String(128))
    visibility: Mapped[str] = mapped_column(String(16), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)


class DataQueryAuditEntity(BaseEntity):
    __tablename__ = "KBOT_DQ_AUDIT"
    audit_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    data_query_run_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), index=True)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, index=True)
    actor_id: Mapped[str | None] = mapped_column(String(256), index=True)
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    previous_hash: Mapped[str | None] = mapped_column(String(64))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
