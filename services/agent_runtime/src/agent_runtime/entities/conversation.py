"""Conversation、Turn、Item、Snapshot 和长期记忆实体。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, Float, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UUIDv7Type,
    VectorField,
)


class AgentConversationEntity(BaseEntity):
    __tablename__ = "KBOT_AGENT_CONVERSATION"

    conversation_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    agent_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    title: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="ACTIVE"
    )
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    last_turn_sequence: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=0
    )
    last_item_sequence: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=0
    )
    retention_policy: Mapped[str] = mapped_column(
        String(32), nullable=False, default="DEFAULT"
    )
    purge_after: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    __mapper_args__ = {"version_id_col": row_version}


class AgentConversationTurnEntity(BaseEntity):
    __tablename__ = "KBOT_AGENT_CONVERSATION_TURN"

    turn_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    conversation_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), nullable=False
    )
    turn_sequence: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False
    )
    user_item_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    root_run_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    assistant_item_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="ACCEPTED"
    )
    raw_input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    context_snapshot_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    idempotency_key: Mapped[str] = mapped_column(
        String(128), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )


class AgentConversationItemEntity(BaseEntity):
    __tablename__ = "KBOT_AGENT_CONVERSATION_ITEM"

    item_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    conversation_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), nullable=False
    )
    item_sequence: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False
    )
    turn_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    item_type: Mapped[str] = mapped_column(String(24), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    visibility: Mapped[str] = mapped_column(
        String(16), nullable=False, default="USER"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AgentMemorySnapshotEntity(BaseEntity):
    __tablename__ = "KBOT_AGENT_MEMORY_SNAPSHOT"

    snapshot_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    conversation_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), nullable=False
    )
    covered_turn_sequence: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False
    )
    summary_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(128))
    prompt_key: Mapped[str] = mapped_column(String(256), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    prompt_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AgentMemoryItemEntity(BaseEntity):
    __tablename__ = "KBOT_AGENT_MEMORY_ITEM"

    memory_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    agent_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    memory_type: Mapped[str] = mapped_column(String(24), nullable=False)
    scope_type: Mapped[str] = mapped_column(String(24), nullable=False)
    canonical_key: Mapped[str] = mapped_column(String(256), nullable=False)
    value_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    search_text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    salience: Mapped[float] = mapped_column(Float, nullable=False)
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    valid_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    sensitivity_level: Mapped[int] = mapped_column(
        Numeric(3, 0), nullable=False, default=0
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    index_profile_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    embedding: Mapped[list[float] | None] = mapped_column(VectorField())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    __mapper_args__ = {"version_id_col": row_version}


class AgentMemoryIndexProfileEntity(BaseEntity):
    __tablename__ = "KBOT_AGENT_MEMORY_INDEX_PROFILE"

    index_profile_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    agent_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    embedding_model_name: Mapped[str] = mapped_column(
        String(128), nullable=False
    )
    embedding_dimension: Mapped[int] = mapped_column(
        Numeric(10, 0), nullable=False
    )
    normalization: Mapped[str] = mapped_column(
        String(16), nullable=False
    )
    config_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AgentMemorySourceEntity(BaseEntity):
    __tablename__ = "KBOT_AGENT_MEMORY_SOURCE"

    memory_source_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    memory_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    conversation_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), nullable=False
    )
    turn_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    item_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    excerpt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    extractor: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AgentMemoryJobEntity(BaseEntity):
    __tablename__ = "KBOT_AGENT_MEMORY_JOB"

    memory_job_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    conversation_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), nullable=False
    )
    turn_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="PENDING"
    )
    attempt_count: Mapped[int] = mapped_column(
        Numeric(10, 0), nullable=False, default=0
    )
    max_attempts: Mapped[int] = mapped_column(
        Numeric(10, 0), nullable=False, default=3
    )
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    lease_owner: Mapped[str | None] = mapped_column(String(256))
    lease_token: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    lease_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))
    result_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
