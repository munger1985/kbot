"""AIOps 对话、人工证据与逐条动作实体。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class OpsConversationEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_CONVERSATION"
    conversation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, index=True)
    agent_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    agent_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")
    source_type: Mapped[str] = mapped_column(String(24), nullable=False, default="CHAT")
    source_alert_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    source_run_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    source_report_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    __mapper_args__ = {"version_id_col": row_version}


class OpsConversationMessageEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_CONVERSATION_MESSAGE"
    __table_args__ = (UniqueConstraint("conversation_id", "sequence_no", name="UK_OPS_CONV_MSG_SEQ"),)
    message_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    conversation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    sequence_no: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    message_type: Mapped[str] = mapped_column(String(48), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    created_by: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)


class OpsConversationRunEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_CONVERSATION_RUN"
    __table_args__ = (
        UniqueConstraint("ops_run_id", name="UK_OPS_CONV_RUN"),
        UniqueConstraint("conversation_id", "sequence_no", name="UK_OPS_CONV_RUN_SEQ"),
    )
    conversation_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    conversation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    ops_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    purpose: Mapped[str] = mapped_column(String(48), nullable=False)
    sequence_no: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)


class EvidenceRequestEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_EVIDENCE_REQUEST"
    request_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    conversation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    parent_request_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="OPEN")
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    suggested_sql: Mapped[str | None] = mapped_column(Text)
    sql_hash: Mapped[str | None] = mapped_column(String(64))
    failure_class: Mapped[str | None] = mapped_column(String(48))
    requested_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class ActionStepEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_ACTION_STEP"
    __table_args__ = (UniqueConstraint("conversation_id", "ordinal", name="UK_OPS_ACTION_STEP_ORD"),)
    action_step_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    conversation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    proposal_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    ordinal: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False)
    sql_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="DRAFT")
    supersedes_step_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    __mapper_args__ = {"version_id_col": row_version}


class ImageEvidenceProcessingEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_IMAGE_EVIDENCE"
    processing_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    conversation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    evidence_request_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    source_artifact_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    processing_mode: Mapped[str] = mapped_column(String(8), nullable=False)
    model_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(128), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="PENDING")
    output_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(2000))
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
