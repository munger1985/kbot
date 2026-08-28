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
    title: Mapped[str | None] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")
    source_type: Mapped[str] = mapped_column(String(24), nullable=False, default="CHAT")
    source_situation_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    source_run_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    source_report_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    last_turn_no: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=0
    )
    last_message_no: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=0
    )
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    __mapper_args__ = {"version_id_col": row_version}


class OpsConversationTurnEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_CONVERSATION_TURN"
    __table_args__ = (
        UniqueConstraint("conversation_id", "turn_no", name="UK_OPS_TURN_NO"),
        UniqueConstraint(
            "conversation_id", "idempotency_key", name="UK_OPS_TURN_IDEMP"
        ),
    )
    turn_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    conversation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    turn_no: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="QUEUED"
    )
    resolved_target_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    input_analysis_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    task_frame_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    current_plan_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    assessment_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    current_plan_revision: Mapped[int] = mapped_column(
        Numeric(10, 0), nullable=False, default=0
    )
    investigation_round: Mapped[int] = mapped_column(
        Numeric(10, 0), nullable=False, default=0
    )
    tool_call_count: Mapped[int] = mapped_column(
        Numeric(10, 0), nullable=False, default=0
    )
    no_progress_count: Mapped[int] = mapped_column(
        Numeric(10, 0), nullable=False, default=0
    )
    sufficiency_status: Mapped[str | None] = mapped_column(String(32))
    sufficiency_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    sufficiency_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    event_cursor: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=0
    )
    error_domain: Mapped[str | None] = mapped_column(String(32))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(2000))
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
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
    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    cancel_requested_by: Mapped[str | None] = mapped_column(String(256))
    __mapper_args__ = {"version_id_col": row_version}


class OpsConversationMessageEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_CONVERSATION_MESSAGE"
    __table_args__ = (UniqueConstraint("conversation_id", "sequence_no", name="UK_OPS_CONV_MSG_SEQ"),)
    message_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    conversation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    turn_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    sequence_no: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    message_type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_schema: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    created_by: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)


class OpsTurnInputItemEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_TURN_INPUT_ITEM"
    __table_args__ = (
        UniqueConstraint("turn_id", "item_no", name="UK_OPS_INPUT_ITEM_NO"),
    )
    input_item_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    turn_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    message_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    item_no: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False)
    content_type: Mapped[str] = mapped_column(String(24), nullable=False)
    media_type: Mapped[str | None] = mapped_column(String(128))
    content_text: Mapped[str | None] = mapped_column(Text)
    source_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    extracted_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    detected_kind: Mapped[str | None] = mapped_column(String(32))
    detection_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4))
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class OpsTurnRunEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_TURN_RUN"
    __table_args__ = (
        UniqueConstraint("ops_run_id", name="UK_OPS_TURN_RUN"),
        UniqueConstraint("turn_id", "sequence_no", name="UK_OPS_TURN_RUN_SEQ"),
    )
    turn_run_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    turn_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    ops_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    sequence_no: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class OpsInvestigationRevisionEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_INVESTIGATION_REVISION"
    __table_args__ = (
        UniqueConstraint("turn_id", "revision_no", name="UK_OPS_INV_REV_NO"),
    )
    revision_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    turn_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    revision_no: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False)
    revision_type: Mapped[str] = mapped_column(String(24), nullable=False)
    trigger_reason: Mapped[str] = mapped_column(String(1000), nullable=False)
    task_frame_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    plan_artifact_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    assessment_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class OpsPlaybookInvocationEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_PLAYBOOK_INVOCATION"
    __table_args__ = (
        UniqueConstraint("revision_id", "ordinal", name="UK_OPS_PLAY_INV_ORD"),
    )
    playbook_invocation_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    turn_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    revision_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    parent_invocation_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    ops_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    ops_task_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    ordinal: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False)
    playbook_id: Mapped[str] = mapped_column(String(128), nullable=False)
    playbook_version: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    input_schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    input_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    output_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    attempt_count: Mapped[int] = mapped_column(
        Numeric(8, 0), nullable=False, default=0
    )
    error_domain: Mapped[str | None] = mapped_column(String(32))
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


class OpsToolInvocationEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_TOOL_INVOCATION"
    __table_args__ = (
        UniqueConstraint("revision_id", "ordinal", name="UK_OPS_TOOL_INV_ORD"),
        UniqueConstraint(
            "revision_id", "action_id", name="UK_OPS_TOOL_INV_ACTION"
        ),
    )
    tool_invocation_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    turn_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    revision_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    playbook_invocation_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    ops_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    ops_task_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    ordinal: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False)
    action_id: Mapped[str] = mapped_column(String(32), nullable=False)
    tool_id: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_version: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_class: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    input_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    policy_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    output_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    attempt_count: Mapped[int] = mapped_column(Numeric(8, 0), nullable=False, default=0)
    error_domain: Mapped[str | None] = mapped_column(String(32))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(2000))
    started_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False
    )
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    __mapper_args__ = {"version_id_col": row_version}


class OpsTurnEvidenceEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_TURN_EVIDENCE"
    __table_args__ = (
        UniqueConstraint(
            "turn_id", "artifact_id", "evidence_role",
            name="UK_OPS_TURN_EVIDENCE",
        ),
    )
    turn_evidence_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    turn_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    artifact_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    tool_invocation_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    evidence_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Numeric(5, 4))
    extraction_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    evidence_role: Mapped[str] = mapped_column(String(24), nullable=False)
    measurement_semantics: Mapped[str] = mapped_column(String(32), nullable=False)
    observed_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    window_start_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    window_end_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    freshness_status: Mapped[str] = mapped_column(String(16), nullable=False)
    usage_reason: Mapped[str] = mapped_column(String(1000), nullable=False)
    linked_by: Mapped[str] = mapped_column(String(256), nullable=False)
    linked_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class OpsAnswerBlockEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_ANSWER_BLOCK"
    __table_args__ = (
        UniqueConstraint("message_id", "block_no", name="UK_OPS_ANSWER_BLOCK_NO"),
    )
    answer_block_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    turn_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    message_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    block_no: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False)
    block_type: Mapped[str] = mapped_column(String(32), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="ACTIVE"
    )
    supersedes_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class OpsAnswerCitationEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_ANSWER_CITATION"
    __table_args__ = (
        UniqueConstraint(
            "answer_block_id", "turn_evidence_id",
            name="UK_OPS_ANSWER_CIT_EVID",
        ),
    )
    answer_block_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True)
    citation_no: Mapped[int] = mapped_column(Numeric(10, 0), primary_key=True)
    turn_evidence_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    label: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class OpsTurnEventEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_TURN_EVENT"
    turn_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True)
    sequence_no: Mapped[int] = mapped_column(Numeric(19, 0), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    event_key: Mapped[str | None] = mapped_column(String(128))
    visibility: Mapped[str] = mapped_column(String(16), nullable=False)
    playbook_invocation_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    tool_invocation_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    answer_block_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    payload_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class EvidenceRequestEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_EVIDENCE_REQUEST"
    request_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    turn_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    parent_request_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    request_type: Mapped[str] = mapped_column(String(24), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="OPEN")
    instruction_text: Mapped[str] = mapped_column(Text, nullable=False)
    request_schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    request_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    query_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    response_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    failure_class: Mapped[str | None] = mapped_column(String(48))
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    requested_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    __mapper_args__ = {"version_id_col": row_version}


class ImageEvidenceProcessingEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_IMAGE_EVIDENCE"
    processing_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    turn_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
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
