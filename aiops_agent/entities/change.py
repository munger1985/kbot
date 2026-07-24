"""AIOps 变更建议、人工审批和执行事实的 SQLAlchemy 映射。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleJSON,
    UniversalTimestamp,
    UUIDv7Type,
)


class ChangeProposalEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_CHANGE_PROPOSAL"

    proposal_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    ops_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    ops_task_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    solution_group_key: Mapped[str] = mapped_column(
        String(128), nullable=False
    )
    command_ordinal: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False
    )
    proposal_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False
    )
    action_type: Mapped[str] = mapped_column(String(32), nullable=False)
    action_template_id: Mapped[str] = mapped_column(
        String(128), nullable=False
    )
    action_template_version: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    action_template_hash: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    renderer_version: Mapped[str] = mapped_column(String(64), nullable=False)
    command_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    parameters_json: Mapped[dict[str, Any]] = mapped_column(
        OracleJSON, nullable=False
    )
    parameters_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    impact_scope_json: Mapped[dict[str, Any] | None] = mapped_column(
        OracleJSON
    )
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    preconditions_json: Mapped[list[dict[str, Any]] | None] = mapped_column(
        OracleJSON
    )
    rollback_plan_json: Mapped[dict[str, Any] | None] = mapped_column(
        OracleJSON
    )
    verification_plan_json: Mapped[dict[str, Any] | None] = mapped_column(
        OracleJSON
    )
    evidence_artifacts_json: Mapped[list[str] | None] = mapped_column(
        OracleJSON
    )
    policy_decision_hash: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    proposal_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    snapshot_artifact_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), nullable=False
    )
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    created_by_task_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), nullable=False
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


class HitlEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_HITL"

    hitl_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    ops_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    ops_task_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    proposal_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    request_type: Mapped[str] = mapped_column(String(32), nullable=False)
    assignee_user_id: Mapped[str] = mapped_column(String(256), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    response_schema_json: Mapped[dict[str, Any] | None] = mapped_column(
        OracleJSON
    )
    input_artifacts_json: Mapped[list[str] | None] = mapped_column(OracleJSON)
    response_json: Mapped[dict[str, Any] | None] = mapped_column(OracleJSON)
    response_uri: Mapped[str | None] = mapped_column(String(2048))
    response_hash: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    requested_by: Mapped[str] = mapped_column(String(256), nullable=False)
    responded_by: Mapped[str | None] = mapped_column(String(256))
    requested_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    responded_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    expires_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    __mapper_args__ = {"version_id_col": row_version}


class ApprovalTokenEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_APPROVAL_TOKEN"

    approval_token_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    proposal_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    hitl_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    nonce_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    approver_id: Mapped[str] = mapped_column(String(256), nullable=False)
    policy_decision_hash: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    target_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False
    )
    parameters_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    issued_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    consumed_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    __mapper_args__ = {"version_id_col": row_version}


class ExecutionEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_EXECUTION"

    execution_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    proposal_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    ops_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    ops_task_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    executor_request_id: Mapped[str] = mapped_column(
        String(128), nullable=False
    )
    proposal_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    action_type: Mapped[str] = mapped_column(String(32), nullable=False)
    action_template_id: Mapped[str] = mapped_column(
        String(128), nullable=False
    )
    action_template_version: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    action_template_hash: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    parameters_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    command_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    execution_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    approval_token_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), nullable=False
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    executor_instance_id: Mapped[str | None] = mapped_column(String(128))
    claimed_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    grant_jti_hash: Mapped[str | None] = mapped_column(String(64))
    status_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    result_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    result_hash: Mapped[str | None] = mapped_column(String(64))
    rollback_of_execution_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
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
