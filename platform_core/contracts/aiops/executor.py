"""AIOps Worker 与高权限 DB Executor 的隔离契约。"""

from __future__ import annotations

from pydantic import Field

from .types import (
    AIOpsContract,
    EXECUTOR_SCHEMA_VERSION,
    ExecutionStatus,
    JsonObject,
    ResourceRef,
    SecretRef,
    Sha256Digest,
    UUIDv7,
    UtcDatetime,
)


class DiagnosticExecutionGrant(AIOpsContract):
    schema_version: str = EXECUTOR_SCHEMA_VERSION
    grant_id: UUIDv7
    task_id: UUIDv7
    lease_token_hash: Sha256Digest
    target_id: UUIDv7
    target_row_version: int = Field(ge=1)
    connection_profile: JsonObject
    secret_ref: SecretRef
    diagnostic_tool_id: str = Field(min_length=1, max_length=128)
    diagnostic_tool_version: str = Field(min_length=1, max_length=64)
    template_hash: Sha256Digest
    max_result_rows: int = Field(gt=0)
    statement_timeout_seconds: int = Field(gt=0)
    expires_at: UtcDatetime


class ReadDiagnosticRequest(AIOpsContract):
    schema_version: str = EXECUTOR_SCHEMA_VERSION
    executor_request_id: UUIDv7
    idempotency_key: str = Field(min_length=1, max_length=256)
    diagnostic_tool_id: str = Field(min_length=1, max_length=128)
    diagnostic_tool_version: str = Field(min_length=1, max_length=64)
    typed_parameters: JsonObject
    grant: DiagnosticExecutionGrant


class ReadDiagnosticResult(AIOpsContract):
    schema_version: str = EXECUTOR_SCHEMA_VERSION
    executor_request_id: UUIDv7
    status: ExecutionStatus
    result_ref: ResourceRef | None = None
    result_hash: Sha256Digest | None = None
    row_count: int | None = Field(default=None, ge=0)
    truncated: bool = False
    error_code: str | None = Field(default=None, max_length=128)


class MutationExecutionRequest(AIOpsContract):
    schema_version: str = EXECUTOR_SCHEMA_VERSION
    execution_id: UUIDv7
    executor_request_id: UUIDv7
    idempotency_key: str = Field(min_length=1, max_length=256)


class MutationClaimGrant(AIOpsContract):
    schema_version: str = EXECUTOR_SCHEMA_VERSION
    execution_id: UUIDv7
    executor_request_id: UUIDv7
    executor_instance_id: str = Field(min_length=1, max_length=256)
    target_id: UUIDv7
    connection_profile: JsonObject
    secret_ref: SecretRef
    action_template_id: str = Field(min_length=1, max_length=128)
    action_template_version: str = Field(min_length=1, max_length=64)
    typed_parameters: JsonObject
    template_hash: Sha256Digest
    proposal_hash: Sha256Digest
    policy_hash: Sha256Digest
    approval_token_hash: Sha256Digest
    statement_timeout_seconds: int = Field(gt=0)
    expires_at: UtcDatetime


class ExecutionStatusEvent(AIOpsContract):
    schema_version: str = EXECUTOR_SCHEMA_VERSION
    event_id: UUIDv7
    executor_request_id: UUIDv7
    execution_id: UUIDv7 | None = None
    status_version: int = Field(ge=1)
    status: ExecutionStatus
    occurred_at: UtcDatetime
    result_ref: ResourceRef | None = None
    error_code: str | None = Field(default=None, max_length=128)


class ExecutionResultRef(AIOpsContract):
    schema_version: str = EXECUTOR_SCHEMA_VERSION
    executor_request_id: UUIDv7
    status: ExecutionStatus
    result_ref: ResourceRef | None = None
    result_hash: Sha256Digest | None = None
