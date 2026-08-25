"""AIOps Worker 与隔离 DB Executor 的严格 Wire 契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .types import (
    AIOpsContract,
    EXECUTOR_SCHEMA_VERSION,
    ExecutionStatus,
    JsonObject,
    ResourceRef,
    Sha256Digest,
    UUIDv7,
    UtcDatetime,
)


class DiagnosticConnectionProfile(AIOpsContract):
    """不含账号和密码的目标数据库连接信息。"""

    host: str = Field(
        min_length=1,
        max_length=253,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9.-]*$",
    )
    port: int = Field(ge=1, le=65535)
    service: str | None = Field(default=None, min_length=1, max_length=256)
    database: str | None = Field(default=None, min_length=1, max_length=256)
    tls_enabled: bool = True
    tls_profile_ref: str | None = Field(default=None, max_length=2048)


class DiagnosticLimits(AIOpsContract):
    statement_timeout_seconds: int = Field(gt=0, le=3600)
    max_result_rows: int = Field(gt=0, le=100000)
    max_result_bytes: int = Field(gt=0, le=100 * 1024 * 1024)
    max_columns: int = Field(default=128, gt=0, le=1024)
    max_cell_chars: int = Field(default=32768, gt=0, le=1_000_000)


class DiagnosticExecutionGrant(AIOpsContract):
    """JWS 内部载荷；不会作为未签名 JSON 接受。"""

    schema_version: Literal["diagnostic-execution-grant.v1"] = (
        "diagnostic-execution-grant.v1"
    )
    issuer: str = Field(min_length=1, max_length=128)
    audience: str = Field(min_length=1, max_length=128)
    grant_id: UUIDv7
    issued_at: UtcDatetime
    expires_at: UtcDatetime
    run_id: UUIDv7
    task_id: UUIDv7
    lease_token_hash: Sha256Digest
    target_id: UUIDv7
    domain_id: int = Field(ge=1)
    target_row_version: int = Field(ge=1)
    db_type: Literal["POSTGRESQL", "ORACLE", "MYSQL"]
    connection_profile: DiagnosticConnectionProfile
    diagnostic_credential_id: UUIDv7
    tool_id: str = Field(pattern=r"^db\.[a-z0-9_.-]{1,124}$")
    tool_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    variant: str = Field(min_length=1, max_length=128)
    template_sha256: Sha256Digest
    parameters_sha256: Sha256Digest
    capability_snapshot_hash: Sha256Digest
    limits: DiagnosticLimits
    trace_id: str = Field(min_length=1, max_length=128)


class ReadDiagnosticRequest(AIOpsContract):
    schema_version: Literal["diagnostic-execution-request.v1"] = (
        "diagnostic-execution-request.v1"
    )
    executor_request_id: UUIDv7
    grant: str = Field(min_length=64, max_length=16384)
    parameters: JsonObject
    idempotency_key: str = Field(min_length=1, max_length=256)


class DatabaseColumn(AIOpsContract):
    name: str = Field(min_length=1, max_length=128)
    logical_type: Literal[
        "STRING", "INTEGER", "DECIMAL", "BOOLEAN", "DATETIME", "NULL"
    ]
    sensitivity: Literal["PUBLIC", "MASKED", "HASHED"]


class DatabaseObservation(AIOpsContract):
    schema_version: Literal["DATABASE_OBSERVATION.v1"] = (
        "DATABASE_OBSERVATION.v1"
    )
    executor_request_id: UUIDv7
    target_id: UUIDv7
    tool_id: str
    tool_version: str
    variant: str
    template_sha256: Sha256Digest
    db_type: Literal["POSTGRESQL", "ORACLE", "MYSQL"]
    db_version: str = Field(min_length=1, max_length=128)
    capability_snapshot_hash: Sha256Digest
    captured_at: UtcDatetime
    duration_ms: int = Field(ge=0)
    columns: tuple[DatabaseColumn, ...]
    rows: tuple[tuple[Any, ...], ...]
    row_count: int = Field(ge=0)
    truncated: bool
    result_sha256: Sha256Digest
    parameters_sha256: Sha256Digest
    warnings: tuple[str, ...] = ()
    provenance: JsonObject = Field(default_factory=dict)


class ReadDiagnosticResult(AIOpsContract):
    schema_version: Literal["diagnostic-execution-result.v1"] = (
        "diagnostic-execution-result.v1"
    )
    executor_request_id: UUIDv7
    status: Literal["SUCCEEDED", "GAP"]
    observation: DatabaseObservation | None = None
    error_code: str | None = Field(default=None, max_length=128)
    retryable: bool = False


class MutationExecutionRequest(AIOpsContract):
    schema_version: str = EXECUTOR_SCHEMA_VERSION
    execution_id: UUIDv7
    executor_request_id: UUIDv7
    idempotency_key: str = Field(min_length=1, max_length=256)


class MutationClaimRequest(AIOpsContract):
    schema_version: str = EXECUTOR_SCHEMA_VERSION
    executor_request_id: UUIDv7
    executor_instance_id: str = Field(min_length=1, max_length=256)
    action_catalog_hash: Sha256Digest


class MutationExecutionGrant(AIOpsContract):
    schema_version: str = EXECUTOR_SCHEMA_VERSION
    issuer: str = Field(min_length=1, max_length=128)
    audience: str = Field(min_length=1, max_length=128)
    grant_id: UUIDv7
    issued_at: UtcDatetime
    expires_at: UtcDatetime
    execution_id: UUIDv7
    executor_request_id: UUIDv7
    executor_instance_id: str = Field(min_length=1, max_length=256)
    target_id: UUIDv7
    domain_id: int = Field(ge=1)
    target_version: int = Field(ge=1)
    db_type: Literal["POSTGRESQL", "ORACLE", "MYSQL"]
    connection_profile: JsonObject
    execution_credential_id: UUIDv7
    action_template_id: str = Field(min_length=1, max_length=128)
    action_template_version: str = Field(min_length=1, max_length=64)
    action_template_variant: str = Field(min_length=1, max_length=128)
    renderer_version: str = Field(min_length=1, max_length=64)
    typed_parameters: JsonObject
    action_template_hash: Sha256Digest
    parameters_hash: Sha256Digest
    command_hash: Sha256Digest
    proposal_hash: Sha256Digest
    policy_decision_hash: Sha256Digest
    approval_token_hash: Sha256Digest
    approver_id: str = Field(min_length=1, max_length=256)
    action_catalog_hash: Sha256Digest
    statement_timeout_seconds: int = Field(gt=0)
    max_database_attempts: Literal[1] = 1
    trace_id: str = Field(min_length=1, max_length=128)


class CredentialIssueRequest(AIOpsContract):
    """Executor 用签名 Grant 换取一次性连接材料。"""
    grant: str = Field(min_length=64, max_length=32768)


class CredentialIssueResponse(AIOpsContract):
    username: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=4096)


class MutationClaimReceipt(AIOpsContract):
    schema_version: str = EXECUTOR_SCHEMA_VERSION
    execution_id: UUIDv7
    executor_request_id: UUIDv7
    status: Literal["SUBMITTED"]
    grant: str = Field(min_length=64, max_length=32768)
    expires_at: UtcDatetime


class ExecutionStatusEvent(AIOpsContract):
    schema_version: str = EXECUTOR_SCHEMA_VERSION
    event_id: UUIDv7
    executor_request_id: UUIDv7
    execution_id: UUIDv7
    executor_instance_id: str = Field(min_length=1, max_length=256)
    grant_jti_hash: Sha256Digest
    status_version: int = Field(ge=3)
    status: Literal["RUNNING", "SUCCEEDED", "FAILED", "UNKNOWN"]
    occurred_at: UtcDatetime
    bounded_result: JsonObject | None = None
    result_hash: Sha256Digest | None = None
    error_code: str | None = Field(default=None, max_length=128)
    retryable: Literal[False] = False

    @model_validator(mode="after")
    def validate_status_payload(self) -> "ExecutionStatusEvent":
        terminal = self.status in {"SUCCEEDED", "FAILED", "UNKNOWN"}
        if self.status == "RUNNING" and (
            self.bounded_result is not None
            or self.result_hash is not None
            or self.error_code is not None
        ):
            raise ValueError("RUNNING 事件不能携带终态结果")
        if terminal and self.result_hash is None:
            raise ValueError("终态事件必须携带结果 Hash")
        return self


class ExecutionResultRef(AIOpsContract):
    schema_version: str = EXECUTOR_SCHEMA_VERSION
    executor_request_id: UUIDv7
    status: ExecutionStatus
    result_ref: ResourceRef | None = None
    result_hash: Sha256Digest | None = None
