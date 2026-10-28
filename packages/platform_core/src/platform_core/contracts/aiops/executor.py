"""AIOps Worker 与隔离 DB Executor 的严格 Wire 契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .investigation import DiagnosticPagination

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
    tls_profile_ref: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )

    @model_validator(mode="after")
    def validate_tls_profile(self) -> "DiagnosticConnectionProfile":
        if self.tls_profile_ref is not None and not self.tls_enabled:
            raise ValueError("TLS Profile只能在启用TLS时配置")
        return self


class DiagnosticLimits(AIOpsContract):
    statement_timeout_seconds: int = Field(gt=0, le=3600)
    max_result_rows: int = Field(gt=0, le=100000)
    max_result_bytes: int = Field(gt=0, le=100 * 1024 * 1024)
    max_columns: int = Field(default=128, gt=0, le=1024)
    max_cell_chars: int = Field(default=32768, gt=0, le=1_000_000)
    pagination: DiagnosticPagination | None = None

    @model_validator(mode="after")
    def validate_pagination_limit(self):
        if self.pagination is not None and self.max_result_rows > self.pagination.page_size * self.pagination.max_pages:
            raise ValueError("游标分页不能扩大总行数预算")
        return self


class DiagnosticPaginationObservation(AIOpsContract):
    """分批取回描述查询结果的读取范围，不声明源数据全量或历史窗口。"""

    mode: Literal["SINGLE_EXECUTION_CURSOR"] = "SINGLE_EXECUTION_CURSOR"
    requested: DiagnosticPagination
    returned_page_rows: tuple[int, ...] = Field(max_length=4)
    stop_reason: Literal["EXHAUSTED", "ROW_LIMIT", "BYTE_LIMIT", "CELL_LIMIT"]
    source_completeness: Literal["UNKNOWN"] = "UNKNOWN"

    @model_validator(mode="after")
    def validate_page_counts(self):
        if len(self.returned_page_rows) > self.requested.max_pages or any(type(count) is not int or not 1 <= count <= self.requested.page_size for count in self.returned_page_rows):
            raise ValueError("游标分页的实际页数或行数超过冻结上限")
        if any(count != self.requested.page_size for count in self.returned_page_rows[:-1]):
            raise ValueError("游标分页仅最后一页可不足一页")
        return self


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


class OracleDynamicQueryPolicyGrant(AIOpsContract):
    """随 Grant 冻结并由 Executor 重放的 Oracle 动态查询策略。"""

    schema_version: Literal["ORACLE_DYNAMIC_QUERY_POLICY.v4"] = (
        "ORACLE_DYNAMIC_QUERY_POLICY.v4"
    )
    allowed_objects: tuple[str, ...] = ()
    allowed_functions: tuple[str, ...]
    allowed_packages: tuple[str, ...]
    max_rows: int = Field(ge=1, le=1000)
    max_sql_chars: int = Field(ge=1, le=100_000)
    max_bind_count: int = Field(ge=0, le=128)
    allow_catalog_object_families: bool
    allow_diagnostics_pack: bool = False
    allow_tuning_pack: bool = False


class PostgreSQLDynamicQueryPolicyGrant(AIOpsContract):
    """随 Grant 冻结并由 Executor 重放的 PostgreSQL 动态查询策略。"""

    schema_version: Literal["POSTGRESQL_DYNAMIC_QUERY_POLICY.v1"] = (
        "POSTGRESQL_DYNAMIC_QUERY_POLICY.v1"
    )
    allowed_objects: tuple[str, ...] = ()
    allowed_functions: tuple[str, ...]
    allowed_schemas: tuple[str, ...]
    max_rows: int = Field(ge=1, le=1000)
    max_sql_chars: int = Field(ge=1, le=100_000)
    max_bind_count: int = Field(ge=0, le=128)


class MySQLDynamicQueryPolicyGrant(AIOpsContract):
    """MySQL 目录查询策略随签名 Grant 冻结。"""

    schema_version: Literal["MYSQL_DYNAMIC_QUERY_POLICY.v1"] = "MYSQL_DYNAMIC_QUERY_POLICY.v1"
    allowed_objects: tuple[str, ...] = ()
    allowed_functions: tuple[str, ...]
    allowed_schemas: tuple[str, ...]
    max_rows: int = Field(ge=1, le=1000)
    max_sql_chars: int = Field(ge=1, le=100_000)
    max_bind_count: int = Field(ge=0, le=128)


class DynamicDiagnosticExecutionGrant(AIOpsContract):
    """数据库动态只读查询的短期签名执行授权。"""

    schema_version: Literal["dynamic-diagnostic-execution-grant.v1"] = (
        "dynamic-diagnostic-execution-grant.v1"
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
    db_type: Literal["ORACLE", "POSTGRESQL", "MYSQL"]
    connection_profile: DiagnosticConnectionProfile
    diagnostic_credential_id: UUIDv7
    tool_id: Literal[
        "db.oracle.readonly_query",
        "db.postgresql.readonly_query",
        "db.mysql.readonly_query",
    ]
    tool_version: Literal["1.0.0"] = "1.0.0"
    variant: Literal[
        "oracle-dynamic-readonly-v1",
        "postgresql-dynamic-readonly-v1",
        "mysql-dynamic-readonly-v1",
    ]
    query_sha256: Sha256Digest
    policy_sha256: Sha256Digest
    policy_snapshot: (
        OracleDynamicQueryPolicyGrant | PostgreSQLDynamicQueryPolicyGrant | MySQLDynamicQueryPolicyGrant
    )
    parameters_sha256: Sha256Digest
    projected_columns: tuple[str, ...] = Field(min_length=1, max_length=128)
    capability_snapshot_hash: Sha256Digest
    limits: DiagnosticLimits
    trace_id: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_database_contract(self) -> "DynamicDiagnosticExecutionGrant":
        expected = {
            "MYSQL": (
                "db.mysql.readonly_query",
                "mysql-dynamic-readonly-v1",
                "MYSQL_DYNAMIC_QUERY_POLICY.v1",
            ),
            "ORACLE": (
                "db.oracle.readonly_query",
                "oracle-dynamic-readonly-v1",
                "ORACLE_DYNAMIC_QUERY_POLICY.v4",
            ),
            "POSTGRESQL": (
                "db.postgresql.readonly_query",
                "postgresql-dynamic-readonly-v1",
                "POSTGRESQL_DYNAMIC_QUERY_POLICY.v1",
            ),
        }[self.db_type]
        if (
            self.tool_id,
            self.variant,
            self.policy_snapshot.schema_version,
        ) != expected:
            raise ValueError("动态查询 Grant 的数据库、工具和策略不一致")
        return self


class ReadDiagnosticRequest(AIOpsContract):
    schema_version: Literal["diagnostic-execution-request.v1"] = (
        "diagnostic-execution-request.v1"
    )
    executor_request_id: UUIDv7
    grant: str = Field(min_length=64, max_length=16384)
    parameters: JsonObject
    idempotency_key: str = Field(min_length=1, max_length=256)


class DynamicReadDiagnosticRequest(AIOpsContract):
    """动态 SQL 只在专用入口出现，不能混入固定目录请求。"""

    schema_version: Literal["dynamic-diagnostic-execution-request.v1"] = (
        "dynamic-diagnostic-execution-request.v1"
    )
    executor_request_id: UUIDv7
    grant: str = Field(min_length=64, max_length=32768)
    sql: str = Field(min_length=1, max_length=100_000)
    parameters: JsonObject
    idempotency_key: str = Field(min_length=1, max_length=256)


class DatabaseColumn(AIOpsContract):
    name: str = Field(min_length=1, max_length=128)
    logical_type: Literal[
        "STRING", "INTEGER", "DECIMAL", "BOOLEAN", "DATETIME", "NULL"
    ]
    sensitivity: Literal["PUBLIC", "MASKED", "HASHED"]


class ObjectAccessObservation(AIOpsContract):
    """零行 SELECT 的历史访问观察，不授予权限，也不证明业务行可见。"""

    schema_version: Literal["OBJECT_ACCESS_OBSERVATION.v1"] = "OBJECT_ACCESS_OBSERVATION.v1"
    scope: Literal["OBJECT_AND_EXPLICIT_COLUMNS"] = "OBJECT_AND_EXPLICIT_COLUMNS"
    grants_authority: Literal[False] = False
    db_type: Literal["POSTGRESQL", "ORACLE", "MYSQL"]
    object_ref: str = Field(min_length=1, max_length=256)
    read_columns: tuple[str, ...] = Field(min_length=1, max_length=128)
    state: Literal["READABLE", "DENIED", "OBJECT_UNAVAILABLE"]
    checked_at: UtcDatetime
    expires_at: UtcDatetime
    source_executor_request_id: UUIDv7
    source_query_sha256: Sha256Digest
    reused: bool = False

    @model_validator(mode="after")
    def validate_scope(self):
        if self.expires_at <= self.checked_at:
            raise ValueError("访问观察有效期必须晚于采集时间")
        if (self.expires_at - self.checked_at).total_seconds() > (300 if self.state == "READABLE" else 30):
            raise ValueError("访问观察有效期超过状态上限")
        if (len(set(self.read_columns)) != len(self.read_columns)
                or any(not column or len(column) > 256 or column == "*" for column in self.read_columns)):
            raise ValueError("访问观察必须声明有界且不重复的显式列")
        return self


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
    object_access: tuple[ObjectAccessObservation, ...] = Field(default=(), max_length=1)
    pagination: DiagnosticPaginationObservation | None = None

    @model_validator(mode="after")
    def validate_pagination_observation(self):
        if self.pagination is not None:
            if sum(self.pagination.returned_page_rows) != self.row_count or self.row_count != len(self.rows):
                raise ValueError("游标分页摘要与原始结果行数不一致")
            if self.truncated != (self.pagination.stop_reason != "EXHAUSTED"):
                raise ValueError("游标分页不能将超限结果标记为完整读取")
        return self


class ReadDiagnosticResult(AIOpsContract):
    schema_version: Literal["diagnostic-execution-result.v1"] = (
        "diagnostic-execution-result.v1"
    )
    executor_request_id: UUIDv7
    status: Literal["SUCCEEDED", "GAP"]
    observation: DatabaseObservation | None = None
    error_code: str | None = Field(default=None, max_length=128)
    retryable: bool = False
    object_access: tuple[ObjectAccessObservation, ...] = Field(default=(), max_length=1)


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


class ResourceLeaseReceipt(AIOpsContract):
    """执行租约的释放围栏；只有执行端收到，不能作为用户证据。"""
    lease_id: UUIDv7
    release_token: str = Field(min_length=40, max_length=64, repr=False)
    execution_deadline_at: UtcDatetime
    held_until: UtcDatetime

    @model_validator(mode="after")
    def validate_lease_window(self):
        if self.held_until <= self.execution_deadline_at:
            raise ValueError("资源租约必须包含清理保留时间")
        return self


class ResourceReleaseReceipt(AIOpsContract):
    released: Literal[True] = True


class ReadonlyExecutionFenceRequest(AIOpsContract):
    """执行端携原始签名与秘密回执核对当前授权，不接受自报 Task 身份。"""
    grant: str = Field(min_length=64, max_length=32768, repr=False)
    resource_lease: ResourceLeaseReceipt = Field(repr=False)


class ReadonlyExecutionFenceReceipt(AIOpsContract):
    allowed: bool
    reason: Literal["EXECUTION_GRANT_INVALID", "EXECUTION_AUTHORIZATION_REVOKED",
        "EXECUTION_CANCELLED", "EXECUTION_LEASE_INVALID", "EXECUTION_DEADLINE_EXCEEDED"] | None = None
    checked_at: UtcDatetime
    execution_deadline_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def validate_execution_fence(self):
        if self.allowed:
            if self.reason is not None or self.execution_deadline_at is None or self.execution_deadline_at <= self.checked_at:
                raise ValueError("有效执行围栏必须有未来期限且不能包含拒绝原因")
        elif self.reason is None or self.execution_deadline_at is not None:
            raise ValueError("拒绝执行必须包含原因且不能提供新的有效期限")
        return self


class CredentialIssueResponse(AIOpsContract):
    username: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=4096)
    resource_lease: ResourceLeaseReceipt | None = None


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
