"""步骤 8 HITL Artifact 与 Worker 暂停信号。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from platform_core.contracts.aiops.types import UtcDatetime


class _HitlContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ManualDiagnosticQuery(_HitlContract):
    query_id: str = Field(min_length=1, max_length=256)
    origin: Literal["CATALOG", "MODEL_GENERATED"]
    purpose: str = Field(min_length=1, max_length=1000)
    diagnostic_question: str = Field(min_length=1, max_length=1000)
    sql_text: str = Field(min_length=1, max_length=20000)
    sql_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    expected_columns: tuple[str, ...]
    expected_types: tuple[str, ...]
    expected_shape: Literal["SINGLE_ROW", "ROW_SET"]
    required: bool = True
    max_rows: int = Field(ge=1, le=5000)
    timeout_hint_seconds: int = Field(ge=1, le=600)
    cost_warning: str = Field(min_length=1, max_length=1000)
    sensitivity_labels: tuple[str, ...] = ()
    supports_if: str = Field(min_length=1, max_length=1000)
    contradicts_if: str = Field(min_length=1, max_length=1000)


class ManualSqlRequest(_HitlContract):
    schema_version: Literal["MANUAL_SQL_REQUEST.v1"] = (
        "MANUAL_SQL_REQUEST.v1"
    )
    hitl_id: str
    run_id: str
    round_no: int = Field(ge=1, le=10)
    target_id: str
    target_display_name: str
    db_type: Literal["POSTGRESQL", "ORACLE", "MYSQL"]
    db_version: str
    expected_instance_identity: dict[str, str]
    evidence_gap_refs: tuple[str, ...] = ()
    hypothesis_keys: tuple[str, ...] = ()
    queries: tuple[ManualDiagnosticQuery, ...] = Field(
        min_length=1, max_length=3
    )
    instructions: tuple[str, ...]
    parser_version: Literal["manual-result.v1"] = "manual-result.v1"
    expires_at: UtcDatetime


class DataRequiredRequest(_HitlContract):
    schema_version: Literal["DATA_REQUEST.v1"] = "DATA_REQUEST.v1"
    hitl_id: str
    run_id: str
    round_no: int = Field(ge=1, le=10)
    target_id: str
    fields: tuple[dict[str, Any], ...] = Field(min_length=1, max_length=12)
    expires_at: UtcDatetime


class DiagnosticQueryApprovalRequest(_HitlContract):
    """用户只授权执行一条已经冻结的动态只读查询。"""

    schema_version: Literal["DIAGNOSTIC_QUERY_APPROVAL_REQUEST.v1"] = (
        "DIAGNOSTIC_QUERY_APPROVAL_REQUEST.v1"
    )
    hitl_id: str
    run_id: str
    task_id: str
    target_id: str
    target_display_name: str
    purpose: str = Field(min_length=1, max_length=2000)
    sql_text: str = Field(min_length=1, max_length=20000)
    query_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    policy_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    referenced_objects: tuple[str, ...] = Field(min_length=1)
    projected_columns: tuple[str, ...] = Field(min_length=1)
    column_sensitivities: tuple[
        Literal["PUBLIC", "MASKED", "HASHED"], ...
    ] = Field(min_length=1)
    parameters: dict[str, str | int | float | bool | None]
    reason_codes: tuple[str, ...] = Field(min_length=1)
    max_rows: int = Field(ge=1, le=1000)
    timeout_seconds: int = Field(ge=1, le=600)
    max_result_bytes: int = Field(ge=1, le=100 * 1024 * 1024)
    expires_at: UtcDatetime

    @model_validator(mode="after")
    def validate_columns(self) -> "DiagnosticQueryApprovalRequest":
        if len(self.projected_columns) != len(self.column_sensitivities):
            raise ValueError("查询投影列与敏感等级数量必须一致")
        return self


class InputSuspension(_HitlContract):
    """Handler 返回该对象时，Worker 改为原子挂起 Task。"""

    schema_version: Literal["INPUT_SUSPENSION.v1"] = "INPUT_SUSPENSION.v1"
    hitl_id: str
    request_type: Literal[
        "DATA_REQUIRED",
        "MANUAL_DIAGNOSTIC_SQL",
        "DIAGNOSTIC_QUERY_APPROVAL",
    ]
    assignee_user_id: str
    prompt_text: str
    response_schema: dict[str, Any]
    request_artifact_type: str
    request_schema_version: str
    request_payload: dict[str, Any]
    expires_at: UtcDatetime
    idempotency_key: str


class HitlOutcome(_HitlContract):
    schema_version: Literal["HITL_OUTCOME.v1"] = "HITL_OUTCOME.v1"
    hitl_id: str | None = None
    status: Literal[
        "NOT_REQUIRED", "ANSWERED", "SKIPPED", "EXPIRED"
    ]
    evidence_artifact_ids: tuple[str, ...] = ()
    gap_code: str | None = None
    submission: dict[str, Any] | None = None


class ManualSqlCandidate(_HitlContract):
    schema_version: Literal["MANUAL_SQL_CANDIDATE.v1"] = (
        "MANUAL_SQL_CANDIDATE.v1"
    )
    db_type: Literal["POSTGRESQL", "ORACLE", "MYSQL"]
    purpose: str = Field(min_length=1, max_length=1000)
    sql_text: str = Field(min_length=1, max_length=20000)
    expected_columns: tuple[str, ...] = Field(min_length=1, max_length=64)


class UserProvidedDatabaseResult(_HitlContract):
    schema_version: Literal["USER_PROVIDED_DB_RESULT.v1"] = (
        "USER_PROVIDED_DB_RESULT.v1"
    )
    hitl_id: str
    query_id: str
    status: Literal["SUCCEEDED", "FAILED", "SKIPPED"]
    raw_output: str | None = Field(default=None, max_length=65536)
    parse_status: Literal["STRUCTURED", "UNSTRUCTURED", "NOT_APPLICABLE"]
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    parse_warning: str | None = Field(default=None, max_length=1000)
    error: str | None = None
    content_sha256: str
    quality_flags: tuple[str, ...] = ("USER_PROVIDED",)

    @model_validator(mode="after")
    def validate_result(self) -> "UserProvidedDatabaseResult":
        if self.status == "SUCCEEDED" and not self.raw_output:
            raise ValueError("成功结果必须保留原始数据库输出")
        if self.parse_status == "STRUCTURED" and not self.columns:
            raise ValueError("结构化结果必须包含列定义")
        if self.parse_status != "STRUCTURED" and (self.columns or self.rows):
            raise ValueError("未结构化结果不能包含行列数据")
        if self.status != "SUCCEEDED" and (
            self.raw_output or self.parse_status != "NOT_APPLICABLE"
        ):
            raise ValueError("失败或跳过结果不能包含数据库输出")
        return self


class UserDiagnosticSubmission(_HitlContract):
    schema_version: Literal["USER_DIAGNOSTIC_SUBMISSION.v1"] = (
        "USER_DIAGNOSTIC_SUBMISSION.v1"
    )
    hitl_id: str
    submitted_by: str
    submitted_at: UtcDatetime
    target_display_name: str
    used_readonly_account: bool
    note: str | None = Field(default=None, max_length=2000)
    results: tuple[UserProvidedDatabaseResult, ...]
    submission_sha256: str
