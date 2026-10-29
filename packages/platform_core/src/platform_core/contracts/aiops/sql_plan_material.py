"""历史计划材料的安全摘要；不接收新参数或授权估算。"""

from typing import Literal

from pydantic import Field

from .types import AIOpsContract, UUIDv7, UtcDatetime


class SqlPlanMaterialSource(AIOpsContract):
    source_artifact_id: UUIDv7
    step_id: str = Field(min_length=1, max_length=128)
    row_index: int = Field(ge=0, le=255)
    database_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    subject: dict[str, str | bool | None] = Field(max_length=8)


class SqlPlanCapturePreflight(AIOpsContract):
    schema_version: Literal["SQL_PLAN_CAPTURE_PREFLIGHT.v1"] = "SQL_PLAN_CAPTURE_PREFLIGHT.v1"
    status: Literal["BLOCKED", "AWAITING_DATABASE_VALIDATION"]
    reason_codes: tuple[Literal[
        "DATABASE_TYPE_UNSUPPORTED", "SOURCE_SQL_TEXT_MISSING", "SOURCE_SQL_LENGTH_EXCEEDED",
        "SINGLE_STATEMENT_REQUIRED", "ESTIMATE_SELECT_ONLY", "SOURCE_SQL_COMPLEXITY_EXCEEDED",
        "SIDE_EFFECT_OR_LOCK_FORBIDDEN", "NATIVE_BIND_STYLE_REQUIRED", "NAMED_BIND_MAPPING_REQUIRED",
        "PARAMETER_LAYOUT_INCOMPLETE", "PARAMETER_COUNT_MISMATCH", "SOURCE_SQL_NOT_REPLAYABLE",
    ], ...] = Field(max_length=1)
    expected_parameter_count: int | None = Field(default=None, ge=0, le=64)
    statement_shape: Literal["SELECT", "OTHER", "UNKNOWN"]
    required_checks: tuple[Literal[
        "SOURCE_SQL_REPLAYABILITY", "ORIGINAL_PARSE_CONTEXT", "NATIVE_PARAMETER_TYPES",
        "OBJECT_AND_FUNCTION_ACCESS", "CAPTURE_POLICY", "INDEPENDENT_APPROVAL",
    ], ...] = Field(min_length=6, max_length=6)
    oracle_parameter_peeking: Literal["NOT_PROVEN", "NOT_APPLICABLE"]
    execution_authorized: Literal[False] = False


class SqlPlanMaterialView(AIOpsContract):
    material_id: UUIDv7
    ops_run_id: UUIDv7
    source: SqlPlanMaterialSource
    parameter_count: int = Field(ge=0, le=64)
    created_at: UtcDatetime
    expires_at: UtcDatetime
    state: Literal["STORED", "EXPIRED", "REVOKED", "UNAVAILABLE"]
    capture_preflight: SqlPlanCapturePreflight | None
    execution_authorized: Literal[False] = False
    representative_parameters: Literal["USER_DECLARED_UNVERIFIED"] = "USER_DECLARED_UNVERIFIED"
    environment: Literal["USER_DECLARED_UNVERIFIED"] = "USER_DECLARED_UNVERIFIED"
    notice: str = "历史材料仅可查看安全摘要或撤回清除；不接受新提交，不发起估算或重放业务 SQL。"


class SqlPlanMaterialPage(AIOpsContract):
    items: tuple[SqlPlanMaterialView, ...] = Field(max_length=16)
