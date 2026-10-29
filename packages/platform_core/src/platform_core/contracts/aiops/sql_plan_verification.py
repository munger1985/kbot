"""计划采集内的安全核验结果；不返回原环境值、绑定原值或可猜测值哈希。"""

from typing import Literal, Annotated

from pydantic import Field, model_validator

from .types import AIOpsContract


class SqlPlanVerificationCheck(AIOpsContract):
    field: Literal["DATABASE", "SCHEMA", "ROLE", "SEARCH_PATH", "OPTIMIZER_SETTINGS", "PARAMETER_TYPES", "PARAMETER_VALUES"]
    status: Literal["MATCH", "MISMATCH", "UNKNOWN", "UNSUPPORTED", "NOT_REQUESTED", "NOT_APPLICABLE"]
    source: Literal["CAPTURE_CONNECTION", "SOURCE_SUBJECT_CATALOG", "DIGEST_SCHEMA",
        "NATIVE_CURSOR", "NATIVE_OPTIMIZER_ENV", "NATIVE_BIND_CAPTURE", "UNAVAILABLE"]


class SqlPlanResolvedObject(AIOpsContract):
    node_id: int = Field(ge=0, le=127)
    object_kind: Literal["TABLE", "INDEX"]
    schema_name: str = Field(min_length=1, max_length=128, pattern=r"^[\w$#.-]+$")
    object_name: str = Field(min_length=1, max_length=128, pattern=r"^[\w$#.-]+$")
    native_object_id: str | None = Field(default=None, pattern=r"^[0-9]{1,20}$")
    resolution: Literal["NATIVE_CATALOG", "SQL_ALIAS_TO_CATALOG"]


class SqlPlanNativeVerification(AIOpsContract):
    schema_version: Literal["SQL_PLAN_NATIVE_VERIFICATION.v1"] = "SQL_PLAN_NATIVE_VERIFICATION.v1"
    capture_context: Literal["MATCHED_DECLARATION"] = "MATCHED_DECLARATION"
    original_context: tuple[SqlPlanVerificationCheck, ...] = Field(min_length=5, max_length=5)
    parameter_types: SqlPlanVerificationCheck
    parameter_values: SqlPlanVerificationCheck
    representativeness: Literal["UNPROVEN", "NOT_APPLICABLE"] = "UNPROVEN"
    resolved_objects: tuple[SqlPlanResolvedObject, ...] = Field(default=(), max_length=128)
    unresolved_object_nodes: tuple[Annotated[int, Field(strict=True, ge=0, le=127)], ...] = Field(default=(), max_length=128)
    native_source_rows: int = Field(default=0, ge=0, le=256)
    limitations: tuple[Literal[
        "PARTIAL_CONTEXT_ONLY", "CAPTURED_VALUES_NOT_ONE_EXECUTION", "POPULATION_NOT_OBSERVED",
        "NATIVE_BIND_CAPTURE_UNSUPPORTED", "NATIVE_SOURCE_UNAVAILABLE", "OBJECT_RESOLUTION_INCOMPLETE",
        "ESTIMATE_NOT_PRODUCTION_EXECUTION", "CATALOG_IS_CURRENT_NOT_HISTORICAL",
    ], ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_checks(self):
        if any(check.status in {"MATCH", "MISMATCH"} and check.source == "UNAVAILABLE"
            for check in (*self.original_context, self.parameter_types, self.parameter_values)):
            raise ValueError("明确匹配状态必须有原生来源")
        if {check.field for check in self.original_context} != {"DATABASE", "SCHEMA", "ROLE", "SEARCH_PATH", "OPTIMIZER_SETTINGS"}:
            raise ValueError("原解析环境必须分别保留五类核验状态")
        if self.parameter_types.field != "PARAMETER_TYPES" or self.parameter_values.field != "PARAMETER_VALUES":
            raise ValueError("参数类型与原生值核验不能混用")
        if len(set(item.node_id for item in self.resolved_objects)) != len(self.resolved_objects):
            raise ValueError("计划对象核验不能重复节点")
        if (len(set(self.unresolved_object_nodes)) != len(self.unresolved_object_nodes)
            or set(self.unresolved_object_nodes) & {item.node_id for item in self.resolved_objects}):
            raise ValueError("已解析和未解析对象必须分开")
        if self.parameter_values.status == "MATCH" and self.parameter_values.source != "NATIVE_BIND_CAPTURE":
            raise ValueError("原生值匹配不能由用户声明或新计划类型推断")
        return self
