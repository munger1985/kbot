"""计划采集的独立审批合同；审批不是执行 Grant，也不授权业务 SQL。"""

from typing import Literal

from pydantic import Field, model_validator

from .sql_plan_material import SqlPlanMaterialSource
from .types import AIOpsContract, UUIDv7, UtcDatetime


class SqlPlanCapturePolicy(AIOpsContract):
    enabled: bool = Field(default=False, strict=True)
    allowed_objects: tuple[str, ...] = Field(default=(), max_length=64)
    require_distinct_approver: bool = Field(default=True, strict=True)
    native_bind_comparison_enabled: bool = Field(default=False, strict=True)
    max_statement_timeout_seconds: int = Field(default=5, ge=1, le=10, strict=True)

    @model_validator(mode="after")
    def validate_object_scope(self):
        if (len(set(self.allowed_objects)) != len(self.allowed_objects)
            or any(not name or name != name.strip() or len(name) > 256
                or any(ord(char) < 32 for char in name) for name in self.allowed_objects)):
            raise ValueError("计划采集对象范围必须为不重复的有效对象名称")
        return self


class SqlPlanCaptureRequest(AIOpsContract):
    request_id: UUIDv7
    material_id: UUIDv7
    mode: Literal["ESTIMATE_ONLY"] = "ESTIMATE_ONLY"
    statement_timeout_seconds: int = Field(default=5, ge=1, le=10, strict=True)


class SqlPlanCaptureDecision(AIOpsContract):
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["APPROVE", "REJECT"]
    acknowledge_estimate_only: bool = Field(default=False, strict=True)
    acknowledge_oracle_plan_table_write: bool = Field(default=False, strict=True)
    acknowledge_native_bind_comparison: bool = Field(default=False, strict=True)


class SqlPlanCaptureView(AIOpsContract):
    request_id: UUIDv7
    ops_run_id: UUIDv7
    material_id: UUIDv7
    source: SqlPlanMaterialSource
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: Literal["PENDING", "APPROVED", "REJECTED", "REVOKED", "EXPIRED", "UNAVAILABLE"]
    mode: Literal["ESTIMATE_ONLY"] = "ESTIMATE_ONLY"
    allowed_objects: tuple[str, ...] = Field(max_length=64)
    statement_timeout_seconds: int = Field(ge=1, le=10)
    require_distinct_approver: bool
    oracle_plan_table_write: bool
    native_bind_comparison: bool = False
    requested_by: str = Field(min_length=1, max_length=128)
    decided_by: str | None = Field(default=None, max_length=128)
    created_at: UtcDatetime
    expires_at: UtcDatetime
    execution_authorized: Literal[False] = False
    capture_scheduled: bool = False
    capture_run_id: UUIDv7 | None = None
    notice: str = "审批回执不授予执行权；采集须由独立后台任务兑换当前授权，不执行 ANALYZE 或业务 SQL。"

    @model_validator(mode="after")
    def validate_scheduling(self):
        if self.capture_scheduled != (self.capture_run_id is not None):
            raise ValueError("采集调度状态与后台运行引用不一致")
        return self


class SqlPlanCapturePage(AIOpsContract):
    items: tuple[SqlPlanCaptureView, ...] = Field(max_length=16)
    can_request: bool
    can_review: bool
    availability: Literal["AVAILABLE", "DISABLED", "POLICY_REQUIRED", "ACCESS_CHANGED"]
    max_statement_timeout_seconds: int | None = Field(default=None, ge=1, le=10, strict=True)
