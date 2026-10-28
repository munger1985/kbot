"""计划采集的独立审批合同；审批不是执行 Grant，也不授权业务 SQL。"""

from typing import Literal

from pydantic import Field

from .sql_plan_material import SqlPlanMaterialSource
from .types import AIOpsContract, UUIDv7, UtcDatetime


class SqlPlanCapturePolicy(AIOpsContract):
    enabled: bool = Field(default=False, strict=True)
    allowed_objects: tuple[str, ...] = Field(default=(), max_length=64)
    require_distinct_approver: bool = Field(default=True, strict=True)
    max_statement_timeout_seconds: int = Field(default=5, ge=1, le=10, strict=True)


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
    requested_by: str = Field(min_length=1, max_length=128)
    decided_by: str | None = Field(default=None, max_length=128)
    created_at: UtcDatetime
    expires_at: UtcDatetime
    execution_authorized: Literal[False] = False
    capture_scheduled: Literal[False] = False
    notice: str = "仅记录独立风险审批，尚未调度计划采集；不执行 ANALYZE 或业务 SQL。"


class SqlPlanCapturePage(AIOpsContract):
    items: tuple[SqlPlanCaptureView, ...] = Field(max_length=16)
    can_request: bool
    can_review: bool
    availability: Literal["AVAILABLE", "DISABLED", "POLICY_REQUIRED", "ACCESS_CHANGED"]
    max_statement_timeout_seconds: int | None = Field(default=None, ge=1, le=10, strict=True)
