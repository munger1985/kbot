"""AIOps 稳定错误码与 Problem Details。"""

from enum import StrEnum

from pydantic import Field

from .types import AIOpsContract, PUBLIC_SCHEMA_VERSION


class AIOpsErrorCode(StrEnum):
    OPS_NOT_FOUND_OR_DENIED = "OPS_NOT_FOUND_OR_DENIED"
    OPS_IDEMPOTENCY_CONFLICT = "OPS_IDEMPOTENCY_CONFLICT"
    OPS_ROW_VERSION_CHANGED = "OPS_ROW_VERSION_CHANGED"
    OPS_STATE_CONFLICT = "OPS_STATE_CONFLICT"
    OPS_POLICY_DENIED = "OPS_POLICY_DENIED"
    OPS_HITL_EXPIRED = "OPS_HITL_EXPIRED"
    OPS_APPROVAL_INVALID = "OPS_APPROVAL_INVALID"
    OPS_EXECUTION_UNKNOWN = "OPS_EXECUTION_UNKNOWN"
    OPS_UPSTREAM_UNAVAILABLE = "OPS_UPSTREAM_UNAVAILABLE"


class FieldError(AIOpsContract):
    field: str
    code: str
    message: str


class ProblemDetails(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    type: str
    title: str
    status: int = Field(ge=400, le=599)
    code: AIOpsErrorCode
    detail: str
    request_id: str
    trace_id: str
    retryable: bool = False
    field_errors: tuple[FieldError, ...] = ()
