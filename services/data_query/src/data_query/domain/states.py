"""Data Query 不依赖基础设施的状态机。"""

from enum import StrEnum


class DataSourceStatus(StrEnum):
    DRAFT = "DRAFT"
    VALIDATING = "VALIDATING"
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    FAILED = "FAILED"


class SchemaSnapshotStatus(StrEnum):
    REQUESTED = "REQUESTED"
    DISCOVERING = "DISCOVERING"
    WAITING_SELECTION = "WAITING_SELECTION"
    CAPTURING = "CAPTURING"
    PARTIAL_READY = "PARTIAL_READY"
    READY = "READY"
    FAILED = "FAILED"
    SUPERSEDED = "SUPERSEDED"


class SemanticModelVersionStatus(StrEnum):
    DRAFT = "DRAFT"
    REVIEW = "REVIEW"
    ACTIVE = "ACTIVE"
    REJECTED = "REJECTED"
    RETIRED = "RETIRED"


class DataQueryRunStatus(StrEnum):
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    PREFLIGHT = "PREFLIGHT"
    QUEUED = "QUEUED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    COMPLETED_EMPTY = "COMPLETED_EMPTY"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"
    CANCEL_PENDING = "CANCEL_PENDING"


class DataQueryExecutionStatus(StrEnum):
    CREATED = "CREATED"
    PREFLIGHT = "PREFLIGHT"
    QUEUED = "QUEUED"
    EXECUTING = "EXECUTING"
    REPAIRING = "REPAIRING"
    SUCCEEDED = "SUCCEEDED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"
    CANCEL_PENDING = "CANCEL_PENDING"


_TRANSITIONS: dict[type[StrEnum], dict[StrEnum, frozenset[StrEnum]]] = {
    DataSourceStatus: {
        DataSourceStatus.DRAFT: frozenset({DataSourceStatus.VALIDATING, DataSourceStatus.DISABLED}),
        DataSourceStatus.VALIDATING: frozenset({DataSourceStatus.ACTIVE, DataSourceStatus.FAILED, DataSourceStatus.DISABLED}),
        DataSourceStatus.ACTIVE: frozenset({DataSourceStatus.VALIDATING, DataSourceStatus.DISABLED, DataSourceStatus.FAILED}),
        DataSourceStatus.FAILED: frozenset({DataSourceStatus.VALIDATING, DataSourceStatus.DISABLED}),
        DataSourceStatus.DISABLED: frozenset({DataSourceStatus.VALIDATING}),
    },
    SchemaSnapshotStatus: {
        SchemaSnapshotStatus.REQUESTED: frozenset({SchemaSnapshotStatus.DISCOVERING, SchemaSnapshotStatus.FAILED}),
        SchemaSnapshotStatus.DISCOVERING: frozenset({SchemaSnapshotStatus.WAITING_SELECTION, SchemaSnapshotStatus.FAILED}),
        SchemaSnapshotStatus.WAITING_SELECTION: frozenset({SchemaSnapshotStatus.CAPTURING, SchemaSnapshotStatus.FAILED}),
        SchemaSnapshotStatus.CAPTURING: frozenset({SchemaSnapshotStatus.READY, SchemaSnapshotStatus.PARTIAL_READY, SchemaSnapshotStatus.FAILED}),
        SchemaSnapshotStatus.PARTIAL_READY: frozenset({SchemaSnapshotStatus.CAPTURING, SchemaSnapshotStatus.SUPERSEDED}),
        SchemaSnapshotStatus.READY: frozenset({SchemaSnapshotStatus.SUPERSEDED}),
        SchemaSnapshotStatus.FAILED: frozenset(),
        SchemaSnapshotStatus.SUPERSEDED: frozenset(),
    },
    SemanticModelVersionStatus: {
        SemanticModelVersionStatus.DRAFT: frozenset({SemanticModelVersionStatus.REVIEW}),
        SemanticModelVersionStatus.REVIEW: frozenset({SemanticModelVersionStatus.ACTIVE, SemanticModelVersionStatus.REJECTED, SemanticModelVersionStatus.DRAFT}),
        SemanticModelVersionStatus.ACTIVE: frozenset({SemanticModelVersionStatus.RETIRED}),
        SemanticModelVersionStatus.REJECTED: frozenset({SemanticModelVersionStatus.DRAFT}),
        SemanticModelVersionStatus.RETIRED: frozenset(),
    },
    DataQueryRunStatus: {
        DataQueryRunStatus.CREATED: frozenset({DataQueryRunStatus.VALIDATING, DataQueryRunStatus.CANCELLED, DataQueryRunStatus.FAILED}),
        DataQueryRunStatus.VALIDATING: frozenset({DataQueryRunStatus.CLARIFICATION_REQUIRED, DataQueryRunStatus.PREFLIGHT, DataQueryRunStatus.REJECTED, DataQueryRunStatus.FAILED, DataQueryRunStatus.CANCELLED}),
        DataQueryRunStatus.CLARIFICATION_REQUIRED: frozenset({DataQueryRunStatus.CANCELLED, DataQueryRunStatus.TIMED_OUT}),
        DataQueryRunStatus.PREFLIGHT: frozenset({DataQueryRunStatus.QUEUED, DataQueryRunStatus.REJECTED, DataQueryRunStatus.FAILED, DataQueryRunStatus.CANCELLED}),
        DataQueryRunStatus.QUEUED: frozenset({DataQueryRunStatus.EXECUTING, DataQueryRunStatus.CANCELLED, DataQueryRunStatus.TIMED_OUT, DataQueryRunStatus.FAILED}),
        DataQueryRunStatus.EXECUTING: frozenset({DataQueryRunStatus.COMPLETED, DataQueryRunStatus.COMPLETED_EMPTY, DataQueryRunStatus.CANCEL_PENDING, DataQueryRunStatus.FAILED, DataQueryRunStatus.TIMED_OUT}),
        DataQueryRunStatus.CANCEL_PENDING: frozenset({DataQueryRunStatus.CANCELLED, DataQueryRunStatus.FAILED, DataQueryRunStatus.TIMED_OUT}),
        DataQueryRunStatus.COMPLETED: frozenset(),
        DataQueryRunStatus.COMPLETED_EMPTY: frozenset(),
        DataQueryRunStatus.REJECTED: frozenset(),
        DataQueryRunStatus.FAILED: frozenset(),
        DataQueryRunStatus.TIMED_OUT: frozenset(),
        DataQueryRunStatus.CANCELLED: frozenset(),
    },
}


def can_transition(current: StrEnum, target: StrEnum) -> bool:
    """仅允许显式声明的状态转换；同状态更新由调用方处理。"""
    if type(current) is not type(target):
        return False
    return target in _TRANSITIONS.get(type(current), {}).get(current, frozenset())
