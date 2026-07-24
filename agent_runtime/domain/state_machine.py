"""Agent Run、Task 与 Delegation 的确定性状态机。"""

from enum import StrEnum


class RunStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING_INPUT = "WAITING_INPUT"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING_EXTERNAL = "WAITING_EXTERNAL"
    RETRY_WAIT = "RETRY_WAIT"
    BLOCKED = "BLOCKED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class DelegationStatus(StrEnum):
    CREATED = "CREATED"
    SUBMITTING = "SUBMITTING"
    RUNNING = "RUNNING"
    WAITING_INPUT = "WAITING_INPUT"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class InvalidStateTransition(ValueError):
    """状态机拒绝非法迁移。"""

    def __init__(self, aggregate: str, current: StrEnum, target: StrEnum):
        super().__init__(
            f"{aggregate} 不允许从 {current.value} 迁移到 {target.value}"
        )
        self.aggregate = aggregate
        self.current = current
        self.target = target


RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.CREATED: frozenset({
        RunStatus.RUNNING,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.EXPIRED,
    }),
    RunStatus.RUNNING: frozenset({
        RunStatus.WAITING_INPUT,
        RunStatus.WAITING_APPROVAL,
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.EXPIRED,
    }),
    RunStatus.WAITING_INPUT: frozenset({
        RunStatus.RUNNING,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.EXPIRED,
    }),
    RunStatus.WAITING_APPROVAL: frozenset({
        RunStatus.RUNNING,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.EXPIRED,
    }),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
    RunStatus.EXPIRED: frozenset(),
}

TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({
        TaskStatus.READY,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
    }),
    TaskStatus.READY: frozenset({
        TaskStatus.RUNNING,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
    }),
    TaskStatus.RUNNING: frozenset({
        TaskStatus.SUCCEEDED,
        TaskStatus.WAITING_EXTERNAL,
        TaskStatus.RETRY_WAIT,
        TaskStatus.FAILED,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
    }),
    TaskStatus.WAITING_EXTERNAL: frozenset({
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }),
    TaskStatus.RETRY_WAIT: frozenset({
        TaskStatus.READY,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }),
    TaskStatus.BLOCKED: frozenset({
        TaskStatus.READY,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }),
    TaskStatus.SUCCEEDED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}

DELEGATION_TRANSITIONS: dict[
    DelegationStatus, frozenset[DelegationStatus]
] = {
    DelegationStatus.CREATED: frozenset({
        DelegationStatus.SUBMITTING,
        DelegationStatus.CANCELLED,
        DelegationStatus.EXPIRED,
    }),
    DelegationStatus.SUBMITTING: frozenset({
        DelegationStatus.RUNNING,
        DelegationStatus.FAILED,
        DelegationStatus.CANCEL_REQUESTED,
        DelegationStatus.EXPIRED,
    }),
    DelegationStatus.RUNNING: frozenset({
        DelegationStatus.WAITING_INPUT,
        DelegationStatus.WAITING_APPROVAL,
        DelegationStatus.COMPLETED,
        DelegationStatus.DEGRADED,
        DelegationStatus.FAILED,
        DelegationStatus.CANCEL_REQUESTED,
        DelegationStatus.CANCELLED,
        DelegationStatus.EXPIRED,
    }),
    DelegationStatus.WAITING_INPUT: frozenset({
        DelegationStatus.RUNNING,
        DelegationStatus.FAILED,
        DelegationStatus.CANCEL_REQUESTED,
        DelegationStatus.EXPIRED,
    }),
    DelegationStatus.WAITING_APPROVAL: frozenset({
        DelegationStatus.RUNNING,
        DelegationStatus.FAILED,
        DelegationStatus.CANCEL_REQUESTED,
        DelegationStatus.EXPIRED,
    }),
    DelegationStatus.CANCEL_REQUESTED: frozenset({
        DelegationStatus.CANCELLED,
        DelegationStatus.COMPLETED,
        DelegationStatus.DEGRADED,
        DelegationStatus.FAILED,
        DelegationStatus.EXPIRED,
    }),
    DelegationStatus.COMPLETED: frozenset(),
    DelegationStatus.DEGRADED: frozenset(),
    DelegationStatus.FAILED: frozenset(),
    DelegationStatus.CANCELLED: frozenset(),
    DelegationStatus.EXPIRED: frozenset(),
}


def _ensure_transition(
    aggregate: str,
    current: StrEnum,
    target: StrEnum,
    transitions: dict,
) -> None:
    if target not in transitions[current]:
        raise InvalidStateTransition(aggregate, current, target)


def ensure_run_transition(current: RunStatus, target: RunStatus) -> None:
    _ensure_transition("Run", current, target, RUN_TRANSITIONS)


def ensure_task_transition(current: TaskStatus, target: TaskStatus) -> None:
    _ensure_transition("Task", current, target, TASK_TRANSITIONS)


def ensure_delegation_transition(
    current: DelegationStatus,
    target: DelegationStatus,
) -> None:
    _ensure_transition(
        "Delegation", current, target, DELEGATION_TRANSITIONS,
    )
