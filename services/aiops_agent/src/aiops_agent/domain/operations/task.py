"""Ops Task 状态规则。"""

from aiops_agent.domain.states import DomainOpsTaskStatus


TERMINAL_TASK_STATUSES = frozenset(
    {
        DomainOpsTaskStatus.BLOCKED,
        DomainOpsTaskStatus.SUCCEEDED,
        DomainOpsTaskStatus.FAILED,
        DomainOpsTaskStatus.CANCELLED,
        DomainOpsTaskStatus.EXPIRED,
    }
)

TASK_TRANSITIONS: dict[
    DomainOpsTaskStatus, frozenset[DomainOpsTaskStatus]
] = {
    DomainOpsTaskStatus.PENDING: frozenset(
        {
            DomainOpsTaskStatus.READY,
            DomainOpsTaskStatus.BLOCKED,
            DomainOpsTaskStatus.CANCELLED,
            DomainOpsTaskStatus.EXPIRED,
        }
    ),
    DomainOpsTaskStatus.READY: frozenset(
        {
            DomainOpsTaskStatus.RUNNING,
            DomainOpsTaskStatus.BLOCKED,
            DomainOpsTaskStatus.CANCELLED,
            DomainOpsTaskStatus.EXPIRED,
        }
    ),
    DomainOpsTaskStatus.RUNNING: frozenset(
        {
            DomainOpsTaskStatus.WAITING_INPUT,
            DomainOpsTaskStatus.WAITING_APPROVAL,
            DomainOpsTaskStatus.SUCCEEDED,
            DomainOpsTaskStatus.RETRY_WAIT,
            DomainOpsTaskStatus.FAILED,
            DomainOpsTaskStatus.CANCELLED,
            DomainOpsTaskStatus.EXPIRED,
        }
    ),
    DomainOpsTaskStatus.RETRY_WAIT: frozenset(
        {
            DomainOpsTaskStatus.READY,
            DomainOpsTaskStatus.BLOCKED,
            DomainOpsTaskStatus.CANCELLED,
            DomainOpsTaskStatus.EXPIRED,
        }
    ),
    DomainOpsTaskStatus.WAITING_INPUT: frozenset(
        {
            DomainOpsTaskStatus.SUCCEEDED,
            DomainOpsTaskStatus.CANCELLED,
            DomainOpsTaskStatus.EXPIRED,
        }
    ),
    DomainOpsTaskStatus.WAITING_APPROVAL: frozenset(
        {
            DomainOpsTaskStatus.READY,
            DomainOpsTaskStatus.SUCCEEDED,
            DomainOpsTaskStatus.FAILED,
            DomainOpsTaskStatus.CANCELLED,
            DomainOpsTaskStatus.EXPIRED,
        }
    ),
}

for _status in DomainOpsTaskStatus:
    TASK_TRANSITIONS.setdefault(_status, frozenset())
