"""Ops Run 状态与阶段规则。"""

from aiops_agent.domain.states import DomainOpsRunStatus


TERMINAL_RUN_STATUSES = frozenset(
    {
        DomainOpsRunStatus.COMPLETED,
        DomainOpsRunStatus.DEGRADED,
        DomainOpsRunStatus.REJECTED,
        DomainOpsRunStatus.FAILED,
        DomainOpsRunStatus.CANCELLED,
        DomainOpsRunStatus.EXPIRED,
    }
)

RUN_TRANSITIONS: dict[
    DomainOpsRunStatus, frozenset[DomainOpsRunStatus]
] = {
    DomainOpsRunStatus.CREATED: frozenset(
        {
            DomainOpsRunStatus.SCOPING,
            DomainOpsRunStatus.FAILED,
            DomainOpsRunStatus.CANCELLED,
            DomainOpsRunStatus.EXPIRED,
        }
    ),
    DomainOpsRunStatus.SCOPING: frozenset(
        {
            DomainOpsRunStatus.OBSERVING,
            DomainOpsRunStatus.DIAGNOSING,
            DomainOpsRunStatus.FAILED,
            DomainOpsRunStatus.CANCELLED,
            DomainOpsRunStatus.EXPIRED,
        }
    ),
    DomainOpsRunStatus.OBSERVING: frozenset(
        {
            DomainOpsRunStatus.DIAGNOSING,
            DomainOpsRunStatus.COMPLETED,
            DomainOpsRunStatus.DEGRADED,
            DomainOpsRunStatus.FAILED,
            DomainOpsRunStatus.CANCELLED,
            DomainOpsRunStatus.EXPIRED,
        }
    ),
    DomainOpsRunStatus.DIAGNOSING: frozenset(
        {
            DomainOpsRunStatus.COMPLETED,
            DomainOpsRunStatus.DEGRADED,
            DomainOpsRunStatus.FAILED,
            DomainOpsRunStatus.CANCELLED,
            DomainOpsRunStatus.EXPIRED,
        }
    ),
}

# 步骤 4 尚不执行的阶段仍显式保留，避免状态规则散落到 Handler。
for _status in DomainOpsRunStatus:
    RUN_TRANSITIONS.setdefault(
        _status,
        frozenset()
        if _status in TERMINAL_RUN_STATUSES
        else frozenset(
            {
                DomainOpsRunStatus.FAILED,
                DomainOpsRunStatus.CANCELLED,
                DomainOpsRunStatus.EXPIRED,
            }
        ),
    )


TASK_TYPE_TO_RUN_PHASE: dict[str, DomainOpsRunStatus | None] = {
    "SCOPE": DomainOpsRunStatus.SCOPING,
    "OBSERVE": DomainOpsRunStatus.OBSERVING,
    "DIAGNOSE": DomainOpsRunStatus.DIAGNOSING,
    "PROPOSE": DomainOpsRunStatus.PROPOSING,
    "APPROVE": DomainOpsRunStatus.WAITING_APPROVAL,
    "EXECUTE": DomainOpsRunStatus.EXECUTING,
    "VERIFY": DomainOpsRunStatus.VERIFYING,
    "COMPARE": DomainOpsRunStatus.VERIFYING,
    "REPORT": None,
}
