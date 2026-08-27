"""Ops Run 通用调度状态与 Task 类型规则。"""

from aiops_agent.domain.states import DomainOpsRunStatus


TERMINAL_RUN_STATUSES = frozenset(
    {
        DomainOpsRunStatus.COMPLETED,
        DomainOpsRunStatus.PARTIAL,
        DomainOpsRunStatus.FAILED,
        DomainOpsRunStatus.CANCELLED,
        DomainOpsRunStatus.EXPIRED,
    }
)

_ACTIVE_OUTCOMES = frozenset(
    {
        DomainOpsRunStatus.WAITING_INPUT,
        DomainOpsRunStatus.WAITING_APPROVAL,
        DomainOpsRunStatus.COMPLETED,
        DomainOpsRunStatus.PARTIAL,
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
            DomainOpsRunStatus.QUEUED,
            DomainOpsRunStatus.RUNNING,
            DomainOpsRunStatus.FAILED,
            DomainOpsRunStatus.CANCELLED,
            DomainOpsRunStatus.EXPIRED,
        }
    ),
    DomainOpsRunStatus.QUEUED: frozenset(
        {
            DomainOpsRunStatus.RUNNING,
            DomainOpsRunStatus.FAILED,
            DomainOpsRunStatus.CANCELLED,
            DomainOpsRunStatus.EXPIRED,
        }
    ),
    DomainOpsRunStatus.RUNNING: _ACTIVE_OUTCOMES
    | frozenset({DomainOpsRunStatus.RUNNING}),
    DomainOpsRunStatus.WAITING_INPUT: frozenset(
        {
            DomainOpsRunStatus.RUNNING,
            DomainOpsRunStatus.FAILED,
            DomainOpsRunStatus.CANCELLED,
            DomainOpsRunStatus.EXPIRED,
        }
    ),
    DomainOpsRunStatus.WAITING_APPROVAL: frozenset(
        {
            DomainOpsRunStatus.RUNNING,
            DomainOpsRunStatus.COMPLETED,
            DomainOpsRunStatus.PARTIAL,
            DomainOpsRunStatus.FAILED,
            DomainOpsRunStatus.CANCELLED,
            DomainOpsRunStatus.EXPIRED,
        }
    ),
}

for _status in TERMINAL_RUN_STATUSES:
    RUN_TRANSITIONS[_status] = frozenset()


TASK_TYPE_TO_RUN_PHASE: dict[str, DomainOpsRunStatus | None] = {
    "INTENT_ROUTE": DomainOpsRunStatus.RUNNING,
    "SKILL_PLAN": DomainOpsRunStatus.RUNNING,
    "SKILL_INVOKE": DomainOpsRunStatus.RUNNING,
    "EVIDENCE_ASSESS": DomainOpsRunStatus.RUNNING,
    "ANSWER": DomainOpsRunStatus.RUNNING,
    "REQUEST_INPUT": DomainOpsRunStatus.WAITING_INPUT,
    "PROPOSE": DomainOpsRunStatus.RUNNING,
    "APPROVE": DomainOpsRunStatus.WAITING_APPROVAL,
    "EXECUTE": DomainOpsRunStatus.RUNNING,
    "VERIFY": DomainOpsRunStatus.RUNNING,
    "ROLLBACK": DomainOpsRunStatus.RUNNING,
    "REPORT": None,
}


LEGACY_TASK_TYPE_MAP: dict[str, str] = {
    "SCOPE": "INTENT_ROUTE",
    "OBSERVE": "SKILL_INVOKE",
    "DIAGNOSE": "EVIDENCE_ASSESS",
    "COMPARE": "VERIFY",
}


def normalize_task_type(task_type: str) -> str:
    """把 Blueprint 业务步骤映射到 Schema 13 通用 Task 类型。"""
    normalized = LEGACY_TASK_TYPE_MAP.get(task_type, task_type)
    if normalized not in TASK_TYPE_TO_RUN_PHASE:
        raise ValueError(f"不支持的通用 Task 类型：{task_type}")
    return normalized
