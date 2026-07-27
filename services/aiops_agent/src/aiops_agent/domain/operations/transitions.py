"""Run/Task 状态迁移的唯一校验入口。"""

from enum import StrEnum

from aiops_agent.domain.states import (
    DomainOpsRunStatus,
    DomainOpsTaskStatus,
)

from .run import RUN_TRANSITIONS
from .task import TASK_TRANSITIONS


class InvalidOperationTransition(ValueError):
    def __init__(self, aggregate: str, current: StrEnum, target: StrEnum):
        super().__init__(
            f"{aggregate} 不允许从 {current.value} 迁移到 {target.value}"
        )
        self.aggregate = aggregate
        self.current = current
        self.target = target


def ensure_run_transition(
    current: DomainOpsRunStatus,
    target: DomainOpsRunStatus,
) -> None:
    if target not in RUN_TRANSITIONS[current]:
        raise InvalidOperationTransition("OpsRun", current, target)


def ensure_task_transition(
    current: DomainOpsTaskStatus,
    target: DomainOpsTaskStatus,
) -> None:
    if target not in TASK_TRANSITIONS[current]:
        raise InvalidOperationTransition("OpsTask", current, target)
