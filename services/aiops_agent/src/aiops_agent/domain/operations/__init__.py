"""Ops Run、Task、Artifact 与 Event 规则。"""

from .errors import ERROR_CATALOG, RuntimeErrorPolicy
from .run import (
    RUN_TRANSITIONS,
    TASK_TYPE_TO_RUN_PHASE,
    TERMINAL_RUN_STATUSES,
)
from .task import TASK_TRANSITIONS, TERMINAL_TASK_STATUSES
from .transitions import (
    InvalidOperationTransition,
    ensure_run_transition,
    ensure_task_transition,
)

__all__ = [
    "ERROR_CATALOG",
    "RUN_TRANSITIONS",
    "TASK_TRANSITIONS",
    "TASK_TYPE_TO_RUN_PHASE",
    "TERMINAL_RUN_STATUSES",
    "TERMINAL_TASK_STATUSES",
    "InvalidOperationTransition",
    "RuntimeErrorPolicy",
    "ensure_run_transition",
    "ensure_task_transition",
]
