"""Run 锁内使用的执行预算；人工等待、机器排队和执行分别记账。"""

from datetime import timedelta

from aiops_agent.domain.operations.run import TERMINAL_RUN_STATUSES
from aiops_agent.domain.operations.transitions import ensure_run_transition
from aiops_agent.domain.states import DomainOpsRunStatus


class ExecutionBudgetError(ValueError):
    pass


def _set_phase(run, status):
    if str(run.status) != status:
        ensure_run_transition(DomainOpsRunStatus(run.status), DomainOpsRunStatus(status))
        run.status = status


def _milliseconds(delta):
    return (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000


def task_deadline(task):
    """未经过人工挂起的任务，期限始终从首次执行开始，不因重试刷新。"""
    deadline = getattr(task, "execution_deadline_at", None)
    if deadline is None and getattr(task, "budget_paused_at", None) is None:
        started = getattr(task, "started_at", None)
        if started is not None:
            deadline = started + timedelta(seconds=int(task.timeout_seconds))
    return deadline


def pause_task_budget(task, *, now):
    deadline = task_deadline(task)
    if deadline is None or deadline <= now:
        raise ExecutionBudgetError("任务剩余执行预算已用尽")
    task.budget_remaining_ms = _milliseconds(deadline - now)
    if task.budget_remaining_ms <= 0:
        raise ExecutionBudgetError("任务剩余执行预算不足一毫秒")
    task.budget_paused_at = now
    task.execution_deadline_at = None


def resume_task_budget(task, *, now):
    if (task.budget_paused_at is None or now < task.budget_paused_at
        or task.budget_remaining_ms is None or task.budget_remaining_ms <= 0):
        raise ExecutionBudgetError("任务预算检查点无效或时钟回退")
    task.execution_deadline_at = now + timedelta(milliseconds=int(task.budget_remaining_ms))
    task.budget_paused_at = None
    task.budget_remaining_ms = None
    # 审批后执行是同一次尝试的延续，不消耗一次失败重试，也不刷新已用预算。
    task.resume_pending = 1


def ensure_run_resumable(run, *, now):
    if str(run.status) in TERMINAL_RUN_STATUSES or run.cancel_requested_at is not None:
        raise ExecutionBudgetError("已结束或取消中的 Run 不能恢复")
    paused = getattr(run, "budget_paused_at", None)
    if paused is not None:
        if (now < paused or run.human_wait_until is None or now >= run.human_wait_until
            or run.budget_remaining_ms is None or run.budget_remaining_ms <= 0):
            raise ExecutionBudgetError("Run 人工等待已过期、预算无效或时钟回退")
    elif run.deadline_at is not None and run.deadline_at <= now:
        raise ExecutionBudgetError("Run 执行预算已用尽")


def sync_run_budget(run, tasks, *, now):
    """依据整个任务集合收敛；独立执行、READY 和重试退避期间均不暂停。"""
    if str(run.status) in TERMINAL_RUN_STATUSES or run.cancel_requested_at is not None:
        return
    waiting = {str(task.status) for task in tasks} & {"WAITING_INPUT", "WAITING_APPROVAL"}
    machine = any(str(task.status) in {"RUNNING", "READY", "RETRY_WAIT"} for task in tasks)
    if not waiting or machine:
        if getattr(run, "budget_paused_at", None) is not None:
            ensure_run_resumable(run, now=now)
            run.budget_waited_ms = int(getattr(run, "budget_waited_ms", None) or 0) + _milliseconds(now - run.budget_paused_at)
            run.deadline_at = now + timedelta(milliseconds=int(run.budget_remaining_ms))
            run.budget_paused_at = None
            run.budget_remaining_ms = None
        if str(run.status) in {"WAITING_INPUT", "WAITING_APPROVAL"}:
            _set_phase(run, "RUNNING")
        return
    if getattr(run, "budget_paused_at", None) is None:
        if run.deadline_at is None or run.deadline_at <= now:
            raise ExecutionBudgetError("Run 已无可结转的执行预算")
        run.budget_remaining_ms = _milliseconds(run.deadline_at - now)
        if run.budget_remaining_ms <= 0:
            raise ExecutionBudgetError("Run 剩余执行预算不足一毫秒")
        run.budget_paused_at = now
        run.deadline_at = None
    _set_phase(run, "WAITING_APPROVAL" if "WAITING_APPROVAL" in waiting else "WAITING_INPUT")


def run_is_due(run, *, now):
    if getattr(run, "budget_paused_at", None) is not None:
        return run.human_wait_until is None or run.human_wait_until <= now
    return run.deadline_at is not None and run.deadline_at <= now
