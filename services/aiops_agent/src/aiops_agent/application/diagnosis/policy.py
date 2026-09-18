"""诊断 Turn 判定与自动入口动手冻结。"""

from __future__ import annotations

from typing import Any, Iterable

from platform_core.contracts.aiops import ActionIntent


_DIAGNOSIS_OBJECTIVES = {"DIAGNOSE", "ASSESS", "COMPARE", "VERIFY"}
_EXPLAIN_OBJECTIVES = {"UNDERSTAND", "EXPLAIN"}
_PLAN_OBJECTIVES = {"PLAN", "CHANGE"}
_AUTOMATIC_WORKFLOWS = {"ALERT_DIAGNOSIS", "INSPECTION"}
_AUTOMATIC_TRIGGERS = {"ALERT", "SCHEDULE"}


def is_automatic_entry(
    *,
    workflow_kind: str | None = None,
    trigger_type: str | None = None,
) -> bool:
    """告警诊断和巡检自动 Run 视为自动入口。"""
    return (
        str(workflow_kind or "") in _AUTOMATIC_WORKFLOWS
        or str(trigger_type or "") in _AUTOMATIC_TRIGGERS
    )


def is_diagnosis_turn(
    *,
    objectives: Iterable[Any] | None = None,
    workflow_kind: str | None = None,
    trigger_type: str | None = None,
) -> bool:
    """按 Task Frame 与入口判定是否走四段诊断契约。"""
    if is_automatic_entry(
        workflow_kind=workflow_kind,
        trigger_type=trigger_type,
    ):
        return True
    values = {str(item) for item in (objectives or ()) if item}
    if not values:
        return False
    if values & _DIAGNOSIS_OBJECTIVES:
        return True
    if values <= _EXPLAIN_OBJECTIVES or values <= _PLAN_OBJECTIVES:
        return False
    return False


def freeze_automatic_entry_intent(
    investigation,
    *,
    workflow_kind: str | None = None,
    trigger_type: str | None = None,
):
    """自动告警/巡检冻结为只诊断，不进入动手。"""
    if not is_automatic_entry(
        workflow_kind=workflow_kind,
        trigger_type=trigger_type,
    ):
        return investigation
    task_frame = investigation.task_frame.model_copy(
        update={
            "action_intent": ActionIntent.NONE,
            "requires_change": False,
        }
    )
    return investigation.model_copy(update={"task_frame": task_frame})
