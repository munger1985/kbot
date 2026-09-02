"""调查计划的稳定分类与用户可见投影。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def tool_class_for(tool_id: str) -> str:
    """把工具标识映射为数据库持久化使用的封闭分类。"""
    if tool_id == "monitor.query_range":
        return "PROMETHEUS"
    if tool_id == "loki.query_range":
        return "LOKI"
    if tool_id == "db.oracle.readonly_query":
        return "ORACLE_SQL_DYNAMIC"
    return "ORACLE_SQL"


def safe_plan_projection(
    plan: Mapping[str, Any],
    *,
    task_frame: Mapping[str, Any] | None = None,
    execution_snapshot: Mapping[str, Any] | None = None,
    tool_invocations: Iterable[object] = (),
) -> dict[str, Any]:
    """生成不包含凭据、隐藏推理和内部策略快照的用户可见计划。"""
    decisions: dict[str, str] = {}
    dynamic_queries: dict[str, dict[str, Any]] = {}
    dynamic = dict(
        (execution_snapshot or {}).get("dynamic_invocations") or {}
    )
    for invocation in dynamic.values():
        if not isinstance(invocation, Mapping):
            continue
        action_id = str(invocation.get("action_id") or "")
        validated = invocation.get("validated_query")
        if not action_id or not isinstance(validated, Mapping):
            continue
        decision = str(validated.get("execution_decision") or "")
        if decision in {"AUTO_EXECUTE", "APPROVAL_REQUIRED"}:
            decisions[action_id] = decision
        dynamic_queries[action_id] = {
            "sql_text": str(validated.get("normalized_sql") or ""),
            "parameters": dict(validated.get("parameters") or {}),
            "approval_reason_codes": [
                str(value)
                for value in validated.get("approval_reason_codes") or ()
            ],
        }
    statuses = {
        str(getattr(item, "action_id", "")): str(
            getattr(item, "status", "PLANNED")
        )
        for item in tool_invocations
        if getattr(item, "action_id", None)
    }
    actions = []
    for ordinal, raw_action in enumerate(plan.get("actions") or (), start=1):
        if not isinstance(raw_action, Mapping):
            continue
        action_id = str(raw_action.get("action_id") or "")
        tool_id = str(raw_action.get("tool_id") or "")
        projection = {
            "ordinal": ordinal,
            "action_id": action_id,
            "question": str(raw_action.get("question") or ""),
            "tool_id": tool_id,
            "tool_class": tool_class_for(tool_id),
            "measurement_semantics": str(
                raw_action.get("measurement_semantics") or "NOT_APPLICABLE"
            ),
            "expected_evidence_kind": str(
                raw_action.get("expected_evidence_kind") or ""
            ),
            "depends_on": [
                str(value) for value in raw_action.get("depends_on") or ()
            ],
            "optional": bool(raw_action.get("optional", False)),
            "execution_mode": decisions.get(action_id, "AUTO_EXECUTE"),
            "status": statuses.get(action_id, "PLANNED"),
        }
        if action_id in dynamic_queries:
            projection.update(dynamic_queries[action_id])
        actions.append(projection)
    hypotheses = []
    for raw_hypothesis in plan.get("hypotheses") or ():
        if not isinstance(raw_hypothesis, Mapping):
            continue
        hypotheses.append(
            {
                "hypothesis_id": str(
                    raw_hypothesis.get("hypothesis_id") or ""
                ),
                "statement": str(raw_hypothesis.get("statement") or ""),
                "rationale": str(raw_hypothesis.get("rationale") or ""),
                "confidence": float(raw_hypothesis.get("confidence") or 0),
            }
        )
    public_task_frame = None
    if task_frame is not None:
        public_task_frame = {
            "objectives": [
                str(value) for value in task_frame.get("objectives") or ()
            ],
            "problem_statement": str(
                task_frame.get("problem_statement") or ""
            ),
            "time_scope": (
                str(task_frame.get("time_scope"))
                if task_frame.get("time_scope")
                else None
            ),
            "known_facts": [
                str(value) for value in task_frame.get("known_facts") or ()
            ],
            "unknowns": [
                str(value) for value in task_frame.get("unknowns") or ()
            ],
            "constraints": [
                str(value) for value in task_frame.get("constraints") or ()
            ],
            "success_criteria": [
                str(value)
                for value in task_frame.get("success_criteria") or ()
            ],
            "action_intent": str(
                task_frame.get("action_intent") or "NONE"
            ),
            "diagnostic_profile": str(
                task_frame.get("diagnostic_profile") or "GENERAL"
            ),
            "subject_ref": dict(task_frame.get("subject_ref") or {}),
            "requires_change": bool(
                task_frame.get("requires_change", False)
            ),
        }
    return {
        "revision_no": int(plan.get("revision_no") or 1),
        "task_frame": public_task_frame,
        "hypotheses": hypotheses,
        "actions": actions,
        "answer_if_no_more_evidence": bool(
            plan.get("answer_if_no_more_evidence", False)
        ),
        "stop_reason": (
            str(plan.get("stop_reason")) if plan.get("stop_reason") else None
        ),
    }
