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
    execution_snapshot: Mapping[str, Any] | None = None,
    tool_invocations: Iterable[object] = (),
) -> dict[str, Any]:
    """生成不包含SQL、参数和内部策略快照的用户可见计划。"""
    decisions: dict[str, str] = {}
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
        actions.append(
            {
                "ordinal": ordinal,
                "action_id": action_id,
                "question": str(raw_action.get("question") or ""),
                "tool_id": tool_id,
                "tool_class": tool_class_for(tool_id),
                "measurement_semantics": str(
                    raw_action.get("measurement_semantics") or "NOT_APPLICABLE"
                ),
                "depends_on": [
                    str(value) for value in raw_action.get("depends_on") or ()
                ],
                "optional": bool(raw_action.get("optional", False)),
                "execution_mode": decisions.get(action_id, "AUTO_EXECUTE"),
                "status": statuses.get(action_id, "PLANNED"),
            }
        )
    return {
        "revision_no": int(plan.get("revision_no") or 1),
        "actions": actions,
    }
