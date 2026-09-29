"""调查动作进展判定：规范化指纹并阻止跨轮重复执行。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from platform_core.contracts.aiops.investigation import InvestigationPlan


def action_fingerprint(tool_id: str, input_payload: Mapping | None) -> str:
    """按工具和规范化输入生成稳定动作指纹。"""
    encoded = json.dumps(
        {
            "tool_id": str(tool_id),
            "input": dict(input_payload or {}),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def execution_action_fingerprints(
    execution_snapshot: Mapping | None,
) -> tuple[str, ...]:
    """从已冻结执行快照恢复所有实际调度过的动作指纹。"""
    execution = dict(execution_snapshot or {})
    fingerprints: list[str] = []
    for invocation in dict(execution.get("direct_invocations") or {}).values():
        if not isinstance(invocation, Mapping):
            continue
        tool = invocation.get("tool")
        if not isinstance(tool, Mapping):
            continue
        tool_id = str(tool.get("tool_id") or "")
        if tool_id:
            fingerprints.append(
                action_fingerprint(tool_id, tool.get("parameters"))
            )
    for invocation in dict(execution.get("dynamic_invocations") or {}).values():
        if not isinstance(invocation, Mapping):
            continue
        validated = invocation.get("validated_query")
        if not isinstance(validated, Mapping):
            continue
        fingerprints.append(
            action_fingerprint(
                str(invocation.get("tool_id") or "db.oracle.readonly_query"),
                {
                    "sql": validated.get("normalized_sql"),
                    "parameters": dict(validated.get("parameters") or {}),
                },
            )
        )
    return tuple(dict.fromkeys(fingerprints))


def plan_action_fingerprints(plan_payload: Mapping | None) -> tuple[str, ...]:
    """从持久化计划恢复所有已经进入执行 DAG 的动作指纹。"""
    if not isinstance(plan_payload, Mapping):
        return ()
    try:
        plan = InvestigationPlan.model_validate(dict(plan_payload))
    except (TypeError, ValueError):
        return ()
    return tuple(
        action_fingerprint(action.tool_id, action.input)
        for action in plan.actions
        if not action.deferred
    )


def drop_repeated_plan_actions(
    plan: InvestigationPlan,
    *,
    executed_fingerprints: tuple[str, ...],
) -> tuple[InvestigationPlan, tuple[str, ...]]:
    """移除此前已经按相同工具和输入调度过的动作，并修复依赖。"""
    executed = set(executed_fingerprints)
    removed = {
        action.action_id
        for action in plan.actions
        if not action.deferred
        and action_fingerprint(action.tool_id, action.input) in executed
    }
    if not removed:
        return plan, ()
    kept = tuple(
        action.model_copy(
            update={
                "depends_on": tuple(
                    dependency
                    for dependency in action.depends_on
                    if dependency not in removed
                )
            }
        )
        for action in plan.actions
        if action.action_id not in removed
    )
    return plan.model_copy(update={"actions": kept}), tuple(sorted(removed))
