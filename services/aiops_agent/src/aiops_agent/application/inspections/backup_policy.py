"""巡检备份策略的固定取证合同，不绑定易漂移的检查项 ID。"""

from __future__ import annotations

from typing import Any

from .check_catalog import compile_selected_check_steps, load_check_catalog


_BACKUP_POLICY_TOOLS = {
    "ORACLE": "db.backup.recent_jobs",
    "POSTGRESQL": "db.postgresql.archiver.status",
    "MYSQL": "db.mysql.binlog.status",
}


def backup_policy_tool_id(database_type: str) -> str | None:
    """调度补证与报告校验使用同一个数据库工具合同。"""
    return _BACKUP_POLICY_TOOLS.get(str(database_type).upper())


def compile_backup_policy_steps(
    evidence_steps: tuple[dict[str, Any], ...],
    *,
    database_type: str,
    schedule_type: str,
    backup_policy: dict[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    """只为已启用的日巡检策略补充缺少的步骤，保留已有模板快照。"""
    if (
        schedule_type != "DAILY"
        or not backup_policy
        or not backup_policy.get("enabled")
    ):
        return evidence_steps
    tool_id = backup_policy_tool_id(database_type)
    if tool_id is None:
        raise ValueError("当前数据库类型没有备份策略固定取证工具")
    prepared = evidence_steps
    if not any(str(step.get("tool_id") or "") == tool_id for step in prepared):
        check_ids = tuple(
            item.check_id
            for group in load_check_catalog().groups
            for item in group.checks
            if item.availability == "READY"
            and item.tool_id == tool_id
            and str(database_type).upper() in item.supported_db_types
        )
        if not check_ids:
            raise ValueError("备份策略固定取证工具没有可执行的检查目录项")
        prepared = (
            *prepared,
            *compile_selected_check_steps(
                check_ids,
                schedule_type=schedule_type,
                database_type=database_type,
            ),
        )
    if str(database_type).upper() == "ORACLE":
        query_days = min(
            90,
            max(7, (int(backup_policy["success_max_age_hours"]) + 23) // 24 + 1),
        )
        prepared = tuple(
            {
                **step,
                "input": {
                    **dict(step.get("input") or {}),
                    "days": max(
                        int(dict(step.get("input") or {}).get("days") or 0),
                        query_days,
                    ),
                },
            }
            if str(step.get("tool_id") or "") == tool_id
            else step
            for step in prepared
        )
    return prepared
