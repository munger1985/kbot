"""Top SQL 周期采集运行状态投影，不将历史缺口解释为目标库无 SQL。"""

from datetime import datetime
from typing import Any


_COLLECTION_FAILURE_REASONS = {
    "TARGET_CONFIGURATION_INCOMPLETE": "数据库 Target 的版本、连接或诊断凭据配置不完整",
    "IDENTITY_TOOL_UNAVAILABLE": "数据库身份采集工具不可用",
}


def collection_runtime_state(target, *, now: datetime) -> dict[str, Any]:
    """只暴露采集运行字段，不包含目标连接或凭据。"""
    last_attempt = getattr(target, "workload_last_collected_at", None)
    next_run = getattr(target, "workload_next_run_at", None)
    policy = dict(getattr(target, "workload_snapshot_policy_json", None) or {})
    # 两个采集周期加本轮执行预算作为调度宽限，不将正常排队误报为故障。
    grace = 2 * int(policy.get("interval_minutes") or 1) * 60 + int(policy.get("round_timeout_seconds") or 30)
    return {
        "last_attempt_at": last_attempt.isoformat() if last_attempt is not None else None,
        "next_run_at": next_run.isoformat() if next_run is not None else None,
        "last_error_code": getattr(target, "workload_last_error_code", None),
        "consecutive_failures": int(getattr(target, "workload_consecutive_failures", 0) or 0),
        "checked_at": now.isoformat(),
        "schedule_delay_seconds": max(0, int((now - next_run).total_seconds())) if next_run is not None else 0,
        "schedule_grace_seconds": grace,
    }


def collection_failure_reason(result) -> str | None:
    """关闭采集时不把历史错误当成当前启用任务的失败。"""
    state = dict(result.get("collection_state") or {})
    code = state.get("last_error_code")
    if not result.get("collection_enabled"):
        return None
    if code:
        reason = _COLLECTION_FAILURE_REASONS.get(str(code), "周期采集最近一次执行失败")
        return f"{reason}（{code}）"
    if state.get("schedule_delay_seconds", 0) > state.get("schedule_grace_seconds", float("inf")):
        return "周期采集调度逾期，计划时间未推进；请检查采集 Scheduler 和任务创建日志"
    return None


def collection_runtime_note(result) -> str:
    parts = []
    if result.get("snapshot_count") is not None:
        parts.append(f"本次读取快照：{result['snapshot_count']} 条")
    if result.get("collection_state") is not None:
        state = dict(result["collection_state"])
        parts.extend((
            f"最近采集尝试：{state.get('last_attempt_at') or '暂无'}",
            f"下次计划：{state.get('next_run_at') or '未安排'}",
            f"连续失败：{state.get('consecutive_failures', 0)} 次",
        ))
    return "；".join(parts)
