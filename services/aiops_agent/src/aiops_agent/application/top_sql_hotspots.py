"""有界 SQL 热点池：连续跟踪、轮换发现与可重放的增量基线。"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal, InvalidOperation

MODE = "HOTSPOT_TRACKING"
IDLE_ROUNDS = 3
COOLDOWN_ROUNDS = 3


def statement_key(statement) -> str:
    """保留引擎原生复合身份，不将不同数据库或用户的 SQL 合并。"""
    identity = str(statement.statement_identity_value)
    if str(statement.statement_identity_type) == "ORACLE_SQL_ID":
        return identity
    if str(statement.statement_identity_type) == "MYSQL_DIGEST":
        schema = dict(statement.database_metrics or {}).get("schema_name")
        return identity + ":" + ("N" if schema is None else "V" + str(schema))
    level = "-" if statement.top_level is None else "1" if statement.top_level else "0"
    return ":".join((identity, str(statement.database_name or ""), str(statement.user_identifier or ""), level))


def _counters(statement) -> dict[str, str]:
    names = {
        "cpu_time", "buffer_gets", "disk_reads", "shared_blks_hit", "shared_blks_read",
        "local_blks_hit", "local_blks_read", "temp_blks_read", "temp_blks_written",
        "rows_examined", "temp_disk_tables", "lock_time_picoseconds",
    }
    result = {
        "executions": str(statement.execution_count),
        "duration": str(statement.total_duration_microseconds),
        "rows": str(statement.rows_processed),
    }
    for key, value in dict(statement.database_metrics or {}).items():
        if key in names and value is not None:
            try:
                number = Decimal(str(value))
                if number.is_finite() and number >= 0:
                    result[key] = str(number)
            except InvalidOperation:
                continue
    return result


def prepare_hotspot_round(*, previous: dict, limit: int) -> dict:
    """至少预留五分之一名额发现新热点；新基线优先获得第二次读取。"""
    entries = list(previous.get("entries") or ())
    discovery = max(1, (limit + 4) // 5)
    eligible = [entry for entry in entries if int(entry.get("idle_rounds", 0)) < IDLE_ROUNDS]
    eligible.sort(key=lambda entry: (
        int(entry.get("observations", 0)) >= 2,
        -float(entry.get("activity_score", 0)),
        int(entry.get("idle_rounds", 0)),
        int(entry.get("discovery_rank", 0)),
        str(entry["key"]),
    ))
    retained = [str(entry["key"]) for entry in eligible[:limit - discovery]]
    excluded = sorted(
        {str(entry["key"]) for entry in entries if str(entry["key"]) not in retained}
        | {str(entry["key"]) for entry in previous.get("cooldown", ()) if int(entry.get("rounds_left", 0)) > 0}
    )
    return {
        "tracked_keys": json.dumps(retained, ensure_ascii=False, separators=(",", ":")),
        "excluded_keys": json.dumps(excluded[:200], ensure_ascii=False, separators=(",", ":")),
    }


def update_hotspot_pool(*, statements, previous: dict, instance: dict, limit: int) -> dict:
    """只用本轮实际读取的计数器更新状态；新 SQL 和重置只建立基线。"""
    previous_instance = dict(previous.get("instance") or {})
    same_instance = previous_instance == instance
    if instance.get("database_type") == "MYSQL" and previous_instance.get("identity") == instance.get("identity"):
        before, after = previous_instance.get("uptime_seconds"), instance.get("uptime_seconds")
        if before is not None and after is not None:
            same_instance = int(after) >= int(before) and abs((
                datetime.fromisoformat(instance["started_at"]) -
                datetime.fromisoformat(previous_instance["started_at"])
            ).total_seconds()) <= 2
    old = {str(entry["key"]): entry for entry in previous.get("entries", ())} if same_instance else {}
    entries, growth = [], {}
    baseline_count = reset_count = 0
    for statement in statements[:limit]:
        key = statement_key(statement)
        if key in growth:
            continue
        counters = _counters(statement)
        prior = old.get(key)
        delta = {}
        reset = False
        if prior:
            for metric, value in counters.items():
                prior_value = dict(prior.get("counters") or {}).get(metric)
                if prior_value is not None:
                    difference = Decimal(value) - Decimal(str(prior_value))
                    reset = reset or difference < 0
                    delta[metric] = max(Decimal(0), difference)
        if not prior or reset:
            baseline_count += 1
            reset_count += int(reset)
            delta = {}
        active = any(value > 0 for value in delta.values())
        growth[key] = delta
        entries.append({
            "key": key, "counters": counters, "discovery_rank": int(statement.rank_no),
            "observations": 1 if not prior or reset else int(prior.get("observations", 1)) + 1,
            "idle_rounds": 0 if not prior or reset or active else int(prior.get("idle_rounds", 0)) + 1,
            "activity_score": 0,
        })
    # 每个指标独立归一份额，避免单位量级或大家相同的执行增量掩盖 CPU、逻辑读热点。
    for metric in sorted({metric for delta in growth.values() for metric in delta}):
        total = sum((delta.get(metric, Decimal(0)) for delta in growth.values()), Decimal(0))
        if total <= 0:
            continue
        for entry in entries:
            score = float(growth[entry["key"]].get(metric, Decimal(0)) / total)
            entry["activity_score"] = max(entry["activity_score"], score)
    current = {entry["key"] for entry in entries}
    cooldown = {
        str(entry["key"]): int(entry["rounds_left"]) - 1
        for entry in previous.get("cooldown", ()) if same_instance and int(entry.get("rounds_left", 0)) > 1
    }
    for key in old.keys() - current:
        cooldown[key] = COOLDOWN_ROUNDS
    cooldown = [{"key": key, "rounds_left": rounds_left} for key, rounds_left in sorted(
        cooldown.items(), key=lambda item: (-item[1], item[0])
    ) if key not in current][:limit * COOLDOWN_ROUNDS]
    return {
        "instance": instance, "entries": entries, "cooldown": cooldown,
        "baseline_count": baseline_count, "counter_reset_count": reset_count,
        "retired_count": len(old.keys() - current),
        "new_count": len(current - old.keys()),
        "instance_changed": bool(previous) and not same_instance,
    }
