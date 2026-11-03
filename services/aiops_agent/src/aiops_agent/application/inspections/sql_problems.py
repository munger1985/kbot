"""按冻结规则识别问题 SQL；累计调用量不作为异常条件。"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from .sql_observations import _decimal


PROBLEM_SQL_TOOL = "db.sql.problem_candidates"
PROBLEM_SQL_COLUMNS = (
    "sql_key", "schema_name", "sql_text", "text_kind", "execution_count",
    "total_elapsed_ms", "average_elapsed_ms", "maximum_elapsed_ms",
    "average_cpu_ms", "average_io_ms", "average_lock_ms", "average_reads",
    "average_rows_examined", "average_spills",
    "workarea_total_executions", "workarea_multipass_executions",
    "workarea_multipass_percent", "maximum_workarea_temp_mb",
    "collection_status",
)


def problem_sql_policy() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("sql_problem_policy.json").read_text(encoding="utf-8"))


def problem_sql_inputs(policy: dict[str, Any]) -> dict[str, int]:
    return {
        **{rule["metric"] + "_limit": int(rule["limit"]) for rule in policy["rules"]},
        "resource_minimum_elapsed_ms": int(policy["resource_minimum_elapsed_ms"]),
    }


def classify_problem_sql(row: dict[str, Any], *, policy: dict[str, Any], database_type: str) -> dict[str, Any] | None:
    """同一 SQL 多种问题合并，按严重性及超限倍数排序，不因累计次数高而入选。"""
    if not row.get("sql_key"):
        return None
    signals = []
    elapsed = _decimal(row.get("average_elapsed_ms"))
    for rule in policy["rules"]:
        if rule.get("database_types") and database_type not in rule["database_types"]:
            continue
        value = _decimal(row.get(rule["metric"]))
        floor = policy["resource_minimum_elapsed_ms"] if rule["minimum_elapsed_ms"] else 0
        if value is None or value < Decimal(str(rule["limit"])):
            continue
        if floor and (elapsed is None or elapsed < floor):
            continue
        label = rule["label"]
        if rule["code"] == "LOCK_WAIT":
            label = "单次表锁等待偏高" if database_type == "MYSQL" else "单次应用等待偏高（需核验锁等原因）"
        unit = rule["unit"]
        if rule["code"] == "DISK_SPILL":
            unit = "次/次" if database_type == "MYSQL" else "块/次"
        signals.append({
            "code": rule["code"], "metric": rule["metric"], "label": label,
            "value": str(value), "limit": rule["limit"], "unit": unit,
            "ratio": str(value / Decimal(str(rule["limit"]))),
            "basis": f"{label}：{value:,.1f} {unit}，筛查阈值 ≥ {rule['limit']} {unit}。",
            "suggestion": rule["suggestion"],
        })
    if not signals:
        return None
    ratio = max(Decimal(signal["ratio"]) for signal in signals)
    severity = "HIGH" if ratio >= policy["high_multiplier"] else "MEDIUM"
    identity = database_type + ":" + str(row["sql_key"])
    return {
        "problem_id": hashlib.sha256(identity.encode()).hexdigest(),
        "sql_key": str(row["sql_key"]), "severity": severity,
        "score": str(ratio), "signals": signals, "observation": dict(row),
        "summary": "；".join(signal["label"] for signal in signals) + "。",
        "basis": "；".join(signal["basis"] for signal in signals),
        "recommendations": list(dict.fromkeys(signal["suggestion"] for signal in signals)),
    }


def problem_sql_sort_key(problem):
    return (-int(problem["severity"] == "HIGH"), -Decimal(problem["score"]), problem["sql_key"])


def freeze_problem_sql_check(fact: dict[str, Any], *, step: dict[str, Any], database_type: str) -> dict[str, Any]:
    """生成期冻结规则、完整性与逐 SQL 问题；展示期不得对历史结果重新判定。"""
    policy = step.get("sql_problem_policy")
    if not isinstance(policy, dict):
        return {**fact, "check_status": "MISSING", "summary": "问题 SQL 检查缺少冻结的判定规则。",
                "sql_problems": [], "sql_problem_status": "UNAVAILABLE"}
    rows = list(fact.get("observations") or ())
    available = bool(rows) and all(row.get("collection_status") == "AVAILABLE" for row in rows)
    # 所有方言都应返回采集状态哨兵；非空候选必须保留有效的次数和单次耗时。
    valid_rows = all(
        (_decimal(row.get("execution_count")) is not None
         and _decimal(row["execution_count"]) > 0
         and _decimal(row.get("average_elapsed_ms")) is not None)
        if row.get("sql_key") else not any(row.get(key) is not None for key in PROBLEM_SQL_COLUMNS if key != "collection_status")
        for row in rows
    )
    matching_inputs = step.get("input") == problem_sql_inputs(policy)
    complete = (available and valid_rows and matching_inputs
                and not fact.get("truncated") and fact.get("check_status") not in {"PARTIAL", "MISSING"})
    problems = [
        problem for row in rows
        if (problem := classify_problem_sql(row, policy=policy, database_type=database_type))
    ]
    problems.sort(key=problem_sql_sort_key)
    findings = [{
        "finding_id": item["problem_id"], "finding_type": "TOP_SQL", "severity": item["severity"],
        "impact": item["summary"], "basis": item["basis"],
        "object_ref": {"sql_id": item["sql_key"], "object_name":
            (str(item["observation"].get("schema_name") or "") + " · "
             + str(item["observation"].get("sql_text") or "来源未返回 SQL 文本")[:160]).strip(" ·")},
        "recommendation": "；".join(item["recommendations"]),
        "evidence_refs": list(fact.get("evidence_refs") or ()),
    } for item in problems]
    summary = f"发现 {len(problems)} 条问题 SQL，按问题严重性排序。" if problems else "当前没有问题 SQL"
    if not complete:
        summary = "问题 SQL 检查未完整完成，不能确认当前没有问题 SQL。" + (
            f" 已保留 {len(problems)} 条已识别问题。" if problems else ""
        )
    return {
        **fact, "title": "问题 SQL", "check_status": ("FINDING" if problems else "CLEAR") if complete else "PARTIAL",
        "summary": summary, "findings": findings, "sql_problems": problems,
        "sql_problem_policy": policy, "sql_problem_status": "COMPLETE" if complete else "PARTIAL",
        "rule_description": "按冻结的单次耗时和资源开销阈值筛查；累计执行次数与累计总耗时不能单独触发问题。",
    }
