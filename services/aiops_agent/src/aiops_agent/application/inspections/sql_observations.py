"""从冻结的巡检观测生成 SQL 阅读明细，只做单位与排版转换，不重判健康。"""

from __future__ import annotations

from decimal import Decimal, DecimalException, InvalidOperation, ROUND_HALF_UP
from typing import Any

from aiops_agent.application.display_dictionary import display_field_name


SQL_TEXT_COLUMNS = ("query_sample_text", "sql_text", "normalized_statement", "digest_text")
SQL_DETAIL_TOOL = "db.mysql.statement.digest.top"


def _decimal(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() and number >= 0 else None


def format_picoseconds(value: object) -> str:
    """原始皮秒转换为自适应耗时；缺失或无效值不补零。"""
    number = _decimal(value)
    if number is None:
        return "未记录"
    try:
        return format_duration_ms(number / Decimal("1000000000"))
    except DecimalException:
        return "未记录"


def format_duration_ms(value: object) -> str:
    """按数值选择毫秒、秒、分钟、小时，保持一位小数。"""
    number = _decimal(value)
    if number is None:
        return "未记录"
    divisor, unit = next(((Decimal(size), unit) for size, unit in (
        (3600000, "h"), (60000, "min"), (1000, "s"), (1, "ms"),
    ) if number >= size), (Decimal(1), "ms"))
    try:
        shown = (number / divisor).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    except DecimalException:
        return "未记录"
    return f"{shown:,.1f} {unit}"


def observation_field(key: str, value: object) -> str:
    """原始字段单位不变，仅统一报告的耗时展示单位。"""
    if key.endswith("_picoseconds"):
        return f"{display_field_name(key.removesuffix('_picoseconds'))}：{format_picoseconds(value)}"
    return f"{display_field_name(key)}：{value}"


def select_sql_text(row: dict[str, Any]) -> tuple[str, str] | None:
    for key in SQL_TEXT_COLUMNS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return display_field_name(key), value.strip()
    return None


def inspection_sql_details(
    facts: list[dict[str, Any]], checks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """只展示冻结的问题 SQL 和逐条分析；不在刷新时重判或补调模型。"""
    from .sql_problems import PROBLEM_SQL_TOOL

    projected = [dict(check) for check in checks]
    reviews = {
        review["problem_id"]: review for fact in facts if fact.get("kind") == "inspection_sql_reviews"
        for review in fact.get("reviews") or () if isinstance(review, dict) and review.get("problem_id")
    }
    details = []
    for fact in facts:
        if fact.get("kind") != "inspection_check" or fact.get("tool_id") != PROBLEM_SQL_TOOL:
            continue
        problems = list(fact.get("sql_problems") or ())
        statements = []
        for index, problem in enumerate(problems, 1):
            row = dict(problem["observation"])
            text = str(row.get("sql_text") or "").strip()
            number = _decimal(row.get("execution_count"))
            statements.append({
                "ordinal": index, "schema_name": str(row.get("schema_name") or ""),
                "text_label": {"SAMPLE": "SQL 文本（执行样本）", "NORMALIZED": "SQL 文本（归一化）"}.get(row.get("text_kind"), "SQL 文本"),
                "statement_text": text, "statement_preview": text[:400], "statement_truncated": len(text) > 400,
                "digest": "", "execution_count_display": f"{number:,.0f}" if number is not None else "未记录",
                "total_latency_display": format_duration_ms(row.get("total_elapsed_ms")),
                "average_latency_display": format_duration_ms(row.get("average_elapsed_ms")),
                "maximum_latency_display": format_duration_ms(row.get("maximum_elapsed_ms")),
                "problem_id": problem["problem_id"], "severity": problem["severity"],
                "basis": problem["basis"], "problem_summary": problem["summary"],
                "rule_recommendations": problem["recommendations"],
                "agent_review": reviews.get(problem["problem_id"], {
                    "status": "UNAVAILABLE", "analysis": "Agent 初步分析尚未生成，以下为规则复核建议。",
                }),
            })
        status = fact.get("sql_problem_status") or "UNAVAILABLE"
        details.append({
            "title": "问题 SQL", "tool_id": PROBLEM_SQL_TOOL, "status": status,
            "summary": fact.get("summary") or "问题 SQL 检查尚未完成。",
            "observed_count": len(problems), "shown_count": len(statements), "statements": statements,
            "note": "按严重性及超限程度排序，不按累计调用量排名。依据当前保留的累计统计筛查，不代表日报窗口增量。CPU 时间不是 CPU 使用率，应用或表锁等待不等于已确认行锁根因；未测维度不补零。",
        })
        for check in projected:
            if check.get("tool_id") == PROBLEM_SQL_TOOL:
                check["actual_display"] = str(fact.get("summary") or "")
    # 历史累计榜单不改判为问题，也不伪装成已完成的新筛查。
    if not details and any(fact.get("kind") == "inspection_check" and fact.get("tool_id") == SQL_DETAIL_TOOL for fact in facts):
        summary = "历史报告尚未执行问题 SQL 筛查，请重新运行巡检。"
        details.append({"title": "问题 SQL", "tool_id": PROBLEM_SQL_TOOL, "status": "NOT_ASSESSED",
                        "summary": summary, "observed_count": 0, "shown_count": 0,
                        "statements": [], "note": "原累计统计不代表问题，历史结论未重写。"})
        for check in projected:
            if check.get("tool_id") == SQL_DETAIL_TOOL:
                check["actual_display"] = summary
    return projected, details
