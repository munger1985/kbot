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
    """皮秒精确换算为毫秒；缺失或无效值不补零，展示保留一位小数。"""
    number = _decimal(value)
    if number is None:
        return "未记录"
    try:
        milliseconds = (number / Decimal("1000000000")).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    except DecimalException:
        return "未记录"
    return f"{milliseconds:,.1f} ms"


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


def _statement(row: dict[str, Any], ordinal: int) -> dict[str, Any]:
    selected = select_sql_text(row)
    label, text = selected or ("SQL 文本", "")
    number = _decimal(row.get("execution_count"))
    count = f"{number:,.0f}" if number is not None and number == number.to_integral_value() else "未记录"
    return {
        "ordinal": ordinal, "schema_name": str(row.get("schema_name") or ""),
        "text_label": label, "statement_text": text, "statement_preview": text[:400],
        "statement_truncated": len(text) > 400,
        "digest": str(row.get("digest") or "") if not selected else "",
        "execution_count_display": count,
        "total_latency_display": format_picoseconds(row.get("total_latency_picoseconds")),
        "average_latency_display": format_picoseconds(row.get("average_latency_picoseconds")),
        "maximum_latency_display": format_picoseconds(row.get("maximum_latency_picoseconds")),
    }


def inspection_sql_details(
    facts: list[dict[str, Any]], checks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """只使用已存观测产生阅读投影，不解析旧文案、不查库、不改写冻结结论或观测。"""
    projected = [dict(check) for check in checks]
    details = []
    for fact in facts:
        if fact.get("kind") != "inspection_check" or fact.get("tool_id") != SQL_DETAIL_TOOL:
            continue
        rows = [row for row in fact.get("observations") or () if isinstance(row, dict)]
        if not rows:
            continue
        statements = [_statement(row, index) for index, row in enumerate(rows[:3], 1)]
        count = fact.get("observation_count")
        observed = count if isinstance(count, int) and not isinstance(count, bool) and count >= len(rows) else len(rows)
        details.append({
            "title": str(fact.get("title") or "高负载 SQL"), "tool_id": SQL_DETAIL_TOOL,
            "observed_count": observed, "shown_count": len(statements), "statements": statements,
            "note": "按来源累计总耗时排序；统计自数据库计数器上次重置起累计，不代表仅本次巡检时段，也不直接判定为慢 SQL。",
        })
        for check in projected:
            if check.get("tool_id") == SQL_DETAIL_TOOL:
                check["actual_display"] = f"记录 {observed} 组语句，展示前 {len(statements)} 组；详见下方“高负载 SQL 明细”。"
    return projected, details
