"""Oracle 原生报告入口的确定性交付与完整正文校验。"""

from collections.abc import Mapping, Sequence
from typing import Any

from aiops_agent.contracts.turn_answer import TurnEvidenceFact

REPORT_STARTERS = {
    "oracle.report.awr": ("AWR_REPORT", "db.oracle.awr.report", "AWR 报告"),
    "oracle.report.ash": ("ASH_REPORT", "db.oracle.ash.report", "ASH 报告"),
    "oracle.report.awr-diff": ("AWR_DIFF", "db.oracle.awr.diff_report", "AWR 对比报告"),
}
NATIVE_REPORT_TOOLS = frozenset(item[1] for item in REPORT_STARTERS.values())


def native_report_spec(task_frame: Mapping[str, Any]) -> tuple[str, str, str] | None:
    """只识别已冻结的报告快捷入口，不改变自然语言诊断的任务目标。"""
    subject = task_frame.get("subject_ref") or {}
    if not isinstance(subject, Mapping):
        return None
    return REPORT_STARTERS.get(subject.get("conversation_starter"))


def native_report_html(
    *, columns: Sequence[Mapping[str, Any]], rows: Sequence[Sequence[Any]],
    row_count: int, truncated: bool,
) -> str | None:
    """展示链接与下载使用同一完整性口径，零行或截断正文不能冒充报告。"""
    if (truncated or row_count != len(rows) or not rows or len(columns) != 1
        or columns[0].get("name") != "output"
        or any(len(row) != 1 or row[0] is not None and not isinstance(row[0], str) for row in rows)):
        return None
    body = "".join(row[0] or "" for row in rows)
    normalized = body.strip().lower()
    if not normalized.startswith(("<!doctype html", "<html")) or not normalized.endswith("</html>"):
        return None
    return body


def complete_report_fact(fact: TurnEvidenceFact) -> bool:
    """只有本轮受控来源的完整原生正文可用于交付。"""
    return (
        fact.trust_level == "SOURCE_VERIFIED" and fact.followup_run_id is None
        and native_report_html(columns=fact.columns, rows=fact.rows,
            row_count=fact.row_count, truncated=fact.truncated) is not None
    )
