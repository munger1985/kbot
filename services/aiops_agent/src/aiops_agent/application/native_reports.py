"""Oracle 原生报告入口的确定性交付与完整正文校验。"""

from collections.abc import Mapping, Sequence
from typing import Any
from datetime import datetime

from aiops_agent.contracts.turn_answer import TurnEvidenceFact

REPORT_STARTERS = {
    "oracle.report.awr": ("AWR_REPORT", "db.oracle.awr.report", "AWR 报告"),
    "oracle.report.ash": ("ASH_REPORT", "db.oracle.ash.report", "ASH 报告"),
    "oracle.report.awr-diff": ("AWR_DIFF", "db.oracle.awr.diff_report", "AWR 对比报告"),
}
NATIVE_REPORT_TOOLS = frozenset(item[1] for item in REPORT_STARTERS.values())


def report_time_parameters(starter_id: str, parameters: dict, question: str, *, reference_time: datetime) -> dict:
    """自然语言中的明确时刻优先于模型日期；对比按基准在前、比较在后绑定。"""
    from aiops_agent.application.investigation.discovery_binding import parse_time_windows
    spec = REPORT_STARTERS.get(starter_id)
    if spec is None:
        return parameters
    count = 2 if spec[0] == "AWR_DIFF" else 1
    windows = parse_time_windows(question, now=reference_time)
    result = dict(parameters)
    fields = (("first_begin_time", "first_end_time"), ("second_begin_time", "second_end_time")) if count == 2 else (("begin_time", "end_time"),)
    if len(windows) == count and all(begin < end for begin, end in windows):
        for (begin_key, end_key), (begin, end) in zip(fields, sorted(windows)):
            result.update({begin_key: begin.isoformat(), end_key: end.isoformat()})
    if count == 2 and all(result.get(key) for pair in fields for key in pair):
        try:
            pairs = sorted(((datetime.fromisoformat(str(result[begin]).replace("Z", "+00:00")), result[begin], result[end]) for begin, end in fields), key=lambda item: item[0])
        except (ValueError, TypeError):
            return result
        for (begin_key, end_key), (_, begin, end) in zip(fields, pairs):
            result.update({begin_key: begin, end_key: end})
    return result


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
