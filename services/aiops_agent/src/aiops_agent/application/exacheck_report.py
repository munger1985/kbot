"""从用户上传的 ExaCheck / ORAchk HTML 抽出 FAIL/WARNING 事实。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any


EXACHECK_REPORT_KIND = "EXACHECK"
EXACHECK_FACT_TOOL_ID = "user.exacheck.report"
EXACHECK_FACT_COLUMNS = (
    "status",
    "check_name",
    "message",
    "host",
    "check_id",
    "file_name",
)
_FACT_STATUSES = frozenset({"FAIL", "WARNING", "INFO"})

_EXA_MARKERS = (
    re.compile(r"\bEXACHK\b", re.I),
    re.compile(r"\bEXACHECK\b", re.I),
    re.compile(r"\bORACHK\b", re.I),
)
_STATUS_RE = re.compile(r"^\s*(FAIL|WARNING|INFO|PASS)\b", re.I)
_HEADER_ALIASES = {
    "status": "status",
    "result": "status",
    "type": "check_name",
    "check": "check_name",
    "check_name": "check_name",
    "check_type": "check_name",
    "name": "check_name",
    "message": "message",
    "summary": "message",
    "description": "message",
    "details": "message",
    "detail": "message",
    "status_on": "host",
    "hosts": "host",
    "host": "host",
    "hostname": "host",
    "status_on_host": "host",
    "check_id": "check_id",
    "checkid": "check_id",
    "id": "check_id",
}


@dataclass(frozen=True, slots=True)
class ExacheckReportFacts:
    """ExaCheck / ORAchk HTML 中可审计的结构化检查行。"""

    file_name: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]

    def as_payload(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
        }


class _HtmlTableCollector(HTMLParser):
    """只收集可见表格单元格，忽略脚本和样式。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript", "template"}:
            self._ignored += 1
            return
        if self._ignored:
            return
        if normalized == "table":
            self._table = []
        elif normalized == "tr" and self._table is not None:
            self._row = []
        elif normalized in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript", "template"}:
            self._ignored = max(0, self._ignored - 1)
            return
        if self._ignored:
            return
        if normalized in {"td", "th"} and self._cell is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self._row.append(text)
            self._cell = None
        elif normalized == "tr" and self._row is not None:
            if any(cell for cell in self._row):
                self._table.append(self._row)
            self._row = None
        elif normalized == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None

    def handle_data(self, data: str) -> None:
        if not self._ignored and self._cell is not None:
            self._cell.append(data)


def parse_exacheck_html(html: str, *, file_name: str = "") -> ExacheckReportFacts | None:
    """识别 ExaCheck/ORAchk HTML；SQLHC/AWR/SQL Monitor 返回空。"""
    if not html or not _is_exacheck_html(html):
        return None
    collector = _HtmlTableCollector()
    collector.feed(html)
    collector.close()
    rows: list[tuple[Any, ...]] = []
    seen: set[tuple[Any, ...]] = set()
    for table in collector.tables:
        for row in _parse_findings_table(table, file_name=file_name):
            if row in seen:
                continue
            seen.add(row)
            rows.append(row)
    return ExacheckReportFacts(
        file_name=file_name,
        columns=EXACHECK_FACT_COLUMNS,
        rows=tuple(rows),
    )


def _is_exacheck_html(html: str) -> bool:
    return any(marker.search(html) for marker in _EXA_MARKERS)


def _parse_findings_table(
    table: list[list[str]],
    *,
    file_name: str,
) -> tuple[tuple[Any, ...], ...]:
    header_index = None
    mapping: dict[int, str] = {}
    for index, row in enumerate(table):
        current = {}
        for column_index, cell in enumerate(row):
            field = _HEADER_ALIASES.get(_normalize_header(cell))
            if field is not None:
                current[column_index] = field
        if "status" in current.values() and (
            "check_name" in current.values() or "message" in current.values()
        ):
            header_index = index
            mapping = current
            break
    if header_index is None:
        return ()
    parsed: list[tuple[Any, ...]] = []
    for row in table[header_index + 1 :]:
        values = {field: None for field in EXACHECK_FACT_COLUMNS}
        values["file_name"] = file_name or None
        for column_index, field in mapping.items():
            if column_index >= len(row):
                continue
            cell = row[column_index].strip()
            if field == "status":
                values[field] = _normalize_status(cell)
            else:
                values[field] = cell or None
        if values["status"] not in _FACT_STATUSES:
            continue
        if values["check_name"] is None and values["message"] is None:
            continue
        parsed.append(tuple(values[name] for name in EXACHECK_FACT_COLUMNS))
    return tuple(parsed)


def _normalize_status(value: str) -> str | None:
    match = _STATUS_RE.match(value or "")
    if match is None:
        return None
    return match.group(1).upper()


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


__all__ = [
    "EXACHECK_FACT_COLUMNS",
    "EXACHECK_FACT_TOOL_ID",
    "EXACHECK_REPORT_KIND",
    "ExacheckReportFacts",
    "parse_exacheck_html",
]
