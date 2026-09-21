"""从用户上传的 SQLHC HTML 抽出补证事实，不替代在线取证。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any


SQLHC_REPORT_KIND = "SQLHC"
SQLHC_FACT_TOOL_ID = "user.sqlhc.report"
SQLHC_FACT_COLUMNS = (
    "sql_id",
    "owner",
    "object_name",
    "object_type",
    "last_analyzed",
    "stale_stats",
    "file_name",
)

_SQLHC_MARKERS = (
    re.compile(r"\bSQLHC\b", re.I),
    re.compile(r"SQL Health[- ]Check", re.I),
    re.compile(r"sqlhc\.sql", re.I),
)
_EXA_MARKERS = (
    re.compile(r"\bEXACHK\b", re.I),
    re.compile(r"\bORACHK\b", re.I),
    re.compile(r"\bEXACHECK\b", re.I),
)
_SQL_ID_RE = re.compile(r"\bSQL[_ ]ID\b[:\s]*([A-Za-z0-9]{13})\b", re.I)
_HEADER_ALIASES = {
    "owner": "owner",
    "table_owner": "owner",
    "index_owner": "owner",
    "object_owner": "owner",
    "table_name": "object_name",
    "index_name": "object_name",
    "object_name": "object_name",
    "segment_name": "object_name",
    "object_type": "object_type",
    "index_type": "object_type",
    "last_analyzed": "last_analyzed",
    "stale_stats": "stale_stats",
    "stale": "stale_stats",
    "sql_id": "sql_id",
}


@dataclass(frozen=True, slots=True)
class SqlhcReportFacts:
    """SQLHC HTML 中可审计的结构化补证行。"""

    sql_id: str | None
    file_name: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]

    def as_payload(self) -> dict[str, Any]:
        return {
            "sql_id": self.sql_id,
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


def parse_sqlhc_html(html: str, *, file_name: str = "") -> SqlhcReportFacts | None:
    """识别 SQLHC HTML 并抽出对象统计行；ExaCheck/AWR/SQL Monitor 返回空。"""
    if not html or not _is_sqlhc_html(html):
        return None
    collector = _HtmlTableCollector()
    collector.feed(html)
    collector.close()
    sql_id = _extract_sql_id(html)
    rows: list[tuple[Any, ...]] = []
    seen: set[tuple[Any, ...]] = set()
    for table in collector.tables:
        parsed = _parse_statistics_table(
            table,
            sql_id=sql_id,
            file_name=file_name,
        )
        for row in parsed:
            if row in seen:
                continue
            seen.add(row)
            rows.append(row)
    return SqlhcReportFacts(
        sql_id=sql_id,
        file_name=file_name,
        columns=SQLHC_FACT_COLUMNS,
        rows=tuple(rows),
    )


def _is_sqlhc_html(html: str) -> bool:
    if any(marker.search(html) for marker in _EXA_MARKERS):
        return False
    return any(marker.search(html) for marker in _SQLHC_MARKERS)


def _extract_sql_id(html: str) -> str | None:
    match = _SQL_ID_RE.search(html)
    if match is None:
        return None
    sql_id = match.group(1).strip().lower()
    if len(sql_id) != 13 or not sql_id.isalnum():
        return None
    return sql_id


def _parse_statistics_table(
    table: list[list[str]],
    *,
    sql_id: str | None,
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
        if "stale_stats" in current.values() and (
            "object_name" in current.values() or "owner" in current.values()
        ):
            header_index = index
            mapping = current
            break
    if header_index is None:
        return ()
    object_type_default = _default_object_type(mapping)
    parsed: list[tuple[Any, ...]] = []
    for row in table[header_index + 1 :]:
        values = {field: None for field in SQLHC_FACT_COLUMNS}
        values["sql_id"] = sql_id
        values["file_name"] = file_name or None
        for column_index, field in mapping.items():
            if column_index >= len(row):
                continue
            cell = row[column_index].strip()
            values[field] = cell or None
        if values["object_type"] is None:
            values["object_type"] = object_type_default
        if values["object_name"] is None and values["owner"] is None:
            continue
        parsed.append(tuple(values[name] for name in SQLHC_FACT_COLUMNS))
    return tuple(parsed)


def _default_object_type(mapping: dict[int, str]) -> str | None:
    names = set(mapping.values())
    if "object_type" in names:
        return None
    # 表名列优先于索引名列，避免一张混列表被误判。
    header_names = [mapping[index] for index in sorted(mapping)]
    if "object_name" in names:
        return "TABLE"
    del header_names
    return None


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


__all__ = [
    "SQLHC_FACT_COLUMNS",
    "SQLHC_FACT_TOOL_ID",
    "SQLHC_REPORT_KIND",
    "SqlhcReportFacts",
    "parse_sqlhc_html",
]
