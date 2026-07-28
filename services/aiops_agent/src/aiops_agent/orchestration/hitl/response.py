"""人工粘贴数据库输出的自动识别、列校验和限界解析。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from typing import Any

from aiops_agent.contracts.hitl import UserProvidedDatabaseResult


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SQLPLUS_SEPARATOR = re.compile(r"^\s*-+(?:\s+-+)*\s*$")
_SQLPLUS_SUMMARY = re.compile(
    r"^\s*(?:\d+\s+rows?\s+selected\.?|no\s+rows\s+selected\.?|"
    r"未选定行。?|已选择\s*\d+\s*行。?)\s*$",
    re.IGNORECASE,
)


def _sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = str(value).replace("\x00", "").strip()
    if len(text) > 4096:
        raise ValueError("人工结果字段长度超限")
    if text.startswith(("=", "+", "-", "@")):
        text = "'" + text
    return text


def _column_key(value: Any) -> str:
    return str(value).strip().strip('"').lower()


def _ordered_rows(
    *,
    columns: tuple[str, ...],
    rows: list[dict[str, Any]],
    expected_columns: tuple[str, ...],
) -> tuple[tuple[Any, ...], ...]:
    normalized_columns = tuple(_column_key(item) for item in columns)
    normalized_expected = tuple(_column_key(item) for item in expected_columns)
    if set(normalized_columns) != set(normalized_expected):
        raise ValueError(
            "人工结果列与请求 Schema 不一致："
            f"期望 {', '.join(expected_columns)}，"
            f"实际 {', '.join(columns)}"
        )
    result = []
    for row in rows:
        normalized_row = {_column_key(key): value for key, value in row.items()}
        result.append(
            tuple(_sanitize(normalized_row.get(column)) for column in normalized_expected)
        )
    return tuple(result)


def _parse_json(
    raw_output: str,
    expected_columns: tuple[str, ...],
) -> tuple[tuple[Any, ...], ...]:
    payload = json.loads(raw_output)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or any(
        not isinstance(item, dict) for item in payload
    ):
        raise ValueError("JSON 数据库输出必须是对象或对象数组")
    columns = (
        tuple(str(item) for item in payload[0])
        if payload
        else expected_columns
    )
    return _ordered_rows(
        columns=columns,
        rows=list(payload),
        expected_columns=expected_columns,
    )


def _parse_delimited(
    raw_output: str,
    expected_columns: tuple[str, ...],
) -> tuple[tuple[Any, ...], ...]:
    sample = raw_output[:8192]
    dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    reader = csv.DictReader(io.StringIO(raw_output), dialect=dialect)
    columns = tuple(reader.fieldnames or ())
    if not columns:
        raise ValueError("分隔文本缺少列名")
    return _ordered_rows(
        columns=columns,
        rows=[dict(item) for item in reader],
        expected_columns=expected_columns,
    )


def _previous_content_line(lines: list[str], index: int) -> int | None:
    for position in range(index - 1, -1, -1):
        if lines[position].strip():
            return position
    return None


def _slice_fixed_width(
    line: str,
    spans: tuple[tuple[int, int], ...],
) -> tuple[str, ...]:
    values = []
    for index, (start, end) in enumerate(spans):
        stop = None if index == len(spans) - 1 else end
        values.append(line[start:stop].strip())
    return tuple(values)


def _parse_sqlplus(
    raw_output: str,
    expected_columns: tuple[str, ...],
) -> tuple[tuple[Any, ...], ...] | None:
    lines = raw_output.splitlines()
    for separator_index, separator in enumerate(lines):
        if not _SQLPLUS_SEPARATOR.fullmatch(separator):
            continue
        header_index = _previous_content_line(lines, separator_index)
        if header_index is None:
            continue
        spans = tuple(
            (match.start(), match.end())
            for match in re.finditer(r"-+", separator)
        )
        if not spans:
            continue
        columns = _slice_fixed_width(lines[header_index], spans)
        if {
            _column_key(item) for item in columns
        } != {
            _column_key(item) for item in expected_columns
        }:
            continue
        row_maps: list[dict[str, Any]] = []
        position = separator_index + 1
        while position < len(lines):
            line = lines[position]
            stripped = line.strip()
            if not stripped:
                position += 1
                continue
            if _SQLPLUS_SUMMARY.fullmatch(stripped) or stripped.startswith("SQL>"):
                break
            values = _slice_fixed_width(line, spans)
            if tuple(_column_key(item) for item in values) == tuple(
                _column_key(item) for item in columns
            ):
                position += 2
                continue
            row_maps.append(dict(zip(columns, values, strict=True)))
            position += 1
        return _ordered_rows(
            columns=columns,
            rows=row_maps,
            expected_columns=expected_columns,
        )
    return None


def _parse_whitespace_table(
    raw_output: str,
    expected_columns: tuple[str, ...],
) -> tuple[tuple[Any, ...], ...]:
    lines = [
        line.strip()
        for line in raw_output.splitlines()
        if line.strip() and not _SQLPLUS_SUMMARY.fullmatch(line.strip())
    ]
    if not lines:
        return ()
    splitter = re.compile(r"\t+|\s{2,}")
    columns = tuple(item for item in splitter.split(lines[0]) if item)
    if len(columns) != len(expected_columns):
        raise ValueError("无法从数据库文本输出中识别完整列名")
    rows = []
    for line in lines[1:]:
        values = tuple(
            item
            for item in splitter.split(line, maxsplit=len(columns) - 1)
        )
        if len(values) != len(columns):
            raise ValueError("数据库文本输出存在无法识别的换行或列对齐")
        rows.append(dict(zip(columns, values, strict=True)))
    return _ordered_rows(
        columns=columns,
        rows=rows,
        expected_columns=expected_columns,
    )


def _parse_raw_output(
    raw_output: str,
    expected_columns: tuple[str, ...],
) -> tuple[tuple[Any, ...], ...]:
    normalized = _ANSI_ESCAPE.sub("", raw_output).replace("\r\n", "\n").strip()
    if not normalized:
        raise ValueError("原始数据库输出为空")
    if _SQLPLUS_SUMMARY.fullmatch(normalized):
        return ()
    if normalized.startswith(("[", "{")):
        return _parse_json(normalized, expected_columns)
    sqlplus_rows = _parse_sqlplus(normalized, expected_columns)
    if sqlplus_rows is not None:
        return sqlplus_rows
    try:
        return _parse_delimited(normalized, expected_columns)
    except (csv.Error, ValueError):
        return _parse_whitespace_table(normalized, expected_columns)


def normalize_raw_response(
    *,
    hitl_id: str,
    query_id: str,
    status: str,
    raw_output: str | None,
    error: str | None,
    expected_columns: tuple[str, ...],
    max_rows: int,
) -> UserProvidedDatabaseResult:
    """自动识别用户粘贴的 SQL*Plus、JSON 或分隔文本。"""
    if status != "SUCCEEDED":
        text = _sanitize(error or "用户未提供结果")
        body = {"status": status, "error": text}
        return UserProvidedDatabaseResult(
            hitl_id=hitl_id,
            query_id=query_id,
            status=status,
            parse_status="NOT_APPLICABLE",
            error=text,
            content_sha256=_hash(body),
        )
    if raw_output is None or len(raw_output.encode()) > 65536:
        raise ValueError("原始数据库输出为空或超过 64 KiB")
    cleaned_output = _ANSI_ESCAPE.sub("", raw_output).replace("\x00", "")
    parse_warning = None
    quality_flags = ("USER_PROVIDED", "AUTO_PARSED")
    try:
        rows = _parse_raw_output(cleaned_output, expected_columns)
        if len(rows) > max_rows:
            raise ValueError("自动识别出的行数超过请求上限")
        columns = expected_columns
        parse_status = "STRUCTURED"
    except (csv.Error, json.JSONDecodeError, ValueError) as exc:
        rows = ()
        columns = ()
        parse_status = "UNSTRUCTURED"
        parse_warning = str(exc)[:1000]
        quality_flags = ("USER_PROVIDED", "UNSTRUCTURED")
    body = {
        "raw_output": cleaned_output,
        "parse_status": parse_status,
        "columns": columns,
        "rows": rows,
        "parse_warning": parse_warning,
    }
    return UserProvidedDatabaseResult(
        hitl_id=hitl_id,
        query_id=query_id,
        status="SUCCEEDED",
        raw_output=cleaned_output,
        parse_status=parse_status,
        columns=columns,
        rows=rows,
        parse_warning=parse_warning,
        content_sha256=_hash(body),
        quality_flags=quality_flags,
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()
