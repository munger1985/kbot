"""CSV/JSON 内联结果的限界解析、列校验和中和。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from typing import Any

from aiops_agent.contracts.hitl import UserProvidedDatabaseResult


def _sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = str(value).replace("\x00", "").strip()
    if len(text) > 4096:
        raise ValueError("人工结果字段长度超限")
    if text.startswith(("=", "+", "-", "@")):
        text = "'" + text
    return text


def normalize_inline_response(
    *,
    hitl_id: str,
    query_id: str,
    status: str,
    result_format: str,
    inline_data: str | None,
    error: str | None,
    expected_columns: tuple[str, ...],
    max_rows: int,
) -> UserProvidedDatabaseResult:
    if status != "SUCCEEDED":
        text = _sanitize(error or "用户未提供结果")
        body = {"status": status, "error": text}
        return UserProvidedDatabaseResult(
            hitl_id=hitl_id,
            query_id=query_id,
            status=status,
            error=text,
            content_sha256=_hash(body),
        )
    if inline_data is None or len(inline_data.encode()) > 65536:
        raise ValueError("内联人工结果为空或超过 64 KiB")
    if result_format == "CSV":
        reader = csv.DictReader(io.StringIO(inline_data))
        columns = tuple(reader.fieldnames or ())
        rows = [
            tuple(_sanitize(row.get(column)) for column in columns)
            for row in reader
        ]
    elif result_format == "JSON":
        payload = json.loads(inline_data)
        if not isinstance(payload, list):
            raise ValueError("JSON 人工结果必须是对象数组")
        columns = tuple(expected_columns)
        rows = []
        for item in payload:
            if not isinstance(item, dict) or set(item) != set(columns):
                raise ValueError("JSON 人工结果列与请求不一致")
            rows.append(tuple(_sanitize(item[column]) for column in columns))
    else:
        raise ValueError("首期内联结果只支持 CSV 或 JSON")
    if columns != expected_columns:
        raise ValueError("人工结果列与请求 Schema 不一致")
    if len(rows) > max_rows:
        raise ValueError("人工结果行数超过请求上限")
    body = {"columns": columns, "rows": rows}
    return UserProvidedDatabaseResult(
        hitl_id=hitl_id,
        query_id=query_id,
        status="SUCCEEDED",
        columns=columns,
        rows=tuple(rows),
        content_sha256=_hash(body),
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
