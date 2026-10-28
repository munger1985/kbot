"""在同一次执行的游标上分批取回；不使用 OFFSET 重跑会变化的性能视图。"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from platform_core.contracts.aiops.executor import DiagnosticLimits, DiagnosticPaginationObservation
from .base import DiagnosticDriverError


def pagination_observation(requested, row_count, stop_reason):
    """记录已返回的各页行数；读完有界查询不代表源数据全量。"""
    counts = [requested.page_size] * (row_count // requested.page_size)
    if row_count % requested.page_size:
        counts.append(row_count % requested.page_size)
    return DiagnosticPaginationObservation(
        requested=requested, returned_page_rows=tuple(counts), stop_reason=stop_reason,
    )


def _wire_row(row):
    """与动态结果归一化使用相同的数值和时间文本，避免指数形式低估字节。"""
    def value(item):
        if isinstance(item, bool):
            return item
        if isinstance(item, (float, Decimal)):
            return format(Decimal(str(item)), "f")
        if isinstance(item, datetime):
            return item.replace(tzinfo=timezone.utc).isoformat() if item.tzinfo is None else item.astimezone(timezone.utc).isoformat()
        return item
    return tuple(value(item) for item in row)


async def read_paged_rows(
    *, fetch_page: Callable[[int], Awaitable], limits: DiagnosticLimits,
    materialize_row: Callable[[tuple], Awaitable[tuple[tuple, bool]]] | None = None,
) -> tuple[tuple[tuple[Any, ...], ...], DiagnosticPaginationObservation]:
    """仅多读取一行哨兵判断是否耗尽；行、字节和单元格预算覆盖整组。"""
    requested = limits.pagination
    if requested is None:
        raise ValueError("游标分页必须具有冻结的页数预算")
    rows, byte_size = [], 0
    stop = "EXHAUSTED"
    while len(rows) < limits.max_result_rows:
        count = min(requested.page_size, limits.max_result_rows - len(rows))
        batch = await fetch_page(count)
        if len(batch) > count:
            raise DiagnosticDriverError("OUTPUT_SCHEMA_INVALID")
        for record in batch:
            row = tuple(record)
            if materialize_row is not None:
                row, clipped = await materialize_row(row)
                if clipped:
                    stop = "CELL_LIMIT"
                    break
            rendered = _wire_row(row)
            if any(value is not None and len(str(value)) > limits.max_cell_chars for value in rendered):
                stop = "CELL_LIMIT"
                break
            encoded_size = len(json.dumps(rendered, ensure_ascii=False, separators=(",", ":"), default=str).encode())
            if byte_size + encoded_size > limits.max_result_bytes:
                stop = "BYTE_LIMIT"
                break
            rows.append(row)
            byte_size += encoded_size
        if stop != "EXHAUSTED" or len(batch) < count:
            break
    if stop == "EXHAUSTED" and len(rows) == limits.max_result_rows:
        sentinel = await fetch_page(1)
        if len(sentinel) > 1:
            raise DiagnosticDriverError("OUTPUT_SCHEMA_INVALID")
        if sentinel:
            stop = "ROW_LIMIT"
    return tuple(rows), pagination_observation(requested, len(rows), stop)
