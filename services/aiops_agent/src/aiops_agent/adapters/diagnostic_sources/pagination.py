"""原生时间边界重叠续取；不以时间戳加一跳过同秒/同纳秒对象。"""

from dataclasses import dataclass

import aiohttp

from aiops_agent.contracts.evidence import SourcePaginationObservation
from .base import DiagnosticSourceAdapterError


@dataclass(frozen=True)
class SourcePageRow:
    position: int
    order: int | str
    identity: str
    content_hash: str
    value: object
    include: bool = True


async def read_source_pages(*, fetch_page, pagination, result_limit, start):
    """边界包含上页尾部，重复行不计入产出；响应变化或无进展明确停止。"""
    limit = min(result_limit, pagination.page_size * pagination.max_pages)
    accepted = {}
    page_counts = []
    hashes = []
    frontier = start
    stop = "PAGE_LIMIT"
    error = None
    for _ in range(pagination.max_pages):
        overlap = sum(row.position == frontier for row in accepted.values())
        fetch_limit = min(801, pagination.page_size + overlap + 1)
        try:
            rows, digest = await fetch_page(frontier, fetch_limit)
            if len(rows) > fetch_limit:
                raise DiagnosticSourceAdapterError("SOURCE_RESPONSE_INVALID", "来源忽略分页行数上限")
            if len({row.identity for row in rows}) != len(rows) or any(row.position < frontier for row in rows):
                raise DiagnosticSourceAdapterError("SOURCE_RESPONSE_INVALID", "来源分页重复对象或越过查询边界")
            if any(row.identity in accepted and accepted[row.identity].content_hash != row.content_hash for row in rows):
                raise DiagnosticSourceAdapterError("SOURCE_PAGE_CHANGED", "重叠边界的原始采样在续取时发生变化")
            ordered = sorted(rows, key=lambda row: (row.position, row.order))
        except (DiagnosticSourceAdapterError, aiohttp.ClientError, TimeoutError) as exc:
            if not page_counts:
                raise
            stop = "SOURCE_ERROR"
            error = exc.code if isinstance(exc, DiagnosticSourceAdapterError) else "SOURCE_UNREACHABLE"
            break
        fresh = [row for row in ordered if row.identity not in accepted]
        kept = sum(row.include for row in accepted.values())
        selected = []
        for row in fresh[:pagination.page_size]:
            if kept == limit:
                break
            selected.append(row)
            kept += int(row.include)
        for row in selected:
            accepted[row.identity] = row
        page_counts.append(sum(row.include for row in selected))
        hashes.append(digest)
        if len(rows) < fetch_limit and len(selected) == len(fresh):
            stop = "EXHAUSTED"
            break
        if kept == limit:
            stop = "ROW_LIMIT"
            break
        if not selected:
            stop = "NON_PROGRESS"
            break
        frontier = selected[-1].position
    summary = SourcePaginationObservation(
        page_size=pagination.page_size, max_pages=pagination.max_pages, result_limit=limit,
        returned_page_rows=tuple(page_counts), response_hashes=tuple(hashes),
        stop_reason=stop, error_code=error,
    )
    return tuple(row.value for row in sorted(accepted.values(), key=lambda row: (row.position, row.order)) if row.include), summary
