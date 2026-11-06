"""AIOps 持久化事件的续传与保活；心跳不占用业务事件游标。"""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from time import monotonic

from fastapi import Request


async def stream_aiops_event_pages(
    *, request: Request, fetch_page: Callable[[int, int], Awaitable[dict]],
    cursor: int, settings,
) -> AsyncIterator[str]:
    """读取单个上游请求时也保持心跳，关闭连接时取消未完成的读取。"""
    pending = None
    last_output_at = monotonic()
    try:
        yield ": connected\n\n"
        while not await request.is_disconnected():
            pending = asyncio.ensure_future(fetch_page(cursor, settings.sse_batch_size))
            while True:
                remaining = max(0, settings.sse_heartbeat_seconds - (monotonic() - last_output_at))
                done, _ = await asyncio.wait({pending}, timeout=remaining)
                if await request.is_disconnected():
                    return
                if done:
                    finished, pending = pending, None
                    page = finished.result()
                    break
                yield ": heartbeat\n\n"
                last_output_at = monotonic()
            events = page["events"]
            for event in events:
                cursor = int(event["sequence_no"])
                yield (
                    f"id: {cursor}\n"
                    f"event: {event['event_type']}\n"
                    f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                )
                last_output_at = monotonic()
            # 终态仍需读完所有分页，不能丢掉断线期间产生的回答增量。
            if page.get("terminal") and len(events) < settings.sse_batch_size:
                yield f"event: done\ndata: {json.dumps({'sequence_no': cursor})}\n\n"
                return
            if monotonic() - last_output_at >= settings.sse_heartbeat_seconds:
                yield ": heartbeat\n\n"
                last_output_at = monotonic()
            if len(events) < settings.sse_batch_size:
                await asyncio.sleep(settings.sse_poll_interval_seconds)
    finally:
        if pending is not None:
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending
