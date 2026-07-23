"""Outbox 至少一次投递循环。"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Protocol

from loguru import logger

from platform_core.identity import uuid7


class OutboxSink(Protocol):
    async def publish(self, event_type: str, payload: dict) -> None: ...


class LoggingOutboxSink:
    """开发期 Sink；后续消息中间件只需实现同一 Port。"""

    async def publish(self, event_type: str, payload: dict) -> None:
        logger.info(
            "AIOps Outbox 投递：event_type={} aggregate_id={}",
            event_type,
            payload.get("ops_run_id", "unknown"),
        )


class AIOpsOutboxDispatcher:
    def __init__(
        self,
        *,
        uow_factory,
        sink: OutboxSink,
        dispatcher_id: str,
        lease_seconds: int,
        interval_seconds: float,
    ):
        self._uow_factory = uow_factory
        self._sink = sink
        self._dispatcher_id = dispatcher_id
        self._lease_seconds = lease_seconds
        self._interval = interval_seconds
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run_once(self) -> bool:
        token = uuid7()
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            recovered = await uow.outbox.recover_expired(
                now=now,
                available_at=now + timedelta(seconds=1),
            )
            if recovered:
                await uow.commit()
                return True
            message = await uow.outbox.claim(
                now=now,
                lease_owner=self._dispatcher_id,
                lease_token=token,
                lease_until=now + timedelta(seconds=self._lease_seconds),
            )
            if message is None:
                return False
            # 先提交 Claim，外部投递绝不占用数据库事务。
            snapshot = {
                "outbox_id": message.outbox_id,
                "event_type": message.event_type,
                "payload": dict(message.payload_json or {}),
                "attempt": int(message.attempt_count),
                "max_attempts": int(message.max_attempts),
            }
            await uow.commit()
        try:
            await self._sink.publish(
                snapshot["event_type"], snapshot["payload"]
            )
        except Exception as exc:
            async with self._uow_factory() as uow:
                now = await uow.runs.database_now()
                retry = snapshot["attempt"] < snapshot["max_attempts"]
                changed = await uow.outbox.release_failed(
                    outbox_id=snapshot["outbox_id"],
                    lease_owner=self._dispatcher_id,
                    lease_token=token,
                    now=now,
                    new_status="RETRY_WAIT" if retry else "FAILED",
                    available_at=now
                    + timedelta(seconds=min(2 ** snapshot["attempt"], 60)),
                    error_code="OUTBOX_PUBLISH_FAILED",
                    error_message=type(exc).__name__,
                )
                if changed:
                    await uow.commit()
            return True
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            changed = await uow.outbox.mark_published(
                outbox_id=snapshot["outbox_id"],
                lease_owner=self._dispatcher_id,
                lease_token=token,
                now=now,
            )
            if changed:
                await uow.commit()
        return True

    async def run_forever(self) -> None:
        logger.info("AIOps Outbox Dispatcher 开始运行")
        while not self._stop.is_set():
            worked = await self.run_once()
            if worked:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._interval
                )
            except TimeoutError:
                pass
        logger.info("AIOps Outbox Dispatcher 已停止")
