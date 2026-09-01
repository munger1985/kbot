"""租约、Retry、Deadline 与取消收敛循环。"""

from __future__ import annotations

import asyncio

from loguru import logger

from platform_core.identity import uuid7


class AIOpsReconciler:
    def __init__(
        self,
        *,
        runtime_service,
        interval_seconds: float,
        turn_queue_service=None,
    ):
        self._service = runtime_service
        self._turn_queue_service = turn_queue_service
        self._interval = interval_seconds
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run_forever(self) -> None:
        logger.info("AIOps Reconciler 开始运行")
        while not self._stop.is_set():
            try:
                worked = await self._service.reconcile_once(
                    trace_id=str(uuid7())
                )
            except Exception as exc:
                logger.exception(
                    "AIOps Reconciler 本轮失败：type={} error={}",
                    type(exc).__name__,
                    str(exc),
                )
                worked = False
            if self._turn_queue_service is not None:
                try:
                    worked = (
                        await self._turn_queue_service.promote_next()
                        or worked
                    )
                except Exception as exc:
                    logger.exception(
                        "AIOps Conversation Turn 排队提升失败：type={} error={}",
                        type(exc).__name__,
                        str(exc),
                    )
            if worked:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._interval
                )
            except TimeoutError:
                pass
        logger.info("AIOps Reconciler 已停止")
