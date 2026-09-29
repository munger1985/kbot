"""工作负载高频事实的小批保留期清理循环。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from loguru import logger


class WorkloadRetentionWorker:
    def __init__(
        self,
        *,
        workload_service,
        interval_seconds: float = 60,
        batch_size: int = 1000,
    ) -> None:
        self._service = workload_service
        self._interval = interval_seconds
        self._batch_size = batch_size
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run_once(self) -> bool:
        result = await self._service.cleanup_expired(
            now=datetime.now(UTC), limit=self._batch_size
        )
        removed = int(result.get("snapshots", 0)) + int(
            result.get("samples", 0)
        )
        if removed:
            logger.info(
                "工作负载保留期清理完成：snapshots={} samples={}",
                result.get("snapshots", 0),
                result.get("samples", 0),
            )
        return removed > 0

    async def run_forever(self) -> None:
        while not self._stop.is_set():
            try:
                worked = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.opt(exception=exc).error(
                    "工作负载保留期清理失败：{}", type(exc).__name__
                )
                worked = False
            if worked:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._interval
                )
            except TimeoutError:
                pass
