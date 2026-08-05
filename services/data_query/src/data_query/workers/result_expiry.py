"""清除超过固定交付窗口的查询结果内容，保留最小审计事实。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from data_query.persistence import DataQueryUnitOfWork


class DataQueryResultExpiryWorker:
    def __init__(
        self, *, uow_factory: Callable[[], DataQueryUnitOfWork], batch_size: int,
    ) -> None:
        self._uow_factory = uow_factory
        self._batch_size = batch_size

    async def process_batch(self, *, now: datetime | None = None) -> int:
        effective_now = now or datetime.now(UTC)
        async with self._uow_factory() as uow:
            assert uow.results
            purged = await uow.results.purge_expired(
                now=effective_now, limit=self._batch_size,
            )
            await uow.commit()
            return purged
