"""复用 AIOps Worker 进程，备份、演练与报告独立轮询。"""

import asyncio
from datetime import UTC, datetime

from loguru import logger

from aiops_agent.application.data_protection_jobs import ProtectionJobs
from aiops_agent.application.data_protection_report import publish_next_drill_report


class DataProtectionWorker:
    def __init__(self, *, uow_factory, backend=None, interval_seconds, clock=lambda: datetime.now(UTC)):
        self._jobs = ProtectionJobs(uow_factory=uow_factory, backend=backend, clock=clock)
        self._uow_factory = uow_factory
        self._clock = clock
        self._interval_seconds = interval_seconds
        self._stopped = asyncio.Event()

    def stop(self):
        self._stopped.set()

    async def _run_schedules(self, operation):
        if operation == "DRILL":
            await self._jobs.confirm_pending_drill()
        work = await self._jobs.claim(operation)
        if work:
            await self._jobs.execute(work)

    async def _report(self):
        await self._jobs.reconcile()
        await publish_next_drill_report(uow_factory=self._uow_factory, now=self._clock())

    async def _loop(self, operation):
        while not self._stopped.is_set():
            try:
                if operation == "REPORT":
                    await self._report()
                else:
                    await self._run_schedules(operation)
            except Exception as exc:
                # Do not log driver output or statements that might carry credentials.
                logger.warning("AIOps 数据保护任务处理失败: operation={} exception={}", operation, type(exc).__name__)
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self._interval_seconds)
            except TimeoutError:
                pass

    async def run_forever(self):
        async with asyncio.TaskGroup() as group:
            for operation in ("BACKUP", "DRILL", "REPORT"):
                group.create_task(self._loop(operation))

