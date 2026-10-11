"""常驻流量调度与生命周期。"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from random import Random

from aiops_simulator.adapters import DatabaseAdapter, create_adapter
from aiops_simulator.config import DatabaseConfig, SimulatorConfig
from aiops_simulator.modes.base import OperationKind, TrafficMode


LOGGER = logging.getLogger("aiops_simulator")


@dataclass
class RunnerStats:
    """单数据库自启动以来的实际执行统计。"""

    read_success: int = 0
    write_success: int = 0
    failures: int = 0
    latency_seconds: float = 0.0
    templates: Counter[str] = field(default_factory=Counter)

    @property
    def successes(self) -> int:
        return self.read_success + self.write_success


class DatabaseRunner:
    """按真实时间曲线调度一个数据库的日常业务事务。"""

    def __init__(
        self,
        config: SimulatorConfig,
        database: DatabaseConfig,
        adapter: DatabaseAdapter,
        mode: TrafficMode,
        randomizer: Random,
        stop_event: asyncio.Event,
    ) -> None:
        self.config = config
        self.database = database
        self.adapter = adapter
        self.mode = mode
        self.randomizer = randomizer
        self.stop_event = stop_event
        self.mix = mode.operation_mix(randomizer)
        self.semaphore = asyncio.Semaphore(database.traffic.max_inflight)
        self.stats = RunnerStats()
        self.inflight: set[asyncio.Task[None]] = set()
        self.consecutive_failures = 0
        self.backoff_until = 0.0

    async def _wait_or_stop(self, seconds: float) -> bool:
        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=max(0.0, seconds))
            return True
        except TimeoutError:
            return False

    async def run(self) -> None:
        while not self.stop_event.is_set():
            now = datetime.now(self.config.timezone)
            rate = self.mode.rate_at(self.database, now)
            if rate <= 0:
                if await self._wait_or_stop(30):
                    break
                continue

            monotonic_now = time.monotonic()
            if monotonic_now < self.backoff_until:
                if await self._wait_or_stop(self.backoff_until - monotonic_now):
                    break
                continue

            delay = self.randomizer.expovariate(rate)
            if await self._wait_or_stop(delay):
                break
            await self.semaphore.acquire()
            if self.stop_event.is_set():
                self.semaphore.release()
                break
            operation = self.mix.next()
            task = asyncio.create_task(self._execute(operation))
            self.inflight.add(task)
            task.add_done_callback(self.inflight.discard)

        await self._drain()

    async def _execute(self, operation: OperationKind) -> None:
        started = time.monotonic()
        try:
            now = datetime.now(self.config.timezone)
            call = self.adapter.read(now) if operation is OperationKind.READ else self.adapter.write(now)
            template = await asyncio.wait_for(
                call,
                timeout=self.config.statement_timeout_seconds + 2,
            )
            if operation is OperationKind.READ:
                self.stats.read_success += 1
            else:
                self.stats.write_success += 1
            self.stats.templates[template] += 1
            self.consecutive_failures = 0
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.stats.failures += 1
            self.consecutive_failures += 1
            backoff = min(30.0, float(2 ** min(self.consecutive_failures, 5)))
            self.backoff_until = time.monotonic() + backoff
            LOGGER.warning(
                "%s业务事务失败，错误类型=%s，连续失败=%d，退避=%.0f秒",
                self.adapter.display_name,
                type(exc).__name__,
                self.consecutive_failures,
                backoff,
            )
        finally:
            self.stats.latency_seconds += time.monotonic() - started
            self.semaphore.release()

    async def _drain(self) -> None:
        if not self.inflight:
            return
        tasks = tuple(self.inflight)
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=30)
        except TimeoutError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class SimulatorEngine:
    """独立管理三库适配器，不调用或控制任何KBot服务。"""

    def __init__(self, config: SimulatorConfig, mode: TrafficMode) -> None:
        self.config = config
        self.mode = mode
        self.stop_event = asyncio.Event()
        self.run_id = str(uuid.uuid4())
        self.adapters: list[DatabaseAdapter] = []
        self.runners: list[DatabaseRunner] = []
        for index, database in enumerate(config.databases):
            if not database.enabled:
                continue
            randomizer = Random(config.random_seed + (index + 1) * 1009)
            adapter = create_adapter(
                database, config, randomizer, self.run_id
            )
            self.adapters.append(adapter)
            self.runners.append(
                DatabaseRunner(
                    config,
                    database,
                    adapter,
                    mode,
                    randomizer,
                    self.stop_event,
                )
            )

    async def open(self, *, require_runtime_table: bool) -> None:
        opened: list[DatabaseAdapter] = []
        try:
            for adapter in self.adapters:
                await adapter.open()
                opened.append(adapter)
                await adapter.validate(require_runtime_table=require_runtime_table)
                LOGGER.info("%s连接与表结构预检通过", adapter.display_name)
        except Exception:
            for adapter in reversed(opened):
                await adapter.close()
            raise

    async def close(self) -> None:
        results = await asyncio.gather(
            *(adapter.close() for adapter in reversed(self.adapters)),
            return_exceptions=True,
        )
        for adapter, result in zip(reversed(self.adapters), results, strict=True):
            if isinstance(result, Exception):
                LOGGER.warning(
                    "%s连接池关闭失败，错误类型=%s",
                    adapter.display_name,
                    type(result).__name__,
                )

    async def validate(self) -> None:
        await self.open(require_runtime_table=True)
        await self.close()

    async def prepare(self) -> None:
        await self.open(require_runtime_table=False)
        try:
            for adapter in self.adapters:
                await adapter.prepare()
                await adapter.validate(require_runtime_table=True)
                LOGGER.info("%s实时活动表准备完成", adapter.display_name)
        finally:
            await self.close()

    async def run(self) -> None:
        await self.open(require_runtime_table=True)
        LOGGER.info(
            "日常流量已启动，运行标识=%s，时区=%s，读写比例=8:2",
            self.run_id,
            self.config.timezone_name,
        )
        tasks = [asyncio.create_task(runner.run()) for runner in self.runners]
        tasks.append(asyncio.create_task(self._summary_loop()))
        tasks.append(asyncio.create_task(self._cleanup_loop()))
        try:
            await self.stop_event.wait()
        finally:
            self.stop_event.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.close()
            LOGGER.info("日常流量已停止，所有数据库连接池已关闭")

    def stop(self) -> None:
        self.stop_event.set()

    async def _summary_loop(self) -> None:
        while not await self._wait_interval(self.config.summary_interval_seconds):
            for runner in self.runners:
                stats = runner.stats
                total = stats.successes
                ratio = stats.read_success / total * 100 if total else 0.0
                average_ms = stats.latency_seconds / max(total + stats.failures, 1) * 1000
                top_templates = ",".join(
                    name for name, _ in stats.templates.most_common(3)
                ) or "无"
                LOGGER.info(
                    "%s累计成功=%d，读=%d，写=%d，读占比=%.1f%%，失败=%d，平均耗时=%.1f毫秒，主要模板=%s",
                    runner.adapter.display_name,
                    total,
                    stats.read_success,
                    stats.write_success,
                    ratio,
                    stats.failures,
                    average_ms,
                    top_templates,
                )

    async def _cleanup_loop(self) -> None:
        last_cleanup_date = None
        while not self.stop_event.is_set():
            now = datetime.now(self.config.timezone)
            if now.hour >= self.config.cleanup_hour and last_cleanup_date != now.date():
                cutoff = now - timedelta(days=self.config.retention_days)
                for adapter in self.adapters:
                    deleted_total = 0
                    try:
                        for _ in range(20):
                            deleted = await adapter.cleanup(
                                cutoff, self.config.cleanup_batch_size
                            )
                            deleted_total += deleted
                            if deleted < self.config.cleanup_batch_size:
                                break
                        LOGGER.info(
                            "%s过期模拟数据清理完成，删除=%d",
                            adapter.display_name,
                            deleted_total,
                        )
                    except Exception as exc:
                        LOGGER.warning(
                            "%s过期数据清理失败，错误类型=%s",
                            adapter.display_name,
                            type(exc).__name__,
                        )
                last_cleanup_date = now.date()
            if await self._wait_interval(60):
                break

    async def _wait_interval(self, seconds: float) -> bool:
        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=seconds)
            return True
        except TimeoutError:
            return False
