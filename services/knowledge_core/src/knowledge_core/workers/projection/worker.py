"""Lease-based runtime for KC INDEX and PROFILE jobs."""

import asyncio
from contextlib import suppress
from loguru import logger

from .client import KcIndexProfileClient
from ..job_wait import AdaptiveJobWait


class KcIndexProfileWorker:
    def __init__(
        self, *, client: KcIndexProfileClient, worker_id: str,
        job_wait: AdaptiveJobWait,
        lease_seconds: int = 600,
        index_batch_size: int = 64,
    ):
        self._client = client
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._job_wait = job_wait
        self._index_batch_size = index_batch_size
        self._stop = asyncio.Event()

    async def run_forever(self) -> None:
        async with self._client:
            try:
                while not self._stop.is_set():
                    try:
                        task = await self._next_task()
                        if task is None:
                            await self._job_wait.wait()
                            continue
                        self._job_wait.reset()
                        await self._run_task(task)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception(
                            "KC Projection Worker 循环执行失败"
                        )
                        await self._job_wait.wait()
            finally:
                await self._job_wait.close()

    def stop(self) -> None:
        self._stop.set()

    async def _next_task(self) -> tuple[str, dict] | None:
        tasks = await self._client.claim_projection(
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
        )
        if not tasks:
            return None
        task = tasks[0]
        kind = {
            "COLLECTION_PURGE": "purge",
            "INDEX": "index",
            "PROFILE": "profile",
        }.get(task.get("job_type"))
        if kind is None:
            raise ValueError(
                f"不支持的 Projection 任务类型：{task.get('job_type')}"
            )
        return kind, task

    async def _run_task(self, task_info: tuple[str, dict]) -> None:
        kind, task = task_info
        heartbeat = asyncio.create_task(self._heartbeat(kind, task))
        try:
            if kind == "index":
                result = await self._client.run_index(task, batch_size=self._index_batch_size)
            elif kind == "profile":
                result = await self._client.run_profile(task)
            else:
                result = await self._client.run_purge(task)
            logger.info("KC {} 任务 {} 已完成：{}", kind.upper(), task["job_id"], result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "KC {} 任务 {} 执行失败", kind.upper(), task["job_id"]
            )
            try:
                if kind == "index":
                    await self._client.fail_index(task, failure_code="WORKER_RUN_FAILED", message=str(exc))
                elif kind == "profile":
                    await self._client.fail_profile(task, failure_code="WORKER_RUN_FAILED", message=str(exc))
                else:
                    await self._client.fail_purge(task, failure_code="WORKER_RUN_FAILED", message=str(exc))
            except Exception:
                logger.exception(
                    "KC {} 任务 {} 上报失败状态失败",
                    kind.upper(),
                    task["job_id"],
                )
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await heartbeat

    async def _heartbeat(self, kind: str, task: dict) -> None:
        while True:
            await asyncio.sleep(max(10, self._lease_seconds / 3))
            if kind == "index":
                await self._client.heartbeat_index(task, lease_seconds=self._lease_seconds)
            elif kind == "profile":
                await self._client.heartbeat_profile(task, lease_seconds=self._lease_seconds)
            else:
                await self._client.heartbeat_purge(task, lease_seconds=self._lease_seconds)
