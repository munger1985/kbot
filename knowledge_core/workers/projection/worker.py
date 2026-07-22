"""Lease-based runtime for KC INDEX and PROFILE jobs."""

import asyncio
from contextlib import suppress
from loguru import logger

from .client import KcIndexProfileClient


class KcIndexProfileWorker:
    def __init__(
        self, *, client: KcIndexProfileClient, worker_id: str,
        lease_seconds: int = 600, poll_interval: float = 2.0,
        index_batch_size: int = 64,
    ):
        self._client = client
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._poll_interval = poll_interval
        self._index_batch_size = index_batch_size
        self._stop = asyncio.Event()

    async def run_forever(self) -> None:
        async with self._client:
            while not self._stop.is_set():
                try:
                    task = await self._next_task()
                    if task is None:
                        await asyncio.sleep(self._poll_interval)
                        continue
                    await self._run_task(task)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("KC INDEX/PROFILE worker loop failed")
                    await asyncio.sleep(self._poll_interval)

    def stop(self) -> None:
        self._stop.set()

    async def _next_task(self) -> tuple[str, dict] | None:
        purge = await self._client.claim_purge(worker_id=self._worker_id, lease_seconds=self._lease_seconds)
        if purge:
            return "purge", purge[0]
        index = await self._client.claim_index(worker_id=self._worker_id, lease_seconds=self._lease_seconds)
        if index:
            return "index", index[0]
        profile = await self._client.claim_profile(worker_id=self._worker_id, lease_seconds=self._lease_seconds)
        if profile:
            return "profile", profile[0]
        return None

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
            logger.info("KC {} job {} completed: {}", kind.upper(), task["job_id"], result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("KC {} job {} failed", kind.upper(), task["job_id"])
            with suppress(Exception):
                if kind == "index":
                    await self._client.fail_index(task, failure_code="WORKER_RUN_FAILED", message=str(exc))
                elif kind == "profile":
                    await self._client.fail_profile(task, failure_code="WORKER_RUN_FAILED", message=str(exc))
                else:
                    await self._client.fail_purge(task, failure_code="WORKER_RUN_FAILED", message=str(exc))
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
