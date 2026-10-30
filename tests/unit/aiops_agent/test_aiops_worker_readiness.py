"""后台协程退出和持续领取失败必须反映到 Worker readiness。"""

import asyncio
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient

from aiops_agent.bootstrap.common import create_process_app
from aiops_agent.bootstrap.worker import _check_worker_readiness
from aiops_agent.workers.handlers import HandlerRegistry
from aiops_agent.workers.task_worker import AIOpsTaskWorker


class AIOpsWorkerReadinessTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime = SimpleNamespace(
            service_name="test-aiops-worker",
            check_database_connection=AsyncMock(return_value={"database": "ok"}),
        )
        self.worker = AIOpsTaskWorker(
            runtime_service=object(),
            handler_registry=HandlerRegistry(()),
            worker_id="test-worker",
            lease_seconds=30,
            heartbeat_seconds=10,
            poll_interval_seconds=1,
        )
        self.background_tasks = {
            name: asyncio.create_task(asyncio.Event().wait())
            for name in (
                "task-worker-1", "reconciler", "outbox-dispatcher", "workload-retention"
            )
        }
        self.tasks = list(self.background_tasks.values())
        self.app = create_process_app(
            title="Worker 就绪测试", description="后台组件状态测试",
            service_name=self.runtime.service_name, service_version="test",
            debug=False, lifespan=None,
        )
        self.app.state.runtime = self.runtime
        self.app.state.ready_check = partial(
            _check_worker_readiness,
            runtime=self.runtime,
            background_tasks=self.background_tasks,
            workers=[self.worker],
        )

    async def asyncTearDown(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def _ready(self):
        async with AsyncClient(
            transport=ASGITransport(app=self.app), base_url="http://worker.test"
        ) as client:
            return await client.get("/ready")

    async def test_running_components_and_connected_database_are_ready(self):
        response = await self._ready()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")
        self.assertEqual(len(response.json()["checks"]), 5)
        self.assertTrue(all(value == "ok" for value in response.json()["checks"].values()))
        self.runtime.check_database_connection.assert_awaited_once()

    async def test_repeated_claim_errors_degrade_readiness_until_recovered(self):
        self.worker._consecutive_loop_errors = 3

        response = await self._ready()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"]["checks"]["component:task-worker-1"], "degraded"
        )
        self.worker._consecutive_loop_errors = 0
        self.assertEqual((await self._ready()).status_code, 200)

    async def test_exited_components_are_not_ready_even_when_database_is_connected(self):
        for state in ("completed", "failed", "cancelled"):
            with self.subTest(state=state):
                async def component():
                    if state == "failed":
                        raise RuntimeError("模拟后台组件退出")

                task = asyncio.create_task(component())
                self.tasks.append(task)
                if state == "cancelled":
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                self.background_tasks["outbox-dispatcher"] = task

                response = await self._ready()

                self.assertEqual(response.status_code, 503)
                self.assertEqual(
                    response.json()["detail"]["checks"]["component:outbox-dispatcher"],
                    "failed",
                )

    async def test_worker_exit_is_not_masked_by_degraded_health(self):
        self.worker._consecutive_loop_errors = 3
        task = self.background_tasks["task-worker-1"]
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        response = await self._ready()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"]["checks"]["component:task-worker-1"], "failed"
        )

    async def test_database_failure_is_not_masked_by_running_components(self):
        self.runtime.check_database_connection.return_value = {"database": "not_configured"}

        response = await self._ready()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["checks"]["database"], "not_configured")


if __name__ == "__main__":
    unittest.main()
