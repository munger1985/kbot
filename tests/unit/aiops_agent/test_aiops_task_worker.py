"""AIOps Task Worker 错误分类测试。"""

import unittest
import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

from aiops_agent.workers.errors import RetryableTaskError
from aiops_agent.workers.handlers import HandlerManifest, HandlerRegistry
from aiops_agent.workers.task_worker import AIOpsTaskWorker
from platform_core.contracts.aiops import TaskLease
from platform_core.identity import uuid7


class _RetryableHandler:
    async def execute(self, context):
        del context
        raise RetryableTaskError("TARGET_CONNECTION_TIMEOUT")


class _RuntimeService:
    def __init__(self) -> None:
        self.failure = None

    async def fail_task(self, command):
        self.failure = command


class _CancellableHandler:
    def __init__(self) -> None:
        self.cleaned = False

    async def execute(self, context):
        del context
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            self.cleaned = True


class _CleanupAwareRuntimeService(_RuntimeService):
    def __init__(self, handler: _CancellableHandler) -> None:
        super().__init__()
        self._handler = handler
        self.cleaned_before_failure = False

    async def fail_task(self, command):
        self.cleaned_before_failure = self._handler.cleaned
        await super().fail_task(command)


class AIOpsTaskWorkerTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _lease(*, timeout_seconds: int = 5) -> TaskLease:
        now = datetime.now(UTC)
        return TaskLease(
            task_id=uuid7(),
            run_id=uuid7(),
            task_key="diagnostic:a1",
            task_type="TOOL_INVOKE",
            handler_id="test.retryable",
            handler_version="1",
            input_schema_version="TEST_INPUT.v1",
            output_schema_version="TEST_RESULT.v1",
            lease_token=uuid7(),
            lease_until=now + timedelta(seconds=30),
            attempt=1,
            max_attempts=2,
            timeout_seconds=timeout_seconds,
            row_version=1,
            target_id=uuid7(),
            agent_id=uuid7(),
            actor_id="tester",
            trace_id="trace-retryable-worker",
            original_request="查询当前Top SQL",
            plan_snapshot={"trigger": {"type": "CHAT"}},
        )

    async def test_retryable_handler_error_maps_to_retryable_failure(self) -> None:
        service = _RuntimeService()
        registry = HandlerRegistry(
            (
                HandlerManifest(
                    handler_id="test.retryable",
                    version="1",
                    output_schema_version="TEST_RESULT.v1",
                    idempotent=True,
                    implementation=_RetryableHandler(),
                ),
            )
        )
        worker = AIOpsTaskWorker(
            runtime_service=service,
            handler_registry=registry,
            worker_id="worker-test",
            lease_seconds=30,
            heartbeat_seconds=10,
            poll_interval_seconds=1,
        )
        await worker._execute(self._lease())

        self.assertIsNotNone(service.failure)
        self.assertEqual(
            "HANDLER_RETRYABLE_FAILURE", service.failure.error_code
        )

    async def test_timeout_waits_for_handler_cleanup_before_failure(self) -> None:
        handler = _CancellableHandler()
        service = _CleanupAwareRuntimeService(handler)
        registry = HandlerRegistry(
            (
                HandlerManifest(
                    handler_id="test.retryable",
                    version="1",
                    output_schema_version="TEST_RESULT.v1",
                    idempotent=True,
                    implementation=handler,
                ),
            )
        )
        worker = AIOpsTaskWorker(
            runtime_service=service,
            handler_registry=registry,
            worker_id="worker-test",
            lease_seconds=30,
            heartbeat_seconds=10,
            poll_interval_seconds=1,
        )

        async def immediate_timeout(tasks, **kwargs):
            del kwargs
            await asyncio.sleep(0)
            return set(), set(tasks)

        with patch(
            "aiops_agent.workers.task_worker.asyncio.wait",
            side_effect=immediate_timeout,
        ):
            await worker._execute(self._lease(timeout_seconds=1))

        self.assertTrue(handler.cleaned)
        self.assertTrue(service.cleaned_before_failure)
        self.assertEqual("HANDLER_TIMEOUT", service.failure.error_code)


class AIOpsTaskWorkerLoopTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _worker(service):
        return AIOpsTaskWorker(
            runtime_service=service,
            handler_registry=HandlerRegistry(()),
            worker_id="worker-loop-test",
            lease_seconds=30,
            heartbeat_seconds=10,
            poll_interval_seconds=0.001,
        )

    async def test_claim_failure_is_logged_and_next_task_still_executes(self):
        lease = AIOpsTaskWorkerTest._lease()
        service = AsyncMock()
        service.claim_task.side_effect = [
            NameError("timedelta is not defined"),
            lease,
        ]
        worker = self._worker(service)
        worker._execute = AsyncMock(side_effect=lambda _: worker.stop())

        with (
            patch("aiops_agent.workers.task_worker.logger") as logs,
            patch.object(worker, "_wait_for_next_poll", wraps=worker._wait_for_next_poll) as wait,
        ):
            await asyncio.wait_for(worker.run_forever(), timeout=1)

        self.assertEqual(service.claim_task.await_count, 2)
        worker._execute.assert_awaited_once_with(lease)
        service.fail_task.assert_not_awaited()
        self.assertEqual(worker.loop_errors_total, 1)
        self.assertEqual(worker.consecutive_loop_errors, 0)
        self.assertTrue(worker.healthy)
        logs.opt.assert_called_once()
        logs.opt.return_value.error.assert_called_once()
        wait.assert_awaited_once()

    async def test_empty_successful_poll_restores_readiness_after_repeated_errors(self):
        service = AsyncMock()
        worker = self._worker(service)
        calls = 0

        async def claim(_command):
            nonlocal calls
            calls += 1
            if calls <= 3:
                raise RuntimeError("模拟任务领取失败")
            self.assertFalse(worker.healthy)
            self.assertEqual(worker.consecutive_loop_errors, 3)
            worker.stop()
            return None

        service.claim_task.side_effect = claim
        with patch("aiops_agent.workers.task_worker.logger"):
            await asyncio.wait_for(worker.run_forever(), timeout=1)

        self.assertEqual(service.claim_task.await_count, 4)
        self.assertEqual(worker.loop_errors_total, 3)
        self.assertEqual(worker.consecutive_loop_errors, 0)
        self.assertTrue(worker.healthy)

    async def test_stop_interrupts_claim_error_backoff_without_another_claim(self):
        service = AsyncMock()
        worker = self._worker(service)
        worker._poll_interval = 60
        failed = asyncio.Event()

        async def claim(_command):
            failed.set()
            raise RuntimeError("模拟任务领取失败")

        service.claim_task.side_effect = claim
        with patch("aiops_agent.workers.task_worker.logger"):
            running = asyncio.create_task(worker.run_forever())
            try:
                await asyncio.wait_for(failed.wait(), timeout=1)
                worker.stop()
                await asyncio.wait_for(running, timeout=1)
            finally:
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)

        service.claim_task.assert_awaited_once()
        self.assertEqual(worker.loop_errors_total, 1)

    async def test_cancellation_propagates_without_retry_or_error_log(self):
        service = AsyncMock()
        service.claim_task.side_effect = asyncio.CancelledError()
        worker = self._worker(service)

        with patch("aiops_agent.workers.task_worker.logger") as logs:
            with self.assertRaises(asyncio.CancelledError):
                await worker.run_forever()

        service.claim_task.assert_awaited_once()
        self.assertEqual(worker.loop_errors_total, 0)
        logs.opt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
