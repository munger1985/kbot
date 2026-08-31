"""AIOps Task Worker 错误分类测试。"""

import unittest
from datetime import UTC, datetime, timedelta

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


class AIOpsTaskWorkerTest(unittest.IsolatedAsyncioTestCase):
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
        now = datetime.now(UTC)
        lease = TaskLease(
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
            timeout_seconds=5,
            row_version=1,
            target_id=uuid7(),
            agent_id=uuid7(),
            actor_id="tester",
            trace_id="trace-retryable-worker",
            original_request="查询当前Top SQL",
            plan_snapshot={"trigger": {"type": "CHAT"}},
        )

        await worker._execute(lease)

        self.assertIsNotNone(service.failure)
        self.assertEqual(
            "HANDLER_RETRYABLE_FAILURE", service.failure.error_code
        )


if __name__ == "__main__":
    unittest.main()
