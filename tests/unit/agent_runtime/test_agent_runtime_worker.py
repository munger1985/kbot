"""无状态 Worker 的领取、执行和提交测试。"""

from datetime import datetime, timedelta, timezone
import unittest

from pydantic import ValidationError

from agent_runtime.application import TaskLease
from agent_runtime.domain.planning import ExecutionMode
from agent_runtime.domain.skills import (
    ArtifactDeclaration,
    DataClassification,
    SkillManifest,
    SkillRegistry,
)
from agent_runtime.runtime import AgentRuntimeWorker, SkillArtifact, SkillResult
from agent_runtime.runtime.worker import _exception_detail
from agent_runtime.specialists.conversation.contracts import (
    ContextRewriteOutput,
)
from platform_core.identity import uuid7


class _EchoSkill:
    async def execute(self, context):
        return SkillResult(
            artifact=SkillArtifact(
                artifact_type="GROUNDED_ANSWER",
                schema_version="GroundedAnswer.v1",
                payload={"answer": context.original_input},
            )
        )


class _RuntimeService:
    def __init__(self, lease):
        self.lease = lease
        self.completed = None
        self.failed = None

    async def claim_task(self, command):
        lease, self.lease = self.lease, None
        return lease

    async def complete_task(self, command):
        self.completed = command

    async def fail_task(self, command):
        self.failed = command

    async def heartbeat_task(self, command):
        raise AssertionError("短任务不应触发续租")


class AgentRuntimeWorkerTest(unittest.IsolatedAsyncioTestCase):
    def test_validation_error_detail_contains_field_and_reason(self):
        with self.assertRaises(ValidationError) as captured:
            ContextRewriteOutput.model_validate(
                {
                    "raw_input": "问题",
                    "standalone_query": "问题",
                    "retrieval_queries": ["问题"],
                    "resolved_references": {"错误": "对象"},
                }
            )

        detail = _exception_detail(captured.exception)

        self.assertIn("resolved_references", detail)
        self.assertIn("valid tuple", detail)

    async def test_worker_executes_registered_skill_and_completes_task(self):
        registry = SkillRegistry()
        registry.register(
            SkillManifest(
                skill_id="echo-answer",
                version="1.0.0",
                owner="test",
                specialist="conversation",
                description="回显测试",
                input_schema="EchoInput.v1",
                output_artifacts=(
                    ArtifactDeclaration(
                        artifact_type="GROUNDED_ANSWER",
                        schema_version="GroundedAnswer.v1",
                    ),
                ),
                execution_mode=ExecutionMode.READ_ONLY,
                idempotent=True,
                timeout_seconds=30,
                data_classification=DataClassification.INTERNAL,
            ),
            _EchoSkill(),
        )
        lease = TaskLease(
            task_id=uuid7(),
            run_id=uuid7(),
            task_key="echo",
            task_type="COMPOSE",
            row_version=2,
            lease_token=uuid7(),
            lease_until=datetime.now(timezone.utc) + timedelta(minutes=2),
            attempt=1,
            timeout_seconds=30,
            execution_kind="LOCAL_SKILL",
            specialist="conversation",
            skill_id="echo-answer",
            skill_version="1.0.0",
            domain_id=10,
            agent_id=uuid7(),
            actor_id="user-1",
            request_id="request-1",
            trace_id="trace-1",
            original_input="你好",
        )
        runtime = _RuntimeService(lease)
        worker = AgentRuntimeWorker(
            runtime_service=runtime,
            skill_registry=registry,
            worker_id="worker-1",
            lease_seconds=120,
            poll_interval_seconds=1,
        )

        self.assertTrue(await worker.run_once())
        self.assertIsNotNone(runtime.completed)
        self.assertIsNone(runtime.failed)
        self.assertEqual(
            runtime.completed.artifact.payload["answer"], "你好"
        )


if __name__ == "__main__":
    unittest.main()
