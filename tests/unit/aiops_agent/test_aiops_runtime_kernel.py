"""AIOps 步骤 4 确定性运行内核测试。"""

import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from pydantic import ValidationError

from aiops_agent.domain.operations import (
    InvalidOperationTransition,
    ensure_run_transition,
    ensure_task_transition,
    normalize_task_type,
)
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.domain.states import (
    DomainOpsRunStatus,
    DomainOpsTaskStatus,
)
from aiops_agent.orchestration import (
    Blueprint,
    BlueprintRegistry,
    BlueprintValidationError,
    TaskSpec,
    create_kernel_blueprint_registry,
)
from aiops_agent.workers import (
    TaskExecutionContext,
    create_kernel_handler_registry,
)
from platform_core.contracts.aiops import ArtifactInput, FailOpsTaskCommand
from platform_core.persistence.orm import UniversalTimestamp
from platform_core.identity import uuid7


class RuntimeStateMachineTest(unittest.TestCase):
    def test_step4_happy_path_transitions_are_explicit(self) -> None:
        ensure_run_transition(
            DomainOpsRunStatus.CREATED,
            DomainOpsRunStatus.RUNNING,
        )
        ensure_run_transition(
            DomainOpsRunStatus.RUNNING,
            DomainOpsRunStatus.RUNNING,
        )
        ensure_run_transition(
            DomainOpsRunStatus.RUNNING,
            DomainOpsRunStatus.COMPLETED,
        )
        ensure_task_transition(
            DomainOpsTaskStatus.PENDING,
            DomainOpsTaskStatus.READY,
        )
        ensure_task_transition(
            DomainOpsTaskStatus.READY,
            DomainOpsTaskStatus.RUNNING,
        )
        ensure_task_transition(
            DomainOpsTaskStatus.RUNNING,
            DomainOpsTaskStatus.SUCCEEDED,
        )
        ensure_task_transition(
            DomainOpsTaskStatus.RUNNING,
            DomainOpsTaskStatus.WAITING_APPROVAL,
        )
        ensure_task_transition(
            DomainOpsTaskStatus.WAITING_APPROVAL,
            DomainOpsTaskStatus.READY,
        )

    def test_terminal_state_cannot_restart(self) -> None:
        with self.assertRaises(InvalidOperationTransition):
            ensure_run_transition(
                DomainOpsRunStatus.COMPLETED,
                DomainOpsRunStatus.RUNNING,
            )
        with self.assertRaises(InvalidOperationTransition):
            ensure_task_transition(
                DomainOpsTaskStatus.SUCCEEDED,
                DomainOpsTaskStatus.RUNNING,
            )

    def test_blueprint_task_types_are_normalized_for_schema_16(self) -> None:
        self.assertEqual("CONTEXT_BUILD", normalize_task_type("SCOPE"))
        self.assertEqual("TOOL_INVOKE", normalize_task_type("OBSERVE"))
        self.assertEqual("EVIDENCE_ASSESS", normalize_task_type("DIAGNOSE"))
        self.assertEqual("VERIFY", normalize_task_type("COMPARE"))
        self.assertEqual("REPORT", normalize_task_type("REPORT"))

    def test_unknown_task_type_is_rejected_before_persistence(self) -> None:
        with self.assertRaisesRegex(ValueError, "不支持的通用 Task 类型"):
            normalize_task_type("UNKNOWN_TASK")


class RuntimeArtifactReferenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_input_artifacts_accepts_task_key_artifact_key_and_id(self) -> None:
        """用户证据没有生产Task，也必须能够按Artifact引用传入Handler。"""
        produced_task_id = uuid7()
        direct_artifact_id = uuid7()
        artifacts = [
            SimpleNamespace(
                artifact_id=uuid7(),
                artifact_key="tool-result:1",
                ops_task_id=produced_task_id,
                artifact_type="TOOL_RESULT",
                schema_version="TOOL_RESULT.v1",
                payload_json={"value": 1},
                payload_uri=None,
                content_hash="a" * 64,
                provenance_json={},
                trust_level="OBSERVED",
                security_level=1,
            ),
            SimpleNamespace(
                artifact_id=direct_artifact_id,
                artifact_key="turn-user-input:1",
                ops_task_id=None,
                artifact_type="USER_PROVIDED_INPUT",
                schema_version="USER_PROVIDED_INPUT.v1",
                payload_json={"text": "ORA-27157"},
                payload_uri=None,
                content_hash="b" * 64,
                provenance_json={},
                trust_level="USER_PROVIDED",
                security_level=1,
            ),
        ]

        class _Runs:
            async def list_tasks(self, *, ops_run_id):
                del ops_run_id
                return [
                    SimpleNamespace(
                        ops_task_id=produced_task_id,
                        task_key="tool:1",
                    )
                ]

            async def list_artifacts(self, *, ops_run_id):
                del ops_run_id
                return artifacts

        runtime = object.__new__(AIOpsRuntimeService)
        leased = await runtime._input_artifacts(
            SimpleNamespace(runs=_Runs()),
            run_id=uuid7(),
            task=SimpleNamespace(
                input_artifacts_json=[
                    "tool:1",
                    "turn-user-input:1",
                    str(direct_artifact_id),
                ]
            ),
        )

        self.assertEqual(
            {"tool-result:1", "turn-user-input:1"},
            {item.artifact_key for item in leased},
        )


class RuntimeArtifactReplanTest(unittest.IsolatedAsyncioTestCase):
    async def test_replan_freezes_answer_and_writes_outbox_command(self) -> None:
        """首轮可重试缺口必须可靠触发重规划，不能提前释放回答Task。"""
        answer = SimpleNamespace(
            task_key="answer:compose",
            status="PENDING",
            depends_on_json=["evidence:assess"],
            input_artifacts_json=["evidence:assess"],
        )
        outbox_rows = []
        events = []

        class _Runs:
            async def list_tasks(self, **kwargs):
                del kwargs
                return [answer]

        class _Outbox:
            async def add(self, row):
                outbox_rows.append(row)

        class _Turns:
            async def add_event(self, row):
                events.append(row)

        runtime = object.__new__(AIOpsRuntimeService)
        turn = SimpleNamespace(
            turn_id=uuid7(),
            current_plan_revision=1,
            status="ASSESSING",
            event_cursor=3,
        )
        run = SimpleNamespace(
            ops_run_id=uuid7(),
            domain_id=7,
            trace_id="trace-replan",
        )
        await runtime._schedule_turn_replan(
            uow=SimpleNamespace(
                runs=_Runs(),
                outbox=_Outbox(),
                turns=_Turns(),
            ),
            run=run,
            turn=turn,
            assessment_artifact=SimpleNamespace(artifact_id=uuid7()),
        )

        self.assertEqual(["evidence:assess:r2"], answer.depends_on_json)
        self.assertEqual("REPLANNING", turn.status)
        self.assertEqual(
            "aiops.turn.replanning_requested",
            outbox_rows[0].event_type,
        )
        self.assertEqual("turn.status", events[-1].event_type)


class RuntimeLeaseValidationTest(unittest.TestCase):
    def test_failure_can_be_persisted_after_owned_lease_expires(self) -> None:
        runtime = object.__new__(AIOpsRuntimeService)
        now = datetime.now(UTC)
        token = uuid7()
        run = SimpleNamespace(cancel_requested_at=None, deadline_at=None)
        task = SimpleNamespace(
            status="RUNNING",
            lease_owner="worker-test",
            lease_token=token,
            lease_until=now - timedelta(seconds=1),
        )

        with self.assertRaises(AIOpsApplicationError) as raised:
            runtime._ensure_lease(
                run=run,
                task=task,
                worker_id="worker-test",
                lease_token=token,
                now=now,
            )
        self.assertEqual("OPS_STALE_LEASE", raised.exception.code)

        runtime._ensure_lease(
            run=run,
            task=task,
            worker_id="worker-test",
            lease_token=token,
            now=now,
            allow_expired=True,
        )

        task.lease_owner = "other-worker"
        with self.assertRaises(AIOpsApplicationError):
            runtime._ensure_lease(
                run=run,
                task=task,
                worker_id="worker-test",
                lease_token=token,
                now=now,
                allow_expired=True,
            )


class RuntimeTerminalFailureTest(unittest.IsolatedAsyncioTestCase):
    async def test_reconciler_cleans_expired_task_for_cancelled_run(self) -> None:
        """已取消 Run 的遗留租约必须可收敛，不能阻塞后续队列。"""
        now = datetime.now(UTC)
        run = SimpleNamespace(
            ops_run_id=uuid7(),
            status="CANCELLED",
            workflow_kind="INSPECTION",
            cancel_requested_at=now - timedelta(minutes=1),
            completed_at=now - timedelta(minutes=1),
        )
        task = SimpleNamespace(
            ops_task_id=uuid7(),
            ops_run_id=run.ops_run_id,
            status="RUNNING",
            task_type="TOOL_INVOKE",
            task_key="observe",
            attempt_count=1,
            error_code=None,
            lease_owner="worker-1",
            lease_token=uuid7(),
            lease_until=now - timedelta(seconds=1),
            heartbeat_at=now - timedelta(minutes=1),
            completed_at=None,
        )
        events = []

        class _Runs:
            async def database_now(self):
                return now

            async def lock_due_run(self, *, now):
                del now
                return None

            async def get_run(self, *, ops_run_id):
                self_outer.assertEqual(run.ops_run_id, ops_run_id)
                return run

            async def lock_expired_task(self, *, now):
                del now
                return task

            async def list_tasks(self, *, ops_run_id, lock=False):
                self_outer.assertEqual(run.ops_run_id, ops_run_id)
                self_outer.assertTrue(lock)
                return [task]

            async def append_event(self, **kwargs):
                events.append(kwargs)
                return SimpleNamespace(sequence_no=len(events))

        class _Changes:
            async def find_expired_proposal(self, *, now):
                del now
                return None

            async def find_expired_hitl(self):
                return None

            async def find_due_execution(self, *, now):
                del now
                return None

        class _Uow:
            runs = _Runs()
            changes = _Changes()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                del args

            async def commit(self):
                self_outer.committed = True

        self_outer = self
        self.committed = False
        runtime = object.__new__(AIOpsRuntimeService)
        runtime._uow_factory = lambda: _Uow()

        worked = await runtime.reconcile_once(trace_id="trace-cancelled")

        self.assertTrue(worked)
        self.assertEqual("CANCELLED", run.status)
        self.assertEqual("CANCELLED", task.status)
        self.assertIsNone(task.lease_owner)
        self.assertIsNone(task.lease_token)
        self.assertIsNone(task.lease_until)
        self.assertTrue(self.committed)
        self.assertEqual("task.status", events[0]["event_type"])

    async def test_parallel_task_failure_does_not_retransition_terminal_run(self) -> None:
        now = datetime.now(UTC)
        run_id = uuid7()
        task_id = uuid7()
        events = []
        task = SimpleNamespace(
            ops_task_id=task_id,
            task_key="log:binding-2",
            task_type="TOOL_INVOKE",
            handler_id="evidence.log-query",
            handler_version="1",
            status="RUNNING",
            error_code=None,
            error_message=None,
            available_at=now,
            completed_at=None,
            lease_owner="worker-1",
            lease_token=uuid7(),
            lease_until=now,
            heartbeat_at=now,
            attempt_count=1,
            max_attempts=1,
            depends_on_json=[],
            row_version=1,
        )
        run = SimpleNamespace(
            ops_run_id=run_id,
            status="FAILED",
            error_code="FIRST_FAILURE",
            error_message="首个失败原因",
            completed_at=now,
            row_version=1,
        )

        class _Runs:
            async def database_now(self):
                return now

            async def get_event_by_key(self, **kwargs):
                del kwargs
                return None

            async def list_tasks(self, **kwargs):
                del kwargs
                return [task]

            async def append_event(self, **kwargs):
                events.append(kwargs)
                return SimpleNamespace(sequence_no=len(events))

        class _Uow:
            runs = _Runs()
            platform_notifications = None

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                del args

            async def commit(self):
                return None

        runtime = object.__new__(AIOpsRuntimeService)
        runtime._uow_factory = lambda: _Uow()
        runtime._lock_run_task = AsyncMock(return_value=(run, task))
        runtime._ensure_lease = Mock()
        runtime._handlers = SimpleNamespace(
            resolve=lambda *_: SimpleNamespace(idempotent=False)
        )

        receipt = await runtime.fail_task(
            FailOpsTaskCommand(
                task_id=task_id,
                worker_id="worker-1",
                lease_token=task.lease_token,
                idempotency_key="parallel-failure",
                trace_id="trace-parallel-failure",
                error_code="HANDLER_TERMINAL_FAILURE",
            )
        )

        self.assertEqual("FAILED", receipt.task_status)
        self.assertEqual("FAILED", receipt.run_status)
        self.assertEqual("FIRST_FAILURE", run.error_code)
        self.assertEqual(1, len(events))
        self.assertEqual("task.status", events[0]["event_type"])


class BlueprintRegistryTest(unittest.TestCase):
    def test_kernel_blueprint_is_scope_observe_report(self) -> None:
        blueprint = create_kernel_blueprint_registry().resolve(
            "kernel.observe-report", "1"
        )
        self.assertEqual(
            ["SCOPE", "OBSERVE", "REPORT"],
            [task.task_type for task in blueprint.tasks],
        )
        self.assertEqual(("scope", "observe"), blueprint.tasks[2].depends_on)

    def test_cycle_is_rejected(self) -> None:
        blueprint = Blueprint(
            blueprint_id="invalid",
            version="1",
            final_task_key="b",
            tasks=(
                TaskSpec(
                    task_key="a",
                    task_type="OBSERVE",
                    handler_id="a",
                    handler_version="1",
                    input_schema_version="X.v1",
                    output_schema_version="X.v1",
                    depends_on=("b",),
                ),
                TaskSpec(
                    task_key="b",
                    task_type="OBSERVE",
                    handler_id="b",
                    handler_version="1",
                    input_schema_version="X.v1",
                    output_schema_version="X.v1",
                    depends_on=("a",),
                ),
            ),
        )
        with self.assertRaises(BlueprintValidationError):
            BlueprintRegistry.validate(blueprint, max_tasks=10)


class KernelHandlerTest(unittest.IsolatedAsyncioTestCase):
    async def test_handlers_produce_frozen_schemas(self) -> None:
        registry = create_kernel_handler_registry()
        base = {
            "run_id": "run-1",
            "task_id": "task-1",
            "task_key": "scope",
            "target_id": "target-1",
            "agent_id": "agent-1",
            "trigger_type": "CHAT",
            "trace_id": "trace-1",
            "attempt": 1,
            "deadline_at": None,
            "plan_snapshot": {
                "target": {"row_version": 3},
                "binding": {"row_version": 2},
            },
            "policy_snapshot": {},
            "input_artifacts": (),
        }
        scope = await registry.resolve(
            "kernel.scope", "1"
        ).implementation.execute(TaskExecutionContext(**base))
        self.assertEqual("SCOPE_RESULT.v1", scope.schema_version)

        observe = await registry.resolve(
            "kernel.observe", "1"
        ).implementation.execute(TaskExecutionContext(**base))
        self.assertEqual("OBSERVATION_SET.v1", observe.schema_version)

        report_context = {
            **base,
            "input_artifacts": (
                {
                    "schema_version": "OBSERVATION_SET.v1",
                    "payload": observe.model_dump(mode="json"),
                },
            ),
        }
        report = await registry.resolve(
            "kernel.report", "1"
        ).implementation.execute(TaskExecutionContext(**report_context))
        self.assertEqual("KERNEL_TEST_REPORT.v1", report.schema_version)

    def test_artifact_requires_content(self) -> None:
        with self.assertRaises(ValidationError):
            ArtifactInput(
                artifact_type="REPORT",
                schema_version="REPORT.v1",
                producer="handler",
                producer_version="1",
            )


class OracleTimeContractTest(unittest.TestCase):
    def test_oracle_timestamp_normalizes_to_utc(self) -> None:
        timestamp = UniversalTimestamp(timezone=True)
        oracle = Mock(name="oracle")
        oracle.name = "oracle"
        value = datetime(2026, 7, 23, 22, 0, tzinfo=UTC)
        bound = timestamp.process_bind_param(value, oracle)
        self.assertIsNone(bound.tzinfo)
        loaded = timestamp.process_result_value(bound, oracle)
        self.assertEqual(UTC, loaded.tzinfo)
        self.assertEqual(value, loaded)

    def test_naive_timestamp_is_rejected(self) -> None:
        timestamp = UniversalTimestamp(timezone=True)
        oracle = Mock(name="oracle")
        oracle.name = "oracle"
        with self.assertRaises(ValueError):
            timestamp.process_bind_param(datetime(2026, 7, 23), oracle)


if __name__ == "__main__":
    unittest.main()
