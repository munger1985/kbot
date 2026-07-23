"""AIOps 步骤 4 确定性运行内核测试。"""

import unittest
from datetime import UTC, datetime
from unittest.mock import Mock

from pydantic import ValidationError

from aiops_agent.domain.operations import (
    InvalidOperationTransition,
    ensure_run_transition,
    ensure_task_transition,
)
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
from platform_core.contracts.aiops import ArtifactInput
from platform_core.persistence.orm import UniversalTimestamp


class RuntimeStateMachineTest(unittest.TestCase):
    def test_step4_happy_path_transitions_are_explicit(self) -> None:
        ensure_run_transition(
            DomainOpsRunStatus.CREATED,
            DomainOpsRunStatus.SCOPING,
        )
        ensure_run_transition(
            DomainOpsRunStatus.SCOPING,
            DomainOpsRunStatus.OBSERVING,
        )
        ensure_run_transition(
            DomainOpsRunStatus.OBSERVING,
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

    def test_terminal_state_cannot_restart(self) -> None:
        with self.assertRaises(InvalidOperationTransition):
            ensure_run_transition(
                DomainOpsRunStatus.COMPLETED,
                DomainOpsRunStatus.SCOPING,
            )
        with self.assertRaises(InvalidOperationTransition):
            ensure_task_transition(
                DomainOpsTaskStatus.SUCCEEDED,
                DomainOpsTaskStatus.RUNNING,
            )


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
