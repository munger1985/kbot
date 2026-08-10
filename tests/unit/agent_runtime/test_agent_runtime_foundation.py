"""Agent Runtime 状态机、计划、Skill 与服务边界测试。"""

import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from agent_runtime.domain import (
    ExecutionKind,
    ExecutionMode,
    InvalidStateTransition,
    PlanDraft,
    PlanLimits,
    PlanValidationError,
    PlanValidator,
    RunStatus,
    SkillManifest,
    SkillRegistry,
    TaskSpec,
    TaskStatus,
    ensure_run_transition,
    ensure_task_transition,
)
from agent_runtime.domain.skills import (
    ArtifactDeclaration,
    DataClassification,
)
from platform_core.contracts import CreateAgentRunRequest


class AgentStateMachineTest(unittest.TestCase):
    def test_allowed_transition_passes(self):
        ensure_run_transition(RunStatus.CREATED, RunStatus.RUNNING)
        ensure_task_transition(TaskStatus.READY, TaskStatus.RUNNING)

    def test_terminal_state_cannot_transition(self):
        with self.assertRaises(InvalidStateTransition):
            ensure_run_transition(RunStatus.COMPLETED, RunStatus.RUNNING)
        with self.assertRaises(InvalidStateTransition):
            ensure_task_transition(TaskStatus.SUCCEEDED, TaskStatus.READY)


class AgentPlanTest(unittest.TestCase):
    @staticmethod
    def _local_task(
        task_key: str,
        *,
        depends_on: tuple[str, ...] = (),
        outputs: tuple[str, ...] = ("GROUNDED_ANSWER",),
    ) -> TaskSpec:
        return TaskSpec(
            task_key=task_key,
            task_type="RETRIEVE",
            execution_kind=ExecutionKind.LOCAL_SKILL,
            specialist="knowledge",
            skill_id="knowledge-retrieval",
            skill_version="1.0.0",
            depends_on=depends_on,
            expected_outputs=outputs,
            timeout_seconds=30,
            execution_mode=ExecutionMode.READ_ONLY,
        )

    def _validator(self) -> PlanValidator:
        return PlanValidator(
            skill_exists=lambda skill_id, version: (
                skill_id, version
            ) == ("knowledge-retrieval", "1.0.0"),
            capability_exists=lambda service, capability: False,
            public_artifact_types={"GROUNDED_ANSWER"},
        )

    def test_valid_plan_passes(self):
        plan = PlanDraft(
            plan_version="1",
            objective="检索并回答",
            tasks=(self._local_task("retrieve"),),
            final_task_key="retrieve",
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )
        self._validator().validate(plan, PlanLimits())

    def test_cycle_is_rejected_with_stable_code(self):
        plan = PlanDraft(
            plan_version="1",
            objective="循环计划",
            tasks=(
                self._local_task("first", depends_on=("second",)),
                self._local_task("second", depends_on=("first",)),
            ),
            final_task_key="second",
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )
        with self.assertRaises(PlanValidationError) as context:
            self._validator().validate(plan, PlanLimits())
        self.assertEqual("PLAN_CYCLE", context.exception.code)

    def test_final_task_must_join_all_required_branches(self):
        plan = PlanDraft(
            plan_version="1",
            objective="遗漏并行分支",
            tasks=(
                self._local_task("first"),
                self._local_task("final"),
            ),
            final_task_key="final",
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )
        with self.assertRaises(PlanValidationError) as context:
            self._validator().validate(plan, PlanLimits())
        self.assertEqual(
            "PLAN_FINAL_TASK_INCOMPLETE", context.exception.code
        )

    def test_local_skill_cannot_declare_delegate(self):
        with self.assertRaises(ValidationError):
            TaskSpec(
                task_key="invalid",
                task_type="RETRIEVE",
                execution_kind=ExecutionKind.LOCAL_SKILL,
                skill_id="knowledge-retrieval",
                skill_version="1.0.0",
                delegate_service="aiops-agent",
                delegate_capability="diagnose",
                timeout_seconds=30,
                execution_mode=ExecutionMode.READ_ONLY,
            )


class SkillRegistryTest(unittest.TestCase):
    @staticmethod
    def _manifest() -> SkillManifest:
        return SkillManifest(
            skill_id="knowledge-retrieval",
            version="1.0.0",
            owner="agent-runtime",
            specialist="knowledge",
            description="从授权 Collection 检索可引用证据",
            input_schema="KnowledgeRetrievalInput.v1",
            output_artifacts=(
                ArtifactDeclaration(
                    artifact_type="CITATION_PACK",
                    schema_version="CitationPack.v2",
                ),
            ),
            permissions=("knowledge.discovery.read",),
            execution_mode=ExecutionMode.READ_ONLY,
            idempotent=True,
            timeout_seconds=30,
            max_retries=2,
            data_classification=DataClassification.INTERNAL,
            external_dependencies=("knowledge_core_api",),
        )

    def test_registry_requires_unique_explicit_version(self):
        registry = SkillRegistry()
        implementation = lambda: None
        registry.register(self._manifest(), implementation)
        self.assertTrue(registry.contains("knowledge-retrieval", "1.0.0"))
        with self.assertRaisesRegex(ValueError, "已注册"):
            registry.register(self._manifest(), implementation)


class AgentContractAndSchemaTest(unittest.TestCase):
    def test_create_request_does_not_accept_domain_from_body(self):
        with self.assertRaises(ValidationError):
            CreateAgentRunRequest(
                agent_id=uuid4(),
                input="测试",
                domain_id=100,
            )

    def test_agent_schema_is_owned_by_service_directory(self):
        root = Path(__file__).resolve().parents[3]
        schema_dir = root / "database" / "oracle" / "agent_runtime"
        sql = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(schema_dir.glob("*.sql"))
        )
        for table in (
            "KBOT_AGENT_RUN",
            "KBOT_AGENT_TASK",
            "KBOT_AGENT_ARTIFACT",
            "KBOT_AGENT_RUN_EVENT",
            "KBOT_AGENT_DELEGATION",
        ):
            self.assertIn(f"CREATE TABLE {table}", sql)
        self.assertNotIn("KBOT_MD_", sql)
        self.assertNotIn("INSERT INTO", sql.upper())

    def test_runtime_api_exposes_only_internal_run_commands(self):
        from agent_runtime.entrypoints.api import app

        paths = {route.path for route in app.routes}
        self.assertIn("/health", paths)
        self.assertIn("/readyz", paths)
        self.assertFalse(any(path.startswith("/api/") for path in paths))
        self.assertIn("/internal/v1/runs", paths)
        self.assertNotIn("/internal/v1/agents", paths)
        self.assertIn("/internal/v1/tasks/claim", paths)


if __name__ == "__main__":
    unittest.main()
