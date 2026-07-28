"""步骤 9 Action Catalog、参数血缘和 Advisory 降级测试。"""

from __future__ import annotations

import asyncio
import unittest

from aiops_agent.actions import ActionRegistry, ActionRenderer
from aiops_agent.actions.validation import validate_rendered_action
from aiops_agent.contracts.diagnosis import (
    EvidenceFact,
    EvidenceIndex,
    RootCauseAssessment,
    SolutionDraft,
)
from aiops_agent.workers.change_handlers import (
    ActionPlanHandler,
    ProposalSnapshotHandler,
)
from aiops_agent.workers.handlers import TaskExecutionContext


class ActionCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ActionRegistry.load()

    def test_registry_has_exact_oracle_and_mysql_variants(self) -> None:
        self.assertEqual(len(self.registry.templates), 2)
        oracle = self.registry.resolve(
            action_template_id="db.session.terminate",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"session_management"},
            entitlements=set(),
            environment="PROD",
        )
        rendered = ActionRenderer().render(
            oracle,
            {"session_id": 42, "serial_number": 9, "instance_id": 1},
        )
        self.assertEqual(
            rendered.command_text,
            "ALTER SYSTEM DISCONNECT SESSION '42,9,@1' IMMEDIATE",
        )

    def test_renderer_rejects_unregistered_command_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "Allowlist"):
            validate_rendered_action(
                "ALTER SYSTEM SET open_cursors=999",
                db_type="ORACLE",
            )


class ActionPlanHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ActionRegistry.load()

    def test_source_verified_blocker_becomes_advisory_plan(self) -> None:
        context = self._context()
        plan = asyncio.run(
            ActionPlanHandler(
                registry=self.registry,
                execution_enabled=False,
            ).execute(context)
        )
        self.assertEqual(plan.decision, "ADVISORY")
        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(
            plan.actions[0].canonical_parameters,
            {"session_id": 42, "serial_number": 9, "instance_id": 1},
        )
        proposal = asyncio.run(
            ProposalSnapshotHandler().execute(
                context.__class__(
                    **{
                        **context.__dict__,
                        "task_id": "proposal-task",
                        "task_key": "change:proposal",
                        "input_artifacts": (
                            {
                                "schema_version": "ACTION_PLAN.v1",
                                "payload": plan.model_dump(mode="json"),
                            },
                        ),
                    }
                )
            )
        )
        self.assertEqual(proposal.status, "CREATED")
        self.assertEqual(proposal.proposal.mode, "ADVISORY")

    def test_user_only_result_cannot_supply_action_parameters(self) -> None:
        context = self._context(user_only=True)
        plan = asyncio.run(
            ActionPlanHandler(
                registry=self.registry,
                execution_enabled=True,
            ).execute(context)
        )
        self.assertEqual(plan.decision, "NO_ACTION")
        self.assertIn(
            "VERIFIED_ACTION_PARAMETERS_UNAVAILABLE",
            plan.decision_reasons,
        )

    def _context(self, *, user_only: bool = False) -> TaskExecutionContext:
        trust = "USER_PROVIDED" if user_only else "SOURCE_VERIFIED"
        blocker = EvidenceFact(
            fact_id="1" * 64,
            source_artifact_id="artifact-blocking",
            source_json_pointer="/rows/0",
            source_type=(
                "USER_RESULT" if user_only else "DATABASE_OBSERVATION"
            ),
            source_group_id="database:blocking",
            trust_level=trust,
            target_id="target-1",
            observed_subject="target-1",
            metric_or_fact_type="db.session.blocking_chain",
            value={
                "blocking_session_id": 42,
                "blocking_instance_id": 1,
            },
            fact_summary="会话 42 正在阻塞其他会话",
        )
        active = EvidenceFact(
            fact_id="2" * 64,
            source_artifact_id="artifact-active",
            source_json_pointer="/rows/0",
            source_type=(
                "USER_RESULT" if user_only else "DATABASE_OBSERVATION"
            ),
            source_group_id="database:active",
            trust_level=trust,
            target_id="target-1",
            observed_subject="target-1",
            metric_or_fact_type="db.session.active",
            value={
                "session_id": 42,
                "serial_number": 9,
                "instance_id": 1,
            },
            fact_summary="活动会话 42",
        )
        evidence = EvidenceIndex(
            target_id="target-1",
            facts=(blocker, active),
            fact_count=2,
            source_group_count=2,
            index_hash="3" * 64,
        )
        root = RootCauseAssessment(
            target_id="target-1",
            suggested_level="PROBABLE",
            eligible_ceiling="PROBABLE",
            effective_level="PROBABLE",
            primary_hypothesis_key="blocking-session",
            supporting_fact_refs=(blocker.fact_id, active.fact_id),
        )
        solution = SolutionDraft(
            immediate_mitigations=("终止已确认的阻塞会话",),
        )
        return TaskExecutionContext(
            run_id="run-1",
            task_id="action-task",
            task_key="change:action-plan",
            target_id="target-1",
            agent_id="agent-1",
            trigger_type="CHAT",
            actor_id="user-1",
            original_request="处理阻塞",
            trace_id="trace-1",
            attempt=1,
            deadline_at=None,
            plan_snapshot={
                "target": {
                    "db_type": "ORACLE",
                    "version_code": "19.0.0",
                    "environment": "PROD",
                    "row_version": 3,
                    "capabilities": {"session_management": True},
                },
                "binding": {
                    "allow_mutation": True,
                    "allowed_actions": ["db.session.terminate"],
                },
            },
            policy_snapshot={
                "rules": {"allow_agent_execution": True}
            },
            input_artifacts=(
                {
                    "schema_version": "EVIDENCE_INDEX.v1",
                    "payload": evidence.model_dump(mode="json"),
                },
                {
                    "schema_version": "ROOT_CAUSE_ASSESSMENT.v1",
                    "payload": root.model_dump(mode="json"),
                },
                {
                    "schema_version": "SOLUTION_DRAFT.v1",
                    "payload": solution.model_dump(mode="json"),
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
