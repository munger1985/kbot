"""专业 DBA Intent Router 与 Skill 规划框架测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from datetime import UTC, datetime

from aiops_agent.contracts.diagnosis import ModelInvocationReceipt
from aiops_agent.application.turn_planning import TurnPlanningService
from aiops_agent.ports.model import StructuredModelResult
from aiops_agent.skills import (
    CapabilityUnavailableError,
    DbaIntentRouter,
    DbaSkillPlanner,
    DbaSkillRegistry,
    SkillCatalogError,
    SkillPlanCompiler,
    SkillUnavailableError,
    build_capability_snapshot,
)
from platform_core.contracts.aiops.conversation import (
    DbaIntent,
    MeasurementSemantics,
)
from platform_core.contracts.aiops.skills import (
    DbaCapabilitySnapshot,
    DbaDomain,
    DbaIntentPlan,
    DbaSkillManifest,
    IntentCandidate,
    PresentationPreference,
    SkillLimits,
    SkillToolStep,
    SourceCapabilitySnapshot,
)
from platform_core.contracts.aiops.types import DatabaseType
from platform_core.identity import uuid7


_INTENT_DOMAINS = {
    DbaIntent.OBSERVE: DbaDomain.SQL_PERFORMANCE,
    DbaIntent.DIAGNOSE: DbaDomain.INSTANCE_PERFORMANCE,
    DbaIntent.EXPLAIN: DbaDomain.CONFIGURATION,
    DbaIntent.PLAN: DbaDomain.PATCH_AND_UPGRADE,
    DbaIntent.CHANGE: DbaDomain.SESSION_AND_LOCK,
    DbaIntent.VERIFY: DbaDomain.CONFIGURATION,
    DbaIntent.INSPECT: DbaDomain.MAINTENANCE,
}


def _intent_plan(intent: DbaIntent) -> DbaIntentPlan:
    return DbaIntentPlan(
        primary_intent=intent,
        candidates=(
            IntentCandidate(
                intent=intent,
                confidence=0.91,
                reason="用户目标明确",
            ),
        ),
        primary_domain=_INTENT_DOMAINS[intent],
        presentation_preference=PresentationPreference.TABLE,
    )


def _manifest(
    *,
    skill_id: str,
    intent: DbaIntent,
    domain: DbaDomain,
    tool_id: str = "db.instance.identity",
    required_target: tuple[str, ...] = ("DB_READONLY",),
    required_source: tuple[str, ...] = (),
) -> DbaSkillManifest:
    return DbaSkillManifest(
        skill_id=skill_id,
        version="1.0.0",
        database_types=(DatabaseType.ORACLE,),
        supported_intents=(intent,),
        domains=(domain,),
        required_source_capabilities=required_source,
        required_target_capabilities=required_target,
        input_schema=f"{skill_id}.input.v1",
        limits=SkillLimits(max_rows=50, timeout_seconds=20),
        tool_dag=(SkillToolStep(step_id="collect", tool_id=tool_id),),
        output_schema=f"{skill_id}.output.v1",
        measurement_semantics=MeasurementSemantics.CURRENT_ACTIVITY,
        presentation_kind=PresentationPreference.TABLE,
    )


def _capabilities(*, reachable: bool = True) -> DbaCapabilitySnapshot:
    return DbaCapabilitySnapshot(
        agent_id="agent-1",
        agent_version_id="agent-version-1",
        target_id="target-1",
        database_type=DatabaseType.ORACLE,
        database_version="19c",
        target_enabled=True,
        target_reachable=reachable,
        target_capabilities=("DB_READONLY",),
        source_snapshots=(
            SourceCapabilitySnapshot(
                source_id="prometheus-1",
                source_type="PROMETHEUS",
                enabled=True,
                reachable=True,
                capabilities=("PROMETHEUS_QUERY",),
            ),
        ),
    )


class _FakeModel:
    def __init__(self, output: DbaIntentPlan) -> None:
        self.output = output

    async def generate_structured(self, **kwargs) -> StructuredModelResult:
        digest = "a" * 64
        return StructuredModelResult(
            output=self.output,
            receipt=ModelInvocationReceipt(
                purpose=kwargs["purpose"],
                schema_id="DBA_INTENT_PLAN.v1",
                model_technical_name="test-model",
                model_revision="1",
                prompt_id=kwargs["prompt_ref"]["prompt_id"],
                prompt_version=kwargs["prompt_ref"]["prompt_version"],
                prompt_sha256=kwargs["prompt_ref"]["prompt_sha256"],
                input_sha256=digest,
                output_sha256=digest,
                duration_ms=1,
            ),
        )


class _PlanningUow:
    def __init__(self) -> None:
        self.turn = SimpleNamespace(
            turn_id=uuid7(),
            conversation_id=uuid7(),
            domain_id=7,
            status="PLANNING",
            event_cursor=2,
            intent_plan_artifact_id=None,
            skill_plan_artifact_id=None,
            primary_intent=None,
            primary_domain=None,
            subject=None,
        )
        self.version = SimpleNamespace(agent_version_id=uuid7())
        self.target = SimpleNamespace(
            target_id=uuid7(),
            db_type="ORACLE",
            version_code="19c",
            status="ENABLED",
            connectivity_status="CONNECTED",
            diagnostic_credential_id=uuid7(),
            execution_credential_id=None,
            capabilities_json={"capabilities": ["DB_READONLY"]},
        )
        self.run = SimpleNamespace(
            ops_run_id=uuid7(),
            agent_id=uuid7(),
            agent_version_id=self.version.agent_version_id,
            target_id=self.target.target_id,
            status="RUNNING",
            trace_id="trace-planning",
            deadline_at=None,
            plan_snapshot_json={},
        )
        self.message = SimpleNamespace(
            sequence_no=1,
            message_type="USER_MESSAGE",
            payload_json={"text": "查看当前 Top SQL"},
        )
        self.artifacts = []
        self.tasks = []
        self.invocations = []
        self.events = []
        self.commit_count = 0
        self.turns = SimpleNamespace(
            get_turn=self._get_turn,
            get_run_link=self._get_run_link,
            list_messages=self._list_messages,
            list_recent_conversation_messages=self._recent,
            add_skill_invocation=self._add_invocation,
            add_event=self._add_event,
        )
        self.runs = SimpleNamespace(
            get_run=self._get_run,
            add_artifact=self._add_artifact,
            add_tasks=self._add_tasks,
        )
        self.agents = SimpleNamespace(
            version=self._get_version,
            version_source_ids=self._source_ids,
        )
        self.targets = SimpleNamespace(get_scoped=self._get_target)
        self.diagnostic_sources = SimpleNamespace(get_scoped=self._get_source)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def commit(self):
        self.commit_count += 1

    async def _get_turn(self, *, domain_id, turn_id, lock=False):
        del lock
        return self.turn if domain_id == 7 and turn_id == self.turn.turn_id else None

    async def _get_run_link(self, *, turn_id, purpose):
        if turn_id == self.turn.turn_id and purpose == "PRIMARY":
            return SimpleNamespace(ops_run_id=self.run.ops_run_id)
        return None

    async def _list_messages(self, *, turn_id):
        return [self.message] if turn_id == self.turn.turn_id else []

    async def _recent(self, **_):
        return []

    async def _get_run(self, *, ops_run_id, lock=False):
        del lock
        return self.run if ops_run_id == self.run.ops_run_id else None

    async def _get_version(self, *, agent_id, agent_version_id):
        if agent_id == self.run.agent_id and agent_version_id == self.version.agent_version_id:
            return self.version
        return None

    async def _source_ids(self, **_):
        return []

    async def _get_target(self, *, target_id, domain_id):
        if target_id == self.target.target_id and domain_id == 7:
            return self.target
        return None

    async def _get_source(self, **_):
        return None

    async def _add_artifact(self, row):
        self.artifacts.append(row)
        return row

    async def _add_tasks(self, rows):
        self.tasks.extend(rows)
        return rows

    async def _add_invocation(self, row):
        self.invocations.append(row)
        return row

    async def _add_event(self, row):
        row.created_at = datetime.now(UTC)
        self.events.append(row)
        return row


class _AgentCatalog:
    async def resolve_diagnosis_model(self, **_):
        return {"technical_name": "test-model", "revision": "1"}


class DbaIntentRouterTest(unittest.IsolatedAsyncioTestCase):
    async def test_all_seven_primary_intents_pass_structured_router(self) -> None:
        for intent in DbaIntent:
            with self.subTest(intent=intent.value):
                plan = _intent_plan(intent)
                result = await DbaIntentRouter(_FakeModel(plan)).route(
                    question="测试问题",
                    conversation_context=(),
                    model_snapshot={
                        "technical_name": "test-model",
                        "revision": "1",
                    },
                    deadline=None,
                    idempotency_key=f"intent:{intent.value}",
                )
                self.assertEqual(intent, result.output.primary_intent)

    async def test_turn_planning_persists_frozen_plans_tasks_and_invocations(self) -> None:
        uow = _PlanningUow()
        intent = _intent_plan(DbaIntent.OBSERVE)
        manifest = _manifest(
            skill_id="oracle.sql.current",
            intent=DbaIntent.OBSERVE,
            domain=DbaDomain.SQL_PERFORMANCE,
        )
        registry = DbaSkillRegistry((manifest,))
        service = TurnPlanningService(
            uow_factory=lambda: uow,
            intent_router=DbaIntentRouter(_FakeModel(intent)),
            skill_planner=DbaSkillPlanner(registry),
            skill_compiler=SkillPlanCompiler(registry),
            agent_catalog=_AgentCatalog(),
        )

        result = await service.execute(
            {"domain_id": 7, "turn_id": str(uow.turn.turn_id)}
        )

        self.assertEqual("COLLECTING", result["status"])
        self.assertEqual("COLLECTING", uow.turn.status)
        self.assertEqual(2, len(uow.artifacts))
        self.assertEqual(3, len(uow.tasks))
        self.assertEqual(1, len(uow.invocations))
        self.assertEqual(
            uow.invocations[0].manifest_hash,
            uow.turn.skill_plan_json["items"][0]["manifest_hash"],
        )
        self.assertEqual(
            ["intent.updated", "skill.plan.created", "turn.status"],
            [event.event_type for event in uow.events],
        )
        replay = await service.execute(
            {"domain_id": 7, "turn_id": str(uow.turn.turn_id)}
        )
        self.assertEqual(result, replay)
        self.assertEqual(2, len(uow.artifacts))
        self.assertEqual(3, len(uow.tasks))


class DbaSkillFrameworkTest(unittest.TestCase):
    def test_capability_snapshot_uses_only_enabled_reachable_resources(self) -> None:
        snapshot = build_capability_snapshot(
            agent_id="agent-1",
            agent_version=SimpleNamespace(agent_version_id="version-1"),
            target=SimpleNamespace(
                target_id="target-1",
                db_type="ORACLE",
                version_code="19c",
                status="ENABLED",
                connectivity_status="CONNECTED",
                diagnostic_credential_id="credential-1",
                execution_credential_id=None,
                capabilities_json={
                    "capabilities": ["DB_SQL_STATS"],
                    "privileges": ["ORACLE_SELECT_V_SQLSTATS"],
                },
            ),
            sources=(
                SimpleNamespace(
                    diagnostic_source_id="source-1",
                    source_type="PROMETHEUS",
                    status="ENABLED",
                    connectivity_status="CONNECTED",
                    declared_capabilities_json={
                        "capabilities": ["PROMETHEUS_QUERY"]
                    },
                    discovered_capabilities_json={
                        "metric.query_range": {"supported": True}
                    },
                ),
                SimpleNamespace(
                    diagnostic_source_id="source-2",
                    source_type="LOKI",
                    status="DISABLED",
                    connectivity_status="CONNECTED",
                    declared_capabilities_json={
                        "capabilities": ["LOKI_QUERY"]
                    },
                    discovered_capabilities_json=None,
                ),
            ),
        )

        self.assertIn("DB_READONLY", snapshot.target_capabilities)
        self.assertIn("DB_SQL_STATS", snapshot.target_capabilities)
        self.assertEqual(
            frozenset({"PROMETHEUS_QUERY", "metric.query_range"}),
            snapshot.available_source_capabilities,
        )

    def test_registry_hash_is_independent_of_registration_order(self) -> None:
        first = _manifest(
            skill_id="oracle.sql.current",
            intent=DbaIntent.OBSERVE,
            domain=DbaDomain.SQL_PERFORMANCE,
        )
        second = _manifest(
            skill_id="oracle.instance.diagnose",
            intent=DbaIntent.DIAGNOSE,
            domain=DbaDomain.INSTANCE_PERFORMANCE,
        )
        left = DbaSkillRegistry((first, second))
        right = DbaSkillRegistry((second, first))

        self.assertEqual(left.catalog_hash, right.catalog_hash)
        self.assertEqual(
            left.manifest_hash(first.skill_id, first.version),
            right.manifest_hash(first.skill_id, first.version),
        )

    def test_registry_rejects_tool_outside_executor_catalog(self) -> None:
        with self.assertRaisesRegex(SkillCatalogError, "目录外 Tool"):
            DbaSkillRegistry(
                (
                    _manifest(
                        skill_id="oracle.sql.current",
                        intent=DbaIntent.OBSERVE,
                        domain=DbaDomain.SQL_PERFORMANCE,
                        tool_id="db.dynamic.sql",
                    ),
                ),
                allowed_tool_ids=frozenset({"db.instance.identity"}),
            )

    def test_planner_is_replayable_and_compiles_traceable_tasks(self) -> None:
        manifest = _manifest(
            skill_id="oracle.sql.current",
            intent=DbaIntent.OBSERVE,
            domain=DbaDomain.SQL_PERFORMANCE,
        )
        registry = DbaSkillRegistry((manifest,))
        planner = DbaSkillPlanner(registry)

        first = planner.plan(
            intent=_intent_plan(DbaIntent.OBSERVE),
            capabilities=_capabilities(),
        )
        second = planner.plan(
            intent=_intent_plan(DbaIntent.OBSERVE),
            capabilities=_capabilities(),
        )
        compiled = SkillPlanCompiler(registry).compile(first)

        self.assertEqual(first, second)
        self.assertEqual(
            registry.manifest_hash(manifest.skill_id, manifest.version),
            first.items[0].manifest_hash,
        )
        self.assertEqual("SKILL_INVOKE", compiled.tasks[0].task_type)
        self.assertEqual("EVIDENCE_ASSESS", compiled.tasks[-2].task_type)
        self.assertEqual("ANSWER", compiled.tasks[-1].task_type)

    def test_planner_rejects_unknown_skill_and_missing_capability(self) -> None:
        manifest = _manifest(
            skill_id="oracle.sql.window",
            intent=DbaIntent.OBSERVE,
            domain=DbaDomain.SQL_PERFORMANCE,
            required_source=("PROMETHEUS_QUERY",),
        )
        planner = DbaSkillPlanner(DbaSkillRegistry((manifest,)))

        with self.assertRaises(SkillUnavailableError):
            planner.plan(
                intent=_intent_plan(DbaIntent.OBSERVE),
                capabilities=_capabilities(),
                suggested_skill_ids=("oracle.not-in-catalog",),
            )
        unavailable = _capabilities().model_copy(
            update={"source_snapshots": ()}
        )
        with self.assertRaises(CapabilityUnavailableError):
            planner.plan(
                intent=_intent_plan(DbaIntent.OBSERVE),
                capabilities=unavailable,
                suggested_skill_ids=(manifest.skill_id,),
            )


if __name__ == "__main__":
    unittest.main()
