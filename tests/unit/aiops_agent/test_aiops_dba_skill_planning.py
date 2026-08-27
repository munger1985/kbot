"""专业 DBA Intent Router 与 Skill 规划框架测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from datetime import UTC, datetime

from aiops_agent.contracts.diagnosis import ModelInvocationReceipt
from aiops_agent.contracts.artifacts.database import (
    DatabaseDiagnosticResult,
    EvidenceGap,
)
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.application.turn_planning import TurnPlanningService
from aiops_agent.ports.model import StructuredModelResult
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.skill_handlers import DbaSkillInvocationHandler
from aiops_agent.skills import (
    CapabilityUnavailableError,
    DbaIntentRouter,
    DbaSkillPlanner,
    DbaSkillRegistry,
    SkillCatalogError,
    SkillExecutionSnapshotBuilder,
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
        required_privileges=("V_$INSTANCE", "V_$DATABASE"),
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
        privileges=("V_$INSTANCE", "V_$DATABASE"),
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
            endpoint_json={
                "host": "db.internal", "port": 1521, "service": "PDB1"
            },
            domain_id=7,
            row_version=1,
            capabilities_json={
                "capabilities": ["DB_READONLY"],
                "privileges": ["V_$INSTANCE", "V_$DATABASE"],
            },
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


class _FrozenToolExecutor:
    def __init__(self) -> None:
        self.task_keys = []

    async def execute(self, context):
        self.task_keys.append(context.task_key)
        tool_id = context.task_key.removeprefix("diagnostic:")
        if tool_id == "db.instance.identity":
            return DatabaseDiagnosticResult(
                target_id=context.target_id,
                tool_id=tool_id,
                status="SUCCEEDED",
            )
        return DatabaseDiagnosticResult(
            target_id=context.target_id,
            tool_id=tool_id,
            status="GAP",
            gap=EvidenceGap(
                code="PRIVILEGE_MISSING",
                tool_id=tool_id,
                detail="缺少最小只读权限",
            ),
        )


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
            execution_snapshot_builder=SkillExecutionSnapshotBuilder(
                skill_registry=registry,
                diagnostic_registry=DiagnosticRegistry.load(),
            ),
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
    def test_skill_handler_executes_only_frozen_tool_dag(self) -> None:
        executor = _FrozenToolExecutor()
        context = TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="skill:1:oracle.sql.top_current",
            target_id=str(uuid7()),
            agent_id=str(uuid7()),
            trigger_type="CHAT",
            trace_id="trace-skill",
            attempt=1,
            deadline_at=None,
            plan_snapshot={
                "skill_execution": {
                    "diagnostic_catalog_hash": "a" * 64,
                    "capability_snapshot_hash": "b" * 64,
                    "database": {
                        "domain_id": 7,
                        "db_type": "ORACLE",
                        "configured_version": "19c",
                        "target_row_version": 1,
                        "connection_profile": {},
                        "diagnostic_credential_id": str(uuid7()),
                    },
                    "invocations": {
                        "skill:1:oracle.sql.top_current": {
                            "skill_id": "oracle.sql.top_current",
                            "skill_version": "1.0.0",
                            "manifest_hash": "c" * 64,
                            "measurement_semantics": "CUMULATIVE_SINCE_LOAD",
                            "output_schema": "oracle.sql.top_current.output.v1",
                            "tools": [
                                {
                                    "step_id": "identity",
                                    "depends_on": [],
                                    "tool_id": "db.instance.identity",
                                    "tool_version": "1.0.0",
                                },
                                {
                                    "step_id": "top_sql",
                                    "depends_on": ["identity"],
                                    "tool_id": "db.sql.top_current",
                                    "tool_version": "1.0.0",
                                },
                            ],
                        }
                    },
                }
            },
            policy_snapshot={},
            input_artifacts=(),
        )

        result = self.run_async(
            DbaSkillInvocationHandler(
                database_handler=executor
            ).execute(context)
        )

        self.assertEqual("PARTIAL", result.status)
        self.assertEqual(
            [
                "diagnostic:db.instance.identity",
                "diagnostic:db.sql.top_current",
            ],
            executor.task_keys,
        )

    @staticmethod
    def run_async(awaitable):
        import asyncio

        return asyncio.run(awaitable)

    def test_repository_catalog_only_references_allowlisted_tools(self) -> None:
        tools = DiagnosticRegistry.load().tools
        registry = DbaSkillRegistry.load(
            allowed_tools=frozenset(
                (item.definition.tool_id, item.definition.version)
                for item in tools
            )
        )
        manifest = registry.latest("oracle.sql.top_current")
        validator = SkillExecutionSnapshotBuilder(
            skill_registry=registry,
            diagnostic_registry=DiagnosticRegistry.load(),
        )
        validator.validate_catalog()

        self.assertEqual(
            MeasurementSemantics.CUMULATIVE_SINCE_LOAD,
            manifest.measurement_semantics,
        )
        self.assertEqual(
            {
                "oracle.sql.top_current",
                "oracle.session.active",
                "oracle.session.blocking_chain",
                "oracle.storage.tablespace",
            },
            {item.skill_id for item in registry.manifests()},
        )
        self.assertEqual(64, len(registry.catalog_hash))

    def test_top_sql_plan_freezes_exact_tools_hashes_and_user_limit(self) -> None:
        diagnostics = DiagnosticRegistry.load()
        registry = DbaSkillRegistry.load(
            allowed_tools=frozenset(
                (item.definition.tool_id, item.definition.version)
                for item in diagnostics.tools
            )
        )
        intent = _intent_plan(DbaIntent.OBSERVE).model_copy(
            update={"subject": "TOP_SQL", "requested_limit": 20}
        )
        capabilities = _capabilities().model_copy(
            update={
                "target_capabilities": (
                    "DB_READONLY",
                    "dynamic_performance_views",
                ),
                "privileges": (
                    "V_$INSTANCE",
                    "V_$DATABASE",
                    "V_$SQLSTATS",
                ),
            }
        )
        plan = DbaSkillPlanner(registry).plan(
            intent=intent,
            capabilities=capabilities,
        )
        compiled = SkillPlanCompiler(registry).compile(plan)
        snapshot = SkillExecutionSnapshotBuilder(
            skill_registry=registry,
            diagnostic_registry=diagnostics,
        ).build(
            plan=plan,
            compiled=compiled,
            capabilities=capabilities,
            database_execution={
                "domain_id": 7,
                "target_row_version": 1,
                "db_type": "ORACLE",
                "configured_version": "19c",
                "connection_profile": {},
                "diagnostic_credential_id": "credential-1",
            },
        )

        invocation = snapshot["invocations"][
            "skill:1:oracle.sql.top_current"
        ]
        self.assertEqual(
            ["db.instance.identity", "db.sql.top_current"],
            [item["tool_id"] for item in invocation["tools"]],
        )
        self.assertEqual(20, invocation["tools"][1]["parameters"]["limit"])
        self.assertEqual(64, len(invocation["tools"][1]["template_sha256"]))

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
                endpoint_json={
                    "host": "db.internal",
                    "port": 1521,
                    "service": "PDB1",
                },
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
                allowed_tools=frozenset(
                    {("db.instance.identity", "1.0.0")}
                ),
            )

    def test_execution_snapshot_rejects_undeclared_tool_privileges(self) -> None:
        diagnostics = DiagnosticRegistry.load()
        manifest = _manifest(
            skill_id="oracle.identity.invalid",
            intent=DbaIntent.OBSERVE,
            domain=DbaDomain.SQL_PERFORMANCE,
        ).model_copy(update={"required_privileges": ()})
        registry = DbaSkillRegistry((manifest,))
        plan = DbaSkillPlanner(registry).plan(
            intent=_intent_plan(DbaIntent.OBSERVE),
            capabilities=_capabilities().model_copy(update={"privileges": ()}),
        )
        compiled = SkillPlanCompiler(registry).compile(plan)

        with self.assertRaisesRegex(SkillCatalogError, "未声明 Tool"):
            SkillExecutionSnapshotBuilder(
                skill_registry=registry,
                diagnostic_registry=diagnostics,
            ).build(
                plan=plan,
                compiled=compiled,
                capabilities=_capabilities(),
                database_execution={},
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
