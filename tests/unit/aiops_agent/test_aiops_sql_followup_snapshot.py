"""真实 Run 创建、Top SQL 编译到选行诊断的跨项目闭环回归。"""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from aiops_agent.application.conversation_starters import ConversationStarterCatalog
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.application.investigation.evidence_scope import RUN_EVIDENCE_SCOPE_FIELDS
from aiops_agent.application.sql_selection import resolve_sql_selection, selection_ref
from aiops_agent.application.turn_queue import TurnQueueService
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.playbooks import PlaybookRegistry, canonical_hash
from aiops_agent.tools import InvestigationTaskCompiler, ToolExecutionSnapshotBuilder
from tests.support.aiops_sql_evidence import backed_sql_fact
from tests.unit.aiops_agent import test_aiops_dba_playbook_planning as planning_support
from tests.unit.aiops_agent import test_aiops_turn_application as turn_support
from tests.unit.aiops_agent.test_aiops_sql_diagnosis import observation, selection_scope


ENGINES = ("ORACLE", "MYSQL", "POSTGRESQL")
SQL_IDENTITIES = {
    "ORACLE": ("sql_id", "6tjx7su0q5ttj", "db.sql.cursor_details"),
    "MYSQL": ("digest", "a" * 64, "db.mysql.statement.digest.top"),
    "POSTGRESQL": ("query_id", "1285382359565720997", "db.postgresql.statement.top"),
}


class SqlFollowUpSnapshotTest(unittest.IsolatedAsyncioTestCase):
    async def _routed(self, engine):
        uow = turn_support._Uow()
        uow.target.db_type = engine
        uow.target.row_version = 409
        await turn_support.ConversationTurnApplicationTest._start(self, uow)
        await TurnQueueService(uow_factory=lambda: uow).accept_created(
            dict(uow.outbox.rows[0].payload_json)
        )
        payload = dict(uow.outbox.rows[1].payload_json)
        planner = turn_support._turn_planner(uow)
        await planner.begin(payload)
        return uow, planner, payload

    async def _created(self, engine):
        uow, planner, payload = await self._routed(engine)
        await planner.start_active_diagnosis(payload)
        return uow, planner, payload

    async def _compile_top_sql(self, created):
        uow = planning_support._PlanningUow()
        uow.target.target_id = created.target.target_id
        uow.target.db_type = created.target.db_type
        uow.target.version_code = {
            "ORACLE": "19c", "MYSQL": "8.0.36", "POSTGRESQL": "16.15",
        }[created.target.db_type]
        uow.target.row_version = created.target.row_version
        uow.run.target_id = uow.target.target_id
        uow.run.plan_snapshot_json = deepcopy(created.runs.rows[0].plan_snapshot_json)
        catalog = ConversationStarterCatalog()
        uow.run.plan_snapshot_json["client_metadata"]["conversation_starter"] = catalog.freeze_natural_language(
            capability_id="database.sql.top-current", parameters={"limit": 10},
        )
        diagnostics = DiagnosticRegistry.load()
        registry = PlaybookRegistry.load(
            allowed_tools=frozenset(
                (item.definition.tool_id, item.definition.version)
                for item in diagnostics.tools
            )
        )
        service = TurnPlanningService(
            uow_factory=lambda: uow,
            investigation_reasoner=planning_support._PastedLogReasoner(),
            playbook_registry=registry,
            task_compiler=InvestigationTaskCompiler(registry),
            tool_snapshot_builder=ToolExecutionSnapshotBuilder(
                playbook_registry=registry, diagnostic_registry=diagnostics,
            ),
            agent_catalog=planning_support._AgentCatalog(),
        )
        result = await service.execute(
            {"domain_id": str(uow.run.domain_id), "turn_id": str(uow.turn.turn_id)}
        )
        self.assertEqual("COLLECTING", result["status"])
        return uow.run.plan_snapshot_json

    def _selection(self, snapshot, engine):
        fixture = selection_scope()
        fixture.target.target_id = snapshot["target"]["target_id"]
        fixture.target.row_version = snapshot["target"]["row_version"]
        fixture.target.db_type = engine
        fixture.run.target_id = fixture.target.target_id
        fixture.run.plan_snapshot_json = deepcopy(snapshot)
        name, value, tool = SQL_IDENTITIES[engine]
        raw, fact = backed_sql_fact(
            observation(
                names=(name, "execution_count"), rows=((value, 100),), tool_id=tool,
            ),
            fixture.run,
        )
        fixture.artifact.payload_json = DbaSufficiencyAssessment(
            status="ANSWERABLE", evidence=(fact,),
        ).model_dump(mode="json")
        fixture.artifact.content_hash = canonical_hash(fixture.artifact.payload_json)
        artifacts = {fixture.artifact.artifact_id: fixture.artifact, raw.artifact_id: raw}
        fixture.uow.runs.get_artifact = AsyncMock(
            side_effect=lambda *, artifact_id: artifacts.get(artifact_id)
        )
        fixture.reference = selection_ref(
            artifact_id=str(fixture.artifact.artifact_id), fact=fact, row_index=0,
        )
        return fixture

    async def _resolve(self, fixture):
        return await resolve_sql_selection(
            uow=fixture.uow, reference=fixture.reference, conversation=fixture.conversation,
            actor_id=fixture.actor, now=fixture.now,
        )

    async def test_created_and_compiled_top_sql_snapshot_can_authorize_followup(self):
        for engine in ENGINES:
            with self.subTest(engine=engine):
                created, _, _ = await self._created(engine)
                expected = {
                    "target_id": str(created.target.target_id),
                    "row_version": 409, "db_type": engine,
                }
                self.assertEqual(expected, created.runs.rows[0].plan_snapshot_json["target"])
                snapshot = await self._compile_top_sql(created)
                self.assertEqual(expected, snapshot["target"])
                fixture = self._selection(snapshot, engine)
                selected = await self._resolve(fixture)
                name, value, _ = SQL_IDENTITIES[engine]
                self.assertEqual(value, selected["subject_ref"][name])
                self.assertEqual("409", selected["target_row_version"])

    async def test_retry_does_not_refresh_the_original_target_identity(self):
        for engine in ENGINES:
            with self.subTest(engine=engine):
                uow, planner, payload = await self._created(engine)
                frozen = deepcopy(uow.runs.rows[0].plan_snapshot_json["target"])
                uow.target.row_version += 1
                await planner.start_active_diagnosis(payload)
                self.assertEqual(1, len(uow.runs.rows))
                self.assertEqual(frozen, uow.runs.rows[0].plan_snapshot_json["target"])

    async def test_target_changes_still_invalidate_selected_evidence(self):
        created, _, _ = await self._created("ORACLE")
        snapshot = await self._compile_top_sql(created)
        for violation in ("version", "disabled", "missing_snapshot", "revoked"):
            with self.subTest(violation=violation):
                fixture = self._selection(snapshot, "ORACLE")
                if violation == "version":
                    fixture.target.row_version += 1
                elif violation == "disabled":
                    fixture.target.status = "DISABLED"
                elif violation == "missing_snapshot":
                    fixture.run.plan_snapshot_json.pop("target")
                else:
                    fixture.uow.agents.version_has_target.return_value = False
                with self.assertRaises((ValueError, PermissionError)) as raised:
                    await self._resolve(fixture)
                if violation == "missing_snapshot":
                    self.assertIn("重新执行 Top SQL", str(raised.exception))

    async def test_changed_or_missing_target_cannot_start_new_primary_run(self):
        for violation in ("disabled", "missing", "selection_version"):
            with self.subTest(violation=violation):
                uow, planner, payload = await self._routed("ORACLE")
                if violation == "disabled":
                    uow.target.status = "DISABLED"
                elif violation == "missing":
                    uow.targets.get_scoped = AsyncMock(return_value=None)
                else:
                    payload["execution_context"]["sql_selection"] = {
                        "target_id": str(uow.target.target_id), "target_row_version": "408",
                    }
                with self.assertRaises(AIOpsApplicationError):
                    await planner.start_active_diagnosis(payload)
                self.assertEqual([], uow.runs.rows)

    async def test_history_can_be_loaded_before_first_plan_is_committed(self):
        uow, _, _ = await self._created("POSTGRESQL")
        turn = uow.turns.turns[0]
        turn.turn_no = 2
        run = uow.runs.rows[0]
        service = object.__new__(TurnPlanningService)
        service._uow_factory = lambda: uow
        context = SimpleNamespace(
            turn_id=turn.turn_id, ops_run_id=run.ops_run_id,
            database_execution={"target_row_version": 409},
            **{key: getattr(run, key) for key in RUN_EVIDENCE_SCOPE_FIELDS},
        )
        reader = AsyncMock(return_value={"sources": []})
        with patch("aiops_agent.application.investigation.service.load_prior_sql_evidence", reader):
            result = await service._load_prior_sql_evidence(context, task_frame={})
        self.assertEqual({"sources": []}, result)
        reader.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
