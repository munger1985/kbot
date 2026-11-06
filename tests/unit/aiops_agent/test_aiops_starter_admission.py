"""报告功能入口、真实能力准入和固定规划的跨项目回归。"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.application.conversation_starters import ConversationStarterCatalog
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.investigation.discovery import (
    available_tools,
    reset_model_deferred_flags,
    rewrite_incomplete_discovery_actions,
    select_planning_candidates,
)
from aiops_agent.application.investigation.errors import TurnPlanningStageError
from aiops_agent.application.investigation.discovery_binding import bind_discovery_parameters
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.application.turns import ConversationTurnService
from aiops_agent.contracts.tool_execution import DbaToolResult, ToolOutcome
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.diagnostics.grants import DiagnosticGrantCodec, canonical_sha256
from aiops_agent.executor import DiagnosticExecutorService
from aiops_agent.executor.drivers import DriverQueryResult
from aiops_agent.playbooks import PlaybookRegistry
from aiops_agent.tools import ToolExecutionSnapshotBuilder
from aiops_agent.tools.capabilities import build_capability_snapshot
from aiops_agent.tools.compiler import InvestigationTaskCompiler
from platform_core.contracts.aiops import ConversationStarterSelection
from platform_core.contracts.aiops.executor import (
    DiagnosticConnectionProfile, DiagnosticExecutionGrant, DiagnosticLimits,
    ReadDiagnosticRequest,
)
from platform_core.contracts.aiops.playbooks import DbaPlaybookPlan
from platform_core.identity import uuid7
from tests.support.aiops_resources import bind_resource_control


REPORT_PARAMETERS = {
    "oracle.report.awr": {
        "begin_time": "2026-11-05T01:00:00.000Z",
        "end_time": "2026-11-05T02:00:00.000Z",
        "timezone": "Asia/Shanghai",
    },
    "oracle.report.awr-diff": {
        "first_begin_time": "2026-11-05T01:00:00.000Z",
        "first_end_time": "2026-11-05T02:00:00.000Z",
        "second_begin_time": "2026-11-05T03:00:00.000Z",
        "second_end_time": "2026-11-05T04:00:00.000Z",
        "timezone": "Asia/Shanghai",
    },
    "oracle.report.ash": {
        "begin_time": "2026-11-05T01:00:00.000Z",
        "end_time": "2026-11-05T02:00:00.000Z",
        "timezone": "Asia/Shanghai",
    },
}


def _target(**overrides):
    values = {
        "target_id": uuid7(),
        "display_name": "报告目标",
        "db_type": "ORACLE",
        "version_code": "19c",
        "status": "ENABLED",
        "readonly_connection_enabled": True,
        "connectivity_status": "CONNECTED",
        "capabilities_json": {
            "capabilities": ["DB_READONLY"],
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class StarterAdmissionTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.diagnostics = DiagnosticRegistry.load()
        cls.builder = ToolExecutionSnapshotBuilder(
            playbook_registry=PlaybookRegistry.load(),
            diagnostic_registry=cls.diagnostics,
        )

    def setUp(self):
        self.catalog = ConversationStarterCatalog()
        self.service = object.__new__(TurnPlanningService)
        self.service._record_planning_route = AsyncMock()
        self.service._tool_snapshot_builder = self.builder

    def _starters(self, *targets):
        return {
            item["starter_id"]: item
            for item in self.catalog.list_for_agent(
                agent_id=uuid7(),
                capabilities={"ACTIVE_DIAGNOSIS"},
                targets=targets,
            )["starters"]
        }

    def _context(self, starter_id, target):
        frozen = self.catalog.freeze_natural_language(
            capability_id=starter_id, parameters=REPORT_PARAMETERS[starter_id],
        )
        capabilities = build_capability_snapshot(
            agent_id=uuid7(),
            agent_version=SimpleNamespace(agent_version_id=uuid7()),
            target=target,
            sources=(),
        )
        return SimpleNamespace(
            question=frozen["user_message"],
            target_context={"db_type": "ORACLE", "display_name": target.display_name},
            conversation_starter=frozen,
            capabilities=capabilities,
        )

    async def _plan(self, context):
        return await self.service._plan_conversation_starter(
            context=context,
            available_tools=available_tools(self.builder, context.capabilities),
            available_playbooks=(),
        )

    def test_oracle_report_shortcuts_do_not_depend_on_license_declarations(self):
        for target in (
            _target(),
            _target(capabilities_json={"entitlements": "ORACLE_DIAGNOSTICS_PACK"}),
            _target(capabilities_json={"privileges": ["ORACLE_DIAGNOSTICS_PACK"]}),
        ):
            starters = self._starters(target)
            for starter_id in REPORT_PARAMETERS:
                with self.subTest(starter_id=starter_id, target=target.capabilities_json):
                    self.assertEqual("AVAILABLE", starters[starter_id]["status"])
                    self.assertIsNone(starters[starter_id]["availability_reason"])
            self.assertEqual("AVAILABLE", starters["database.performance.current"]["status"])

    def test_reports_remain_oracle_only(self):
        starters = self._starters(_target())
        for starter_id in REPORT_PARAMETERS:
            with self.subTest(starter_id=starter_id):
                self.assertEqual("AVAILABLE", starters[starter_id]["status"])
                self.assertEqual(["ORACLE"], starters[starter_id]["supported_db_types"])
                self.assertNotIn("required_entitlements", starters[starter_id])
        for db_type in ("MYSQL", "POSTGRESQL"):
            self.assertTrue(set(REPORT_PARAMETERS).isdisjoint(
                self._starters(_target(db_type=db_type)),
            ))
            for starter_id, parameters in REPORT_PARAMETERS.items():
                with self.subTest(starter_id=starter_id, db_type=db_type):
                    with self.assertRaisesRegex(AIOpsApplicationError, "数据库类型"):
                        self.catalog.freeze(
                            selection=ConversationStarterSelection(
                                starter_id=starter_id,
                                catalog_version=self.catalog.version,
                                parameters=parameters,
                            ),
                            target=_target(db_type=db_type),
                            capabilities={"ACTIVE_DIAGNOSIS"},
                        )

    def test_mixed_targets_do_not_hide_eligible_oracle_report(self):
        self.assertEqual(
            "AVAILABLE",
            self._starters(_target(), _target())["oracle.report.awr"]["status"],
        )
        self.assertEqual(
            "AVAILABLE",
            self._starters(_target())["oracle.report.awr"]["status"],
        )

    async def test_report_planning_requires_no_commercial_license_declaration(self):
        for starter_id in REPORT_PARAMETERS:
            with self.subTest(starter_id=starter_id):
                context = self._context(starter_id, _target())
                self.assertNotIn("required_entitlements", context.conversation_starter)
                self.assertNotIn("entitlements", context.capabilities.model_dump())
                investigation, tools, _, _ = await self._plan(context)
                self.assertTrue(investigation.plan.actions)
                self.assertTrue(tools)

    async def test_missing_tool_is_business_error_not_internal_failure(self):
        for starter_id in REPORT_PARAMETERS:
            with self.subTest(starter_id=starter_id):
                context = self._context(starter_id, _target(version_code=None))
                with self.assertRaises(AIOpsApplicationError) as raised:
                    await self._plan(context)
                self.assertEqual("AIOPS_STARTER_TOOLS_UNAVAILABLE", raised.exception.code)
                self.assertFalse(raised.exception.retryable)
        self.service._record_planning_route.assert_not_awaited()

    async def test_reports_pass_real_catalog_parameter_validation_without_license_flags(self):
        for starter_id in REPORT_PARAMETERS:
            with self.subTest(starter_id=starter_id):
                context = self._context(starter_id, _target())
                investigation, tools, _, _ = await self._plan(context)
                selected_ids = {tool["tool_id"] for tool in tools}
                self.assertNotIn("required_entitlements", context.conversation_starter)
                investigation = self.service._bind_target_to_plan(
                    investigation=self.service._apply_default_temporal_windows(
                        reset_model_deferred_flags(investigation),
                    ),
                    target_context=context.target_context,
                    available_tools=tools,
                )
                investigation = rewrite_incomplete_discovery_actions(
                    investigation=investigation, available_tools=tools,
                ) or investigation
                prepared, *_ = self.service._prepare_query_inputs(
                    investigation=investigation, context=context,
                )
                reports = [a for a in prepared.plan.actions if a.tool_id.endswith(".report") or a.tool_id.endswith(".diff_report")]
                self.assertTrue(reports)
                self.assertEqual(
                    starter_id != "oracle.report.ash",
                    all(action.deferred for action in reports),
                )
                self.assertEqual(1, len(prepared.task_frame.completion_requirements))
                if starter_id == "oracle.report.ash":
                    self.assertEqual(
                        {"begin_time": "2026-11-05T01:00:00.000000+00:00",
                         "end_time": "2026-11-05T02:00:00.000000+00:00"},
                        reports[0].input,
                    )

    async def test_execute_does_not_wrap_admission_error_or_retry_it(self):
        context = self._context("oracle.report.awr", _target(version_code=None))
        with self.assertRaises(AIOpsApplicationError) as raised:
            await self._plan(context)
        error = raised.exception
        self.service._execute_once = AsyncMock(side_effect=error)
        with self.assertRaises(AIOpsApplicationError) as returned:
            await self.service.execute({})
        self.assertIs(error, returned.exception)
        self.assertFalse(returned.exception.retryable)

    async def test_ui_reports_bind_compile_execute_and_download_html(self):
        """数据库边界使用模拟驱动，其余使用真实目录、绑定、Grant 与下载路径。"""
        playbooks = PlaybookRegistry.load()
        plan = DbaPlaybookPlan(catalog_hash=playbooks.catalog_hash, items=())
        snapshots = {
            "tool_outcomes": [{
                "tool_id": "db.oracle.awr.snapshots", "status": "SUCCEEDED",
                "observation": {
                    "columns": [{"name": name} for name in (
                        "snapshot_id", "instance_number", "begin_time", "end_time", "startup_time",
                    )],
                    "rows": [
                        [100 + hour, 1, f"2026-11-05T{hour:02}:00:00+00:00",
                         f"2026-11-05T{hour + 1:02}:00:00+00:00", "2026-11-01T00:00:00+00:00"]
                        for hour in range(4)
                    ],
                },
            }],
        }
        for starter_id, parameters in REPORT_PARAMETERS.items():
            with self.subTest(starter_id=starter_id):
                target = _target()
                context = self._context(starter_id, target)
                context.conversation_starter = self.catalog.freeze(
                    selection=ConversationStarterSelection(
                        starter_id=starter_id, catalog_version=self.catalog.version,
                        parameters=parameters,
                    ), target=target, capabilities={"ACTIVE_DIAGNOSIS"},
                )
                investigation, tools, _, _ = await self._plan(context)
                investigation = self.service._bind_target_to_plan(
                    investigation=self.service._apply_default_temporal_windows(
                        reset_model_deferred_flags(investigation),
                    ), target_context=context.target_context, available_tools=tools,
                )
                investigation = rewrite_incomplete_discovery_actions(
                    investigation=investigation, available_tools=tools,
                ) or investigation
                prepared, *_ = self.service._prepare_query_inputs(
                    investigation=investigation, context=context,
                )
                bound = bind_discovery_parameters(
                    plan=prepared.plan, tool_results=(snapshots,),
                    available_tools=tools, question=context.question,
                )
                self.assertFalse(bound.unbound_action_ids)
                self.assertFalse(any(action.deferred for action in bound.plan.actions))
                compiled = InvestigationTaskCompiler(playbooks).compile(
                    plan, investigation_actions=bound.plan.actions,
                )
                snapshot = self.builder.build(
                    plan=plan, compiled=compiled, capabilities=context.capabilities,
                    database_execution={}, direct_actions=bound.plan.actions,
                )
                report_tools = [
                    invocation["tool"] for invocation in snapshot["direct_invocations"].values()
                    if invocation["tool"]["tool_id"].endswith((".report", ".diff_report"))
                ]
                self.assertEqual(3 if starter_id == "oracle.report.awr-diff" else 1, len(report_tools))
                if starter_id == "oracle.report.awr":
                    self.assertEqual({"begin_snapshot_id": 100, "end_snapshot_id": 101}, report_tools[0]["parameters"])
                if starter_id == "oracle.report.awr-diff":
                    diff = next(tool for tool in report_tools if tool["tool_id"].endswith(".diff_report"))
                    self.assertEqual({
                        "baseline_begin_snapshot_id": 100, "baseline_end_snapshot_id": 101,
                        "after_begin_snapshot_id": 102, "after_end_snapshot_id": 103,
                    }, diff["parameters"])
                for tool in report_tools:
                    await self._execute_and_download_report(tool, snapshot, context)

    async def _execute_and_download_report(self, tool, snapshot, context):
        codec = DiagnosticGrantCodec(secret="g" * 32, issuer="test-worker", audience="test-executor")
        now = datetime.now(UTC)
        limits = DiagnosticLimits.model_validate(tool["limits"])
        grant = DiagnosticExecutionGrant(
            issuer="test-worker", audience="test-executor", grant_id=uuid7(),
            issued_at=now, expires_at=now + timedelta(minutes=5),
            run_id=uuid7(), task_id=uuid7(), lease_token_hash="a" * 64,
            target_id=context.capabilities.target_id, domain_id=1,
            target_row_version=1, db_type="ORACLE",
            connection_profile=DiagnosticConnectionProfile(host="db.internal", port=1521, service="PDB1"),
            diagnostic_credential_id=uuid7(), tool_id=tool["tool_id"], tool_version=tool["tool_version"],
            variant=tool["variant"], template_sha256=tool["template_sha256"],
            parameters_sha256=canonical_sha256(tool["parameters"]),
            capability_snapshot_hash=snapshot["capability_snapshot_hash"],
            limits=limits, trace_id="report-regression",
        )
        control = AsyncMock()
        control.issue_credential.return_value = SimpleNamespace(username="readonly", password="test-only")
        bind_resource_control(control, codec)
        body = f"<html><body>{tool['tool_id']} 报告</body></html>"
        driver = SimpleNamespace(db_type="ORACLE", execute=AsyncMock(return_value=DriverQueryResult(
            columns=("output",), rows=(("<html>",), (body[6:],)),
            truncated=False, db_version="19.24.0.0.0",
        )))
        executor = DiagnosticExecutorService(
            registry=self.diagnostics, grant_codec=codec, control_plane=control,
            drivers=(driver,), hard_limits=limits, concurrency=1,
        )
        result = await executor.execute(ReadDiagnosticRequest(
            executor_request_id=uuid7(), grant=codec.issue(grant),
            parameters=tool["parameters"], idempotency_key=str(uuid7()),
        ))
        self.assertEqual("SUCCEEDED", result.status)
        self.assertFalse(result.observation.truncated)
        self.assertEqual(tool["parameters"], driver.execute.await_args.kwargs["parameters"])
        payload = DbaToolResult(
            source_type="TOOL", source_id=tool["tool_id"], source_version=tool["tool_version"],
            definition_hash=tool["template_sha256"], output_schema="DBA_TOOL_RESULT.v1",
            measurement_semantics="HISTORICAL_SAMPLES", presentation_kind="TABLE", status="SUCCEEDED",
            tool_outcomes=(ToolOutcome(
                step_id="report", tool_id=tool["tool_id"], tool_version=tool["tool_version"],
                status="SUCCEEDED", observation=result.observation,
            ),),
        )
        conversation_id, actor_id, run_id, artifact_id = uuid7(), uuid7(), uuid7(), uuid7()
        uow = SimpleNamespace(
            conversations=SimpleNamespace(get_conversation=AsyncMock(return_value=SimpleNamespace(created_by=actor_id))),
            turns=SimpleNamespace(
                get_turn=AsyncMock(return_value=SimpleNamespace(conversation_id=conversation_id)),
                get_run_link=AsyncMock(return_value=SimpleNamespace(ops_run_id=run_id)),
            ),
            runs=SimpleNamespace(get_artifact=AsyncMock(return_value=SimpleNamespace(
                artifact_id=artifact_id, ops_run_id=run_id, artifact_type="DBA_TOOL_RESULT",
                schema_version="DBA_TOOL_RESULT.v1", payload_json=payload.model_dump(mode="json"),
                payload_uri=None, provenance_json={}, content_hash="a" * 64, byte_size=len(body.encode()),
            ))),
        )
        manager = AsyncMock()
        manager.__aenter__.return_value = uow
        content, media_type, file_name = await ConversationTurnService(uow_factory=lambda: manager).get_artifact_content(
            domain_id=1, conversation_id=conversation_id, turn_id=uuid7(),
            actor_id=actor_id, artifact_id=artifact_id,
        )
        self.assertEqual(body.encode(), content)
        self.assertEqual("text/html", media_type)
        self.assertTrue(file_name.endswith(".html"))

    def test_ash_input_normalization_matches_sql_and_is_idempotent(self):
        tool = next(item for item in self.diagnostics.tools if item.definition.tool_id == "db.oracle.ash.report")
        self.assertIn('YYYY-MM-DD"T"HH24:MI:SS.FFTZH:TZM', tool.sql)
        for begin, end, expected_begin, expected_end in (
            ("2026-11-05T01:00:00.000Z", "2026-11-05T02:00:00.000Z",
             "2026-11-05T01:00:00.000000+00:00", "2026-11-05T02:00:00.000000+00:00"),
            ("2026-11-05T09:00:00+08:00", "2026-11-05T10:00:00+08:00",
             "2026-11-05T01:00:00.000000+00:00", "2026-11-05T02:00:00.000000+00:00"),
            ("2026-11-05T09:00:00.123456+08:00", "2026-11-05T09:00:00.123457+08:00",
             "2026-11-05T01:00:00.123456+00:00", "2026-11-05T01:00:00.123457+00:00"),
        ):
            with self.subTest(begin=begin):
                normalized = self.diagnostics.validate_parameters(tool, {"begin_time": begin, "end_time": end})
                self.assertEqual({"begin_time": expected_begin, "end_time": expected_end}, normalized)
                self.assertEqual(normalized, self.diagnostics.validate_parameters(tool, normalized))

    def test_awr_snapshots_extract_utc_from_timezone_columns(self):
        tool = next(item for item in self.diagnostics.tools if item.definition.tool_id == "db.oracle.awr.snapshots")
        self.assertIn("sys_extract_utc(begin_interval_time_tz)", tool.sql.lower())
        self.assertIn("sys_extract_utc(end_interval_time_tz)", tool.sql.lower())

    def test_unexpected_candidate_error_keeps_safe_planning_stage(self):
        try:
            select_planning_candidates(
                tools=(), playbooks=(),
                tool_ids=("db.oracle.awr.report",), playbook_ids=(),
            )
        except ValueError as cause:
            error = TurnPlanningStageError(cause)
        self.assertEqual("SELECT_PLANNING_CANDIDATES", error.stage)
        self.assertEqual("ValueError", error.cause_type)
        self.assertNotIn("未注册工具", str(error))


if __name__ == "__main__":
    unittest.main()
