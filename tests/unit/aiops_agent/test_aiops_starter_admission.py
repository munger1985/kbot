"""报告功能入口、许可准入和固定规划的跨项目回归。"""

from __future__ import annotations

import unittest
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
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.playbooks import PlaybookRegistry
from aiops_agent.tools import ToolExecutionSnapshotBuilder
from aiops_agent.tools.capabilities import build_capability_snapshot
from platform_core.identity import uuid7


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


def _target(*, licensed: bool = False, **overrides):
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
            "entitlements": ["ORACLE_DIAGNOSTICS_PACK"] if licensed else [],
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

    def test_all_reports_require_explicit_entitlement_in_menu(self):
        for target in (
            _target(),
            _target(capabilities_json={"entitlements": "ORACLE_DIAGNOSTICS_PACK"}),
            _target(capabilities_json={"privileges": ["ORACLE_DIAGNOSTICS_PACK"]}),
        ):
            starters = self._starters(target)
            for starter_id in REPORT_PARAMETERS:
                with self.subTest(starter_id=starter_id, target=target.capabilities_json):
                    self.assertEqual("UNAVAILABLE", starters[starter_id]["status"])
                    self.assertIn("ORACLE_DIAGNOSTICS_PACK", starters[starter_id]["availability_reason"])
            self.assertEqual("AVAILABLE", starters["database.performance.current"]["status"])

    def test_entitled_reports_remain_oracle_only(self):
        starters = self._starters(_target(licensed=True))
        for starter_id in REPORT_PARAMETERS:
            with self.subTest(starter_id=starter_id):
                self.assertEqual("AVAILABLE", starters[starter_id]["status"])
                self.assertEqual(["ORACLE"], starters[starter_id]["supported_db_types"])
                self.assertNotIn("required_entitlements", starters[starter_id])
        for db_type in ("MYSQL", "POSTGRESQL"):
            self.assertTrue(set(REPORT_PARAMETERS).isdisjoint(
                self._starters(_target(db_type=db_type)),
            ))

    def test_mixed_targets_do_not_hide_eligible_oracle_report(self):
        self.assertEqual(
            "AVAILABLE",
            self._starters(_target(), _target(licensed=True))["oracle.report.awr"]["status"],
        )
        self.assertEqual(
            "UNAVAILABLE",
            self._starters(_target())["oracle.report.awr"]["status"],
        )

    async def test_natural_language_and_frozen_plans_recheck_entitlements(self):
        for starter_id in REPORT_PARAMETERS:
            with self.subTest(starter_id=starter_id):
                context = self._context(starter_id, _target())
                self.assertEqual(
                    ["ORACLE_DIAGNOSTICS_PACK"],
                    context.conversation_starter["required_entitlements"],
                )
                with self.assertRaises(AIOpsApplicationError) as raised:
                    await self._plan(context)
                self.assertEqual("AIOPS_STARTER_ENTITLEMENT_REQUIRED", raised.exception.code)
                self.assertFalse(raised.exception.retryable)
                self.assertIn("ORACLE_DIAGNOSTICS_PACK", raised.exception.message)
        self.service._record_planning_route.assert_not_awaited()

    async def test_missing_tool_is_business_error_not_internal_failure(self):
        for starter_id in REPORT_PARAMETERS:
            with self.subTest(starter_id=starter_id):
                context = self._context(starter_id, _target(licensed=True, version_code=None))
                with self.assertRaises(AIOpsApplicationError) as raised:
                    await self._plan(context)
                self.assertEqual("AIOPS_STARTER_TOOLS_UNAVAILABLE", raised.exception.code)
                self.assertFalse(raised.exception.retryable)
        self.service._record_planning_route.assert_not_awaited()

    async def test_entitled_reports_pass_real_catalog_parameter_validation(self):
        for starter_id in REPORT_PARAMETERS:
            with self.subTest(starter_id=starter_id):
                context = self._context(starter_id, _target(licensed=True))
                investigation, tools, _, _ = await self._plan(context)
                selected_ids = {tool["tool_id"] for tool in tools}
                required_entitlements = {
                    entitlement
                    for tool in self.diagnostics.tools
                    if tool.definition.db_type == "ORACLE"
                    and tool.definition.tool_id in selected_ids
                    for entitlement in tool.definition.required_entitlements
                }
                self.assertEqual(
                    required_entitlements,
                    set(context.conversation_starter["required_entitlements"]),
                )
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
                        {k: v for k, v in REPORT_PARAMETERS[starter_id].items() if k != "timezone"},
                        reports[0].input,
                    )

    async def test_execute_does_not_wrap_admission_error_or_retry_it(self):
        context = self._context("oracle.report.awr", _target())
        with self.assertRaises(AIOpsApplicationError) as raised:
            await self._plan(context)
        error = raised.exception
        self.service._execute_once = AsyncMock(side_effect=error)
        with self.assertRaises(AIOpsApplicationError) as returned:
            await self.service.execute({})
        self.assertIs(error, returned.exception)
        self.assertFalse(returned.exception.retryable)

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
