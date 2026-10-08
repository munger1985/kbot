"""新会话功能入口目录与确定性计划测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.application.conversation_starters import ConversationStarterCatalog
from aiops_agent.application.investigation.service import TurnPlanningService
from platform_core.contracts.aiops import ConversationStarterSelection


def _target(**overrides):
    values = {
        "target_id": "target-1",
        "db_type": "ORACLE",
        "status": "ENABLED",
        "readonly_connection_enabled": True,
        "connectivity_status": "CONNECTED",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ConversationStarterCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = ConversationStarterCatalog()

    def test_catalog_filters_database_type_and_hides_internal_planning(self) -> None:
        payload = self.catalog.list_for_target(_target(db_type="MYSQL"))
        ids = {item["starter_id"] for item in payload["starters"]}
        self.assertIn("database.health.overview", ids)
        self.assertIn("database.replication.status", ids)
        self.assertNotIn("oracle.report.awr", ids)
        self.assertTrue(all("planning" not in item for item in payload["starters"]))

    def test_unavailable_target_disables_every_starter(self) -> None:
        payload = self.catalog.list_for_target(
            _target(readonly_connection_enabled=False)
        )
        self.assertTrue(payload["starters"])
        self.assertEqual(
            {"UNAVAILABLE"},
            {item["status"] for item in payload["starters"]},
        )

    def test_unknown_connectivity_is_limited_but_still_selectable(self) -> None:
        payload = self.catalog.list_for_target(
            _target(connectivity_status="UNKNOWN")
        )
        self.assertEqual(
            {"LIMITED"},
            {item["status"] for item in payload["starters"]},
        )

    def test_awr_diff_requires_equal_time_windows(self) -> None:
        selection = ConversationStarterSelection(
            starter_id="oracle.report.awr-diff",
            catalog_version="1.2.0",
            parameters={
                "first_begin_time": "2026-09-28T08:00:00+00:00",
                "first_end_time": "2026-09-28T09:00:00+00:00",
                "second_begin_time": "2026-09-28T10:00:00+00:00",
                "second_end_time": "2026-09-28T12:00:00+00:00",
                "timezone": "Asia/Shanghai",
            },
        )
        with self.assertRaisesRegex(Exception, "必须等长"):
            self.catalog.freeze(selection=selection, target=_target())

    def test_freeze_uses_server_catalog_planning_and_message(self) -> None:
        frozen = self.catalog.freeze(
            selection=ConversationStarterSelection(
                starter_id="oracle.runbook.adg-build",
                catalog_version="1.2.0",
            ),
            target=_target(),
        )
        self.assertEqual("IMPLEMENTATION", frozen["planning"]["kind"])
        self.assertEqual(
            "ORACLE_ADG_BUILD",
            frozen["planning"]["implementation_profile"],
        )
        self.assertEqual("执行功能：ADG 部署文档", frozen["user_message"])

    def test_runbook_starters_publish_optional_parameter_schema(self) -> None:
        payload = self.catalog.list_for_target(_target())
        datapump = next(
            item
            for item in payload["starters"]
            if item["starter_id"] == "oracle.runbook.datapump"
        )
        fields = {item["name"]: item for item in datapump["input_schema"]}
        self.assertIn("DATAPUMP_DIRECTORY_PATH", fields)
        self.assertIn("SOURCE_SCHEMAS", fields)
        self.assertTrue(all(not item["required"] for item in fields.values()))

    def test_datapump_parameters_are_normalized_and_empty_values_ignored(self) -> None:
        frozen = self.catalog.freeze(
            selection=ConversationStarterSelection(
                starter_id="oracle.runbook.datapump",
                catalog_version="1.2.0",
                parameters={
                    "DATAPUMP_DIRECTORY_PATH": "",
                    "SOURCE_SCHEMAS": "app_core, APP_REPORT\napp_core",
                    "DATAPUMP_PARALLEL": "8",
                },
            ),
            target=_target(),
        )
        self.assertEqual(
            {"SOURCE_SCHEMAS": "APP_CORE,APP_REPORT", "DATAPUMP_PARALLEL": 8},
            frozen["parameters"],
        )

    def test_rman_recovery_time_is_normalized_for_oracle(self) -> None:
        payload = self.catalog.list_for_target(_target())
        recovery = next(
            item
            for item in payload["starters"]
            if item["starter_id"] == "oracle.runbook.rman-recovery"
        )
        fields = {item["name"]: item for item in recovery["input_schema"]}
        self.assertEqual(
            "oracle_datetime",
            fields["RECOVERY_TARGET_TIME"]["type"],
        )

        frozen = self.catalog.freeze(
            selection=ConversationStarterSelection(
                starter_id="oracle.runbook.rman-recovery",
                catalog_version="1.2.0",
                parameters={
                    "RECOVERY_SCENARIO": "DATABASE_PITR",
                    "RECOVERY_TARGET_TIME": "2026-9-29 17:00:00",
                },
            ),
            target=_target(),
        )
        self.assertEqual(
            "2026-09-29 17:00:00",
            frozen["parameters"]["RECOVERY_TARGET_TIME"],
        )

    def test_rman_recovery_time_rejects_invalid_calendar_date(self) -> None:
        with self.assertRaisesRegex(Exception, "YYYY-MM-DD HH24:MI:SS"):
            self.catalog.freeze(
                selection=ConversationStarterSelection(
                    starter_id="oracle.runbook.rman-recovery",
                    catalog_version="1.2.0",
                    parameters={
                        "RECOVERY_TARGET_TIME": "2026-02-30 17:00:00",
                    },
                ),
                target=_target(),
            )

    def test_runbook_parameters_reject_unknown_and_unsafe_values(self) -> None:
        with self.assertRaisesRegex(Exception, "未知字段"):
            self.catalog.freeze(
                selection=ConversationStarterSelection(
                    starter_id="oracle.runbook.datapump",
                    catalog_version="1.2.0",
                    parameters={"PASSWORD": "secret"},
                ),
                target=_target(),
            )
        with self.assertRaisesRegex(Exception, "安全的绝对路径"):
            self.catalog.freeze(
                selection=ConversationStarterSelection(
                    starter_id="oracle.runbook.datapump",
                    catalog_version="1.2.0",
                    parameters={"DATAPUMP_DIRECTORY_PATH": "../../tmp"},
                ),
                target=_target(),
            )

    def test_rac_parameter_validation_rejects_duplicate_nodes(self) -> None:
        with self.assertRaisesRegex(Exception, "不能相同"):
            self.catalog.freeze(
                selection=ConversationStarterSelection(
                    starter_id="oracle.runbook.rac-build",
                    catalog_version="1.2.0",
                    parameters={
                        "NODE1_HOST": "rac01.example.com",
                        "NODE2_HOST": "rac01.example.com",
                    },
                ),
                target=_target(),
            )

    def test_awr_diff_compiler_builds_snapshots_reports_and_diff(self) -> None:
        context = SimpleNamespace(
            question="执行 AWR 对比",
            target_context={"db_type": "ORACLE", "display_name": "TestDB"},
            conversation_starter={"starter_id": "oracle.report.awr-diff"},
        )
        output, tool_ids = TurnPlanningService._starter_diagnostic_output(
            object.__new__(TurnPlanningService),
            context=context,
            kind="AWR_DIFF",
            planning={},
            parameters={
                "first_begin_time": "2026-09-28T08:00:00+00:00",
                "first_end_time": "2026-09-28T09:00:00+00:00",
                "second_begin_time": "2026-09-28T10:00:00+00:00",
                "second_end_time": "2026-09-28T11:00:00+00:00",
            },
            title="生成 AWR 对比报告",
        )
        self.assertEqual(
            (
                "db.instance.identity",
                "db.oracle.awr.snapshots",
                "db.oracle.awr.report",
                "db.oracle.awr.report",
                "db.oracle.awr.diff_report",
            ),
            tuple(action.tool_id for action in output.plan.actions),
        )
        self.assertIn("db.oracle.awr.diff_report", tool_ids)
        self.assertEqual(1, len(output.task_frame.completion_requirements))


class ConversationStarterPlanningTest(unittest.IsolatedAsyncioTestCase):
    def _diagnostic_output(
        self,
        *,
        starter_id: str,
        kind: str,
        visualization_profile_id: str,
        parameters: dict | None = None,
    ):
        service = object.__new__(TurnPlanningService)
        context = SimpleNamespace(
            question="执行功能",
            target_context={"db_type": "ORACLE", "display_name": "TestDB"},
            conversation_starter={"starter_id": starter_id},
        )
        output, _ = service._starter_diagnostic_output(
            context=context,
            kind=kind,
            planning={
                "visualization_profile_id": visualization_profile_id,
                "tool_ids": ["db.instance.identity"],
            },
            parameters=parameters or {},
            title="测试功能",
        )
        return output

    async def test_health_starter_adds_one_hour_monitoring_profile(self) -> None:
        output = self._diagnostic_output(
            starter_id="database.health.overview",
            kind="TOOLS",
            visualization_profile_id="health.overview",
        )

        self.assertEqual(
            "health.overview", output.task_frame.visualization_profile_id
        )
        self.assertEqual(3600, output.task_frame.visualization_window_seconds)
        self.assertEqual(
            "COMBINED", output.task_frame.evidence_source_strategy
        )
        self.assertTrue(
            TurnPlanningService._requires_monitoring_snapshot(
                investigation=output,
                inspection=False,
                alert_diagnosis=False,
            )
        )

    async def test_performance_starter_adds_fifteen_minute_profile(
        self,
    ) -> None:
        output = self._diagnostic_output(
            starter_id="database.performance.current",
            kind="CURRENT_PERFORMANCE",
            visualization_profile_id="performance.current",
        )

        self.assertEqual(
            "performance.current", output.task_frame.visualization_profile_id
        )
        self.assertEqual(900, output.task_frame.visualization_window_seconds)
        self.assertEqual(
            "COMBINED", output.task_frame.evidence_source_strategy
        )

    async def test_storage_starter_uses_requested_days_for_chart_window(
        self,
    ) -> None:
        output = self._diagnostic_output(
            starter_id="database.storage.trend",
            kind="STORAGE_TREND",
            visualization_profile_id="storage.trend",
            parameters={"days": 7},
        )

        self.assertEqual(
            "storage.trend", output.task_frame.visualization_profile_id
        )
        self.assertEqual(604_800, output.task_frame.requested_window_seconds)
        self.assertEqual(
            604_800, output.task_frame.visualization_window_seconds
        )
        self.assertEqual(
            "MONITORING_FIRST",
            output.task_frame.evidence_source_strategy,
        )

    async def test_adg_starter_maps_directly_to_implementation_profile(self) -> None:
        service = object.__new__(TurnPlanningService)
        service._record_planning_route = AsyncMock()
        context = SimpleNamespace(
            question="执行功能：ADG 部署文档",
            target_context={"db_type": "ORACLE", "display_name": "TestDB"},
            conversation_starter={
                "starter_id": "oracle.runbook.adg-build",
                "catalog_version": "1.2.0",
                "title": "ADG 部署文档",
                "parameters": {"STANDBY_HOST": "testdb-dr"},
                "planning": {
                    "kind": "IMPLEMENTATION",
                    "implementation_profile": "ORACLE_ADG_BUILD",
                },
            },
        )
        investigation, tools, playbooks, route = (
            await service._plan_conversation_starter(
                context=context,
                available_tools=(
                    {"tool_id": "db.instance.identity"},
                    {"tool_id": "db.ha.adg_precheck"},
                ),
                available_playbooks=(
                    {"playbook_id": "oracle.ha.adg_build"},
                ),
            )
        )
        self.assertEqual(
            "ORACLE_ADG_BUILD",
            investigation.task_frame.implementation_profile,
        )
        self.assertEqual(
            {"STANDBY_HOST": "testdb-dr"},
            investigation.task_frame.subject_ref["implementation_parameters"],
        )
        self.assertEqual(
            {"STANDBY_HOST": "testdb-dr"},
            investigation.task_frame.subject_ref["implementation_generation"]["supplied_parameters"],
        )
        self.assertEqual(
            ("db.instance.identity", "db.ha.adg_precheck"),
            tuple(item["tool_id"] for item in tools),
        )
        self.assertEqual(
            ("oracle.ha.adg_build",),
            tuple(item["playbook_id"] for item in playbooks),
        )
        self.assertEqual("CONVERSATION_STARTER", route["mode"])
        service._record_planning_route.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
