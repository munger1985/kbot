"""冻结巡检工具输入的隔离、参数边界与默认模板可执行性。"""

from __future__ import annotations

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.application.inspections.check_catalog import (
    compile_selected_check_steps, default_selected_check_ids,
)
from aiops_agent.application.investigation import (
    InvestigationPlanValidationError, TurnPlanningService,
)
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.playbooks import PlaybookRegistry
from aiops_agent.tools import ToolExecutionSnapshotBuilder
from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot
from platform_core.identity import uuid7


class FixedInspectionInputTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        diagnostics = DiagnosticRegistry.load()
        playbooks = PlaybookRegistry.load(allowed_tools=frozenset(
            (tool.definition.tool_id, tool.definition.version)
            for tool in diagnostics.tools
        ))
        self.builder = ToolExecutionSnapshotBuilder(
            playbook_registry=playbooks, diagnostic_registry=diagnostics,
        )
        self.service = object.__new__(TurnPlanningService)
        self.service._tool_snapshot_builder = self.builder
        self.service._record_planning_route = AsyncMock()

    def context(self, steps: tuple[dict, ...], database_type: str = "ORACLE"):
        return SimpleNamespace(
            inspection={"template_id": "daily-test", "template_version": "1.0.0",
                        "schedule_type": "DAILY", "evidence_steps": steps},
            target_context={"db_type": database_type, "display_name": "测试数据库"},
            question="执行固定巡检",
            capabilities=DbaCapabilitySnapshot(
                agent_id=str(uuid7()), agent_version_id=str(uuid7()), target_id=str(uuid7()),
                database_type=database_type,
                database_version={"ORACLE": "19c", "MYSQL": "8.0", "POSTGRESQL": "16"}[database_type],
                target_enabled=True, target_reachable=True,
                target_capabilities=("DB_READONLY",),
            ),
        )

    async def test_bad_object_step_does_not_abort_other_checks_or_guess_schema(self):
        steps = (
            {"tool_id": "db.instance.performance", "title": "实例性能"},
            {"tool_id": "db.index.health", "title": "指定索引", "input": {}},
        )
        before = deepcopy(steps)
        context = self.context(steps)
        plan, _, route = await self.service._plan_template_inspection(
            context=context, fixed_tools=self.builder.discover_tools(context.capabilities),
        )
        self.assertEqual(["db.instance.performance"], [action.tool_id for action in plan.plan.actions])
        self.assertEqual(steps, before)
        gap = route["evidence_gaps"][0]
        self.assertEqual("db.index.health", gap["step_id"])
        self.assertEqual("INSPECTION_FIXED_TOOL_INPUT_INVALID", gap["code"])
        self.assertIn("schema_name", gap["detail"])
        self.assertIn("不能猜测", gap["detail"])

    async def test_explicit_object_parameters_remain_frozen(self):
        context = self.context(({
            "tool_id": "db.index.health", "title": "指定索引",
            "input": {"schema_name": "APP", "index_name": "ORDERS_PK"},
        },))
        plan, _, route = await self.service._plan_template_inspection(
            context=context, fixed_tools=self.builder.discover_tools(context.capabilities),
        )
        self.assertEqual({"schema_name": "APP", "index_name": "ORDERS_PK"}, plan.plan.actions[0].input)
        self.assertEqual([], route["evidence_gaps"])

    async def test_all_invalid_steps_fail_with_actionable_template_reason(self):
        context = self.context(({"tool_id": "db.index.health"},))
        with self.assertRaisesRegex(InvestigationPlanValidationError, "schema_name"):
            await self.service._plan_template_inspection(
                context=context, fixed_tools=self.builder.discover_tools(context.capabilities),
            )

    async def test_current_defaults_validate_for_every_database(self):
        for database_type in ("ORACLE", "MYSQL", "POSTGRESQL"):
            with self.subTest(database_type=database_type):
                steps = compile_selected_check_steps(
                    default_selected_check_ids("DAILY"), database_type=database_type,
                )
                context = self.context(steps, database_type)
                plan, _, route = await self.service._plan_template_inspection(
                    context=context, fixed_tools=self.builder.discover_tools(context.capabilities),
                )
                self.assertEqual([], route["evidence_gaps"])
                self.assertEqual(len(steps), len(plan.plan.actions))


if __name__ == "__main__":
    unittest.main()
