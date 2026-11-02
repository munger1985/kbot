"""固定巡检的路由、授权与单目标执行范围回归。"""

from __future__ import annotations

import unittest
from copy import deepcopy
from unittest.mock import AsyncMock

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.inspections.routing import is_scheduled_inspection
from aiops_agent.application.turn_planner import TurnPlannerService
from platform_core.identity import uuid7


class ScheduledInspectionRoutingTest(unittest.IsolatedAsyncioTestCase):
    def _context(self) -> dict:
        targets = [str(uuid7()) for _ in range(3)]
        business_system_id = str(uuid7())
        resources = [
            {
                "system_resource_id": str(uuid7()),
                "resource_id": str(uuid7()),
                "resource_code": f"DB-{index}",
                "display_name": f"数据库 {index}",
                "resource_type_key": "DATABASE",
                "location_name": None,
                "attributes": {},
                "database": {"db_type": db_type},
                "topology_roles": [],
                "target_id": target_id,
                "target_ready": True,
            }
            for index, (target_id, db_type) in enumerate(zip(
                targets, ("ORACLE", "POSTGRESQL", "MYSQL"), strict=True,
            ))
        ]
        return {
            "turn_id": str(uuid7()), "plan_revision": 1,
            "policy_snapshot_hash": "a" * 64,
            "capabilities": ["ACTIVE_DIAGNOSIS", "INSPECTION_QUERY"],
            "focused_target_id": targets[0],
            "question": "查看历史巡检结果，不要执行数据库 0，请查看数据库 2",
            "context_hint": {},
            "systems": [{
                "business_system_id": business_system_id,
                "system_code": "PROD", "display_name": "生产系统",
                "environment": "PROD", "resources": resources, "relations": [],
            }],
            "execution_context": {
                "trigger_type": "SCHEDULE", "workflow_kind": "INSPECTION",
                "inspection_fire_id": str(uuid7()),
                "observation_start": "2026-11-01T16:00:00+00:00",
                "observation_end": "2026-11-02T02:16:00+00:00",
                "inspection": {
                    "business_system_id": business_system_id,
                    "template_id": str(uuid7()), "template_hash": "b" * 64,
                    "evidence_steps": [{"tool_id": "db.instance.identity"}],
                },
            },
        }

    async def test_every_database_bypasses_model_and_pins_its_frozen_target(self) -> None:
        context = self._context()
        model = AsyncMock()
        model.generate_structured.side_effect = AssertionError("固定巡检不调用意图模型")
        planner = TurnPlannerService(uow_factory=AsyncMock(), structured_model=model)
        for resource in context["systems"][0]["resources"]:
            with self.subTest(db_type=resource["database"]["db_type"]):
                context["focused_target_id"] = resource["target_id"]
                before = deepcopy(context)
                plan, receipt = await planner._plan(context)
                scope, internal, clarification = planner._resolve_scope(context, plan)

                self.assertTrue(plan.requires_active_diagnosis)
                self.assertEqual(["ACTIVE_DIAGNOSIS"], list(plan.requested_capabilities))
                self.assertEqual("SCHEDULED_INSPECTION", receipt.model_dump()["route"])
                self.assertEqual(resource["target_id"], receipt.target_id)
                self.assertEqual("2026-11-02T02:16:00+00:00", plan.time_range.end_at.isoformat())
                self.assertIsNone(clarification)
                self.assertEqual(1, len(scope.resources))
                self.assertEqual(
                    [resource["target_id"]],
                    [item["target_id"] for item in internal["execution_targets"]],
                )
                self.assertEqual(before, context)
        model.generate_structured.assert_not_awaited()

    async def test_scheduled_route_does_not_depend_on_model_configuration(self) -> None:
        planner = TurnPlannerService(uow_factory=AsyncMock())
        plan, _ = await planner._plan(self._context())
        self.assertTrue(plan.requires_active_diagnosis)

    async def test_active_diagnosis_authorization_is_still_required(self) -> None:
        context = self._context()
        context["capabilities"] = ["INSPECTION_QUERY"]
        with self.assertRaisesRegex(AIOpsApplicationError, "未授权"):
            await TurnPlannerService(uow_factory=AsyncMock())._plan(context)

    async def test_incomplete_execution_contract_never_falls_back_to_chat(self) -> None:
        for missing in ("focused_target_id", "inspection_fire_id", "observation_start", "inspection"):
            context = self._context()
            if missing == "focused_target_id":
                context.pop(missing)
            else:
                context["execution_context"].pop(missing)
            with self.subTest(missing=missing):
                with self.assertRaises(AIOpsApplicationError):
                    await TurnPlannerService(uow_factory=AsyncMock())._plan(context)

    async def test_revoked_target_never_expands_to_other_authorized_databases(self) -> None:
        context = self._context()
        planner = TurnPlannerService(uow_factory=AsyncMock())
        plan, _ = await planner._plan(context)
        context["systems"][0]["resources"].pop(0)
        with self.assertRaisesRegex(AIOpsApplicationError, "授权资源范围"):
            planner._resolve_scope(context, plan)

    async def test_unavailable_target_is_not_completed_as_a_read_only_query(self) -> None:
        context = self._context()
        planner = TurnPlannerService(uow_factory=AsyncMock())
        plan, _ = await planner._plan(context)
        context["systems"][0]["resources"][0]["target_ready"] = False
        with self.assertRaisesRegex(AIOpsApplicationError, "未启用只读取证"):
            planner._resolve_scope(context, plan)

    async def test_shared_target_in_two_systems_still_has_one_execution_target(self) -> None:
        context = self._context()
        context["systems"].insert(0, {
            **context["systems"][0],
            "business_system_id": str(uuid7()),
        })
        planner = TurnPlannerService(uow_factory=AsyncMock())
        plan, _ = await planner._plan(context)
        scope, internal, _ = planner._resolve_scope(context, plan)
        self.assertEqual(1, len(internal["execution_targets"]))
        self.assertEqual(
            context["execution_context"]["inspection"]["business_system_id"],
            str(scope.business_systems[0].business_system_id),
        )

    def test_words_and_follow_up_conversations_do_not_activate_scheduled_route(self) -> None:
        for execution in (
            {}, {"trigger_type": "CHAT", "workflow_kind": "INSPECTION"},
            {"trigger_type": "SCHEDULE", "workflow_kind": "CHAT_TURN"},
        ):
            with self.subTest(execution=execution):
                self.assertFalse(is_scheduled_inspection({
                    "question": "执行日常巡检", "execution_context": execution,
                }))


if __name__ == "__main__":
    unittest.main()
