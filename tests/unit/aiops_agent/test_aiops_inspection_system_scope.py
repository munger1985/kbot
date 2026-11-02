"""系统归属巡检的范围、授权与动态清单回归。"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, create_autospec

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.inspections.scope import inspection_system_scope, require_inspection_agent_scope
from aiops_agent.repositories.agent import AIOpsAgentRepository
from platform_core.contracts.aiops import InspectionPlanCreate, InspectionPlanPatch
from platform_core.identity import uuid7
from pydantic import ValidationError
from tests.support.inspection_system import configure_inspection_system


class InspectionSystemScopeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.system_id, self.tenant_id = uuid7(), uuid7()
        self.domain_id = 100
        self.target = SimpleNamespace(target_id=uuid7(), display_name="ERP 主库", db_type="ORACLE")
        self.uow = SimpleNamespace(agents=create_autospec(AIOpsAgentRepository, instance=True, spec_set=True))
        self.rows = configure_inspection_system(
            self.uow, [self.target], domain_id=self.domain_id, business_system_id=self.system_id,
            tenant_id=self.tenant_id,
        )

    async def resolve(self):
        return await inspection_system_scope(
            uow=self.uow, domain_id=self.domain_id, business_system_id=self.system_id,
        )

    async def test_system_defines_scope_not_agent_other_system_targets(self):
        self.uow.agents.active_version_target_ids = AsyncMock(return_value=[self.target.target_id, uuid7()])
        snapshot, targets = await self.resolve()
        self.assertEqual([str(self.target.target_id)], snapshot["target_ids"])
        self.assertEqual((self.target,), targets)
        self.uow.agents.active_version_target_ids.assert_not_awaited()
        self.uow.business_systems.get_scoped.assert_awaited_once_with(
            business_system_id=self.system_id, domain_id=self.domain_id,
        )

    async def test_unreachable_target_is_not_silently_excluded(self):
        self.target.connectivity_status = "FAILED"
        _, targets = await self.resolve()
        self.assertEqual((self.target,), targets)

    async def test_missing_disabled_or_cross_domain_target_blocks_complete_scope(self):
        for mode in ("missing", "disabled", "cross_domain"):
            with self.subTest(mode=mode):
                membership, resource, database, _ = self.rows[0]
                target = SimpleNamespace(**vars(self.target))
                if mode == "disabled": target.status = "DISABLED"
                if mode == "cross_domain": target.domain_id = 999
                self.uow.business_systems.list_system_resource_rows.return_value = [
                    (membership, resource, database, None if mode == "missing" else target)
                ]
                with self.assertRaises(AIOpsApplicationError) as caught:
                    await self.resolve()
                self.assertEqual("AIOPS_INSPECTION_SYSTEM_RESOURCE_INVALID", caught.exception.code)
                self.assertIn("ERP 主库", caught.exception.message)

    async def test_retired_resources_are_explicitly_excluded_and_targets_deduplicated(self):
        _, resource, database, target = self.rows[0]
        self.rows.append((SimpleNamespace(status="ACTIVE"), resource, database, target))
        retired = SimpleNamespace(**vars(resource)); retired.resource_id = uuid7(); retired.lifecycle_status = "RETIRED"
        self.rows.append((SimpleNamespace(status="ACTIVE"), retired, None, None))
        snapshot, targets = await self.resolve()
        self.assertEqual(1, len(targets))
        self.assertEqual(str(retired.resource_id), snapshot["excluded_resources"][0]["resource_id"])

    async def test_new_database_joins_next_scope_without_mutating_previous_snapshot(self):
        first, _ = await self.resolve()
        resource = SimpleNamespace(**vars(self.rows[0][1])); resource.resource_id = uuid7()
        target = SimpleNamespace(**vars(self.target)); target.target_id = uuid7()
        self.rows.append((SimpleNamespace(status="ACTIVE"), resource, SimpleNamespace(target_id=target.target_id), target))
        second, targets = await self.resolve()
        self.assertEqual(1, len(first["target_ids"]))
        self.assertEqual(2, len(second["target_ids"]))
        self.assertEqual(2, len(targets))

    async def test_unavailable_system_and_empty_system_are_not_success(self):
        self.uow.business_systems.get_scoped.return_value = None
        with self.assertRaises(AIOpsApplicationError) as caught:
            await self.resolve()
        self.assertEqual("AIOPS_INSPECTION_SYSTEM_UNAVAILABLE", caught.exception.code)
        self.uow.business_systems.get_scoped.return_value = SimpleNamespace(status="ENABLED")
        self.rows.clear()
        with self.assertRaises(AIOpsApplicationError) as caught:
            await self.resolve()
        self.assertEqual("AIOPS_INSPECTION_SYSTEM_DATABASE_REQUIRED", caught.exception.code)

    async def test_revoked_system_authorization_and_missing_capability_fail_explicitly(self):
        self.uow.agents.version_has_business_system.return_value = False
        with self.assertRaises(AIOpsApplicationError) as caught:
            await require_inspection_agent_scope(uow=self.uow, agent_version_id=uuid7(), business_system_id=self.system_id)
        self.assertEqual("AIOPS_INSPECTION_SYSTEM_FORBIDDEN", caught.exception.code)
        self.uow.agents.version_has_business_system.return_value = True
        self.uow.agents.version_capabilities.return_value = ["MONITORING_QUERY"]
        with self.assertRaises(AIOpsApplicationError) as caught:
            await require_inspection_agent_scope(uow=self.uow, agent_version_id=uuid7(), business_system_id=self.system_id)
        self.assertEqual("AIOPS_INSPECTION_AGENT_CAPABILITY_REQUIRED", caught.exception.code)

    def test_plan_requires_system_and_patch_cannot_transfer_ownership(self):
        with self.assertRaises(ValidationError):
            InspectionPlanCreate(display_name="缺少系统", agent_id=uuid7(), schedule_type="DAILY",
                                 cron_expression="0 2 * * *", timezone="Asia/Shanghai",
                                 inspection_template_id=uuid7(), timeout_seconds=1800)
        with self.assertRaises(ValidationError):
            InspectionPlanPatch(business_system_id=self.system_id)


if __name__ == "__main__":
    unittest.main()
