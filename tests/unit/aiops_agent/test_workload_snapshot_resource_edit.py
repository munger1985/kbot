"""业务系统历史 SQL 采集策略编辑和并发校验。"""

from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from aiops_agent.application.configuration import AIOpsConfigurationService, ConfigurationScope
from aiops_agent.application.errors import AIOpsApplicationError
from platform_core.contracts.aiops import ManagedResourcePatch, TargetPatch
from platform_core.identity import uuid7


class WorkloadSnapshotResourceEditTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.system_id, self.link_id, self.resource_id, self.target_id = (uuid7() for _ in range(4))
        self.scope = ConfigurationScope(domain_id=100, principal_id="PORTAL:aiops", actor_id="operator-1", request_id="test", trace_id="test")
        self.system = SimpleNamespace(status="DISABLED")
        self.link = SimpleNamespace(resource_id=self.resource_id, status="ACTIVE")
        self.resource = SimpleNamespace(resource_id=self.resource_id, resource_code="DB-001", resource_type_key="DATABASE", display_name="DB", identity_kind="STABLE", lifecycle_status="ACTIVE", owner_group_id=uuid7(), location_name=None, row_version=3)
        self.target = SimpleNamespace(target_id=self.target_id, db_type="POSTGRESQL", version_code="16", environment="PROD", deployment_type="SINGLE_INSTANCE", rac_node_number=None, db_role="PRIMARY", connectivity_status="CONNECTED", row_version=7, workload_snapshot_policy_json={}, workload_next_run_at=None, workload_consecutive_failures=4)
        self.systems = SimpleNamespace(get_scoped=AsyncMock(return_value=self.system), get_system_resource_scoped=AsyncMock(return_value=self.link), get_resource_scoped=AsyncMock(return_value=self.resource), get_database_profile=AsyncMock(return_value=SimpleNamespace(target_id=self.target_id)))
        self.systems.find_system_for_target = AsyncMock(return_value=self.system)
        self.targets = SimpleNamespace(get_scoped=AsyncMock(return_value=self.target), list_source_bindings=AsyncMock(return_value=[]))
        self.uow = SimpleNamespace(business_systems=self.systems, targets=self.targets, flush=AsyncMock(), commit=AsyncMock())
        uow = self.uow

        class Context:
            async def __aenter__(self):
                return uow

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        self.service = object.__new__(AIOpsConfigurationService)
        self.service._uow_factory = Context
        self.service._now = lambda: datetime(2026, 10, 27, tzinfo=UTC)

    def request(self, enabled=True):
        return ManagedResourcePatch(database={"workload_snapshot_policy": {"enabled": enabled, "interval_minutes": 5, "retention_days": 30, "top_statement_limit": 100, "round_timeout_seconds": 47}, "target_row_version": 7, "business_system_id": self.system_id, "system_resource_id": self.link_id})

    async def test_three_engines_persist_policy_and_schedule(self):
        for engine, version in (("POSTGRESQL", "16"), ("MYSQL", "8.4"), ("ORACLE", "19c")):
            with self.subTest(engine=engine):
                self.target.db_type, self.target.version_code = engine, version
                await self.service.patch_managed_resource(scope=self.scope, resource_id=self.resource_id, request=self.request(), expected_version=3)
                self.assertTrue(self.target.workload_snapshot_policy_json["enabled"])
                self.assertEqual(47, self.target.workload_snapshot_policy_json["round_timeout_seconds"])
                self.assertEqual(self.service._now(), self.target.workload_next_run_at)
                self.assertEqual(0, self.target.workload_consecutive_failures)
                self.assertIsNone(self.target.workload_last_error_code)
                self.assertEqual(self.scope.actor_id, self.target.updated_by)
        self.assertEqual(3, self.uow.commit.await_count)

    async def test_disable_stops_scheduling_without_deleting_snapshots(self):
        await self.service.patch_managed_resource(scope=self.scope, resource_id=self.resource_id, request=self.request(False), expected_version=3)
        self.assertIsNone(self.target.workload_next_run_at)
        self.assertFalse(self.target.workload_snapshot_policy_json["enabled"])

    async def test_online_system_and_wrong_resource_are_rejected(self):
        self.system.status = "ENABLED"
        with self.assertRaisesRegex(AIOpsApplicationError, "运行中的业务系统不能修改"):
            await self.service.patch_managed_resource(scope=self.scope, resource_id=self.resource_id, request=self.request(), expected_version=3)
        self.system.status = "DISABLED"
        self.link.resource_id = uuid7()
        with self.assertRaisesRegex(AIOpsApplicationError, "不属于当前业务系统"):
            await self.service.patch_managed_resource(scope=self.scope, resource_id=self.resource_id, request=self.request(), expected_version=3)
        self.uow.commit.assert_not_awaited()

    async def test_stale_target_version_is_rejected(self):
        self.target.row_version = 8
        with self.assertRaisesRegex(AIOpsApplicationError, "资源已被其他请求更新"):
            await self.service.patch_managed_resource(scope=self.scope, resource_id=self.resource_id, request=self.request(), expected_version=3)
        self.uow.commit.assert_not_awaited()

    async def test_bound_target_policy_cannot_bypass_business_system_edit(self):
        for status in ("ENABLED", "DISABLED"):
            self.system.status = status
            with self.assertRaisesRegex(AIOpsApplicationError, "只能在下线后的业务系统编辑页修改"):
                await self.service.patch_target(scope=self.scope, target_id=self.target_id, request=TargetPatch(workload_snapshot_policy={"enabled": True}), expected_version=7)
        self.systems.find_system_for_target.assert_awaited_with(target_id=self.target_id, domain_id=self.scope.domain_id)
        self.uow.commit.assert_not_awaited()

    async def test_target_patch_persists_oracle_entitlements_with_other_capabilities(self):
        self.target.db_type = "ORACLE"
        self.target.version_code = "19c"
        self.target.readonly_connection_enabled = False
        self.target.controlled_change_enabled = False
        self.target.endpoint_json = None
        self.target.oracle_container_scope = None
        self.target.oracle_pdb_name = None
        self.target.capabilities_json = {"capability_probe": {"ok": True}}
        self.uow.session = SimpleNamespace(flush=AsyncMock())
        capabilities = {"capability_probe": {"ok": True}, "entitlements": ["ORACLE_DIAGNOSTICS_PACK"]}
        with patch("aiops_agent.application.configuration.target_service._target_detail", return_value="updated"), patch("aiops_agent.application.configuration.target_service.add_configuration_event", new_callable=AsyncMock):
            await self.service.patch_target(scope=self.scope, target_id=self.target_id,
                request=TargetPatch(capabilities=capabilities), expected_version=7)
        self.assertEqual(capabilities, self.target.capabilities_json)
        self.uow.commit.assert_awaited_once()

    async def test_resubmitting_unchanged_bound_policy_does_not_reschedule(self):
        self.target.workload_next_run_at = datetime(2026, 10, 28, tzinfo=UTC)
        with patch("aiops_agent.application.configuration.target_service._target_detail", return_value="unchanged"):
            result = await self.service.patch_target(scope=self.scope, target_id=self.target_id, request=TargetPatch(workload_snapshot_policy={"enabled": False}), expected_version=7)
        self.assertEqual("unchanged", result)
        self.assertEqual(datetime(2026, 10, 28, tzinfo=UTC), self.target.workload_next_run_at)
        self.uow.commit.assert_not_awaited()

    def test_policy_requires_all_scope_and_version_fields(self):
        for incomplete in ({"workload_snapshot_policy": {"enabled": True}}, {"target_row_version": 7}):
            with self.assertRaises(ValidationError):
                ManagedResourcePatch(database=incomplete)


if __name__ == "__main__":
    unittest.main()
