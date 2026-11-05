"""已有业务系统数据库连接、凭据和监控映射编辑回归。"""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from pydantic import ValidationError

from aiops_agent.application.configuration import AIOpsConfigurationService, ConfigurationScope
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.entities import ResourceSourceBindingEntity
from aiops_agent.ports.diagnostic_source import InstanceDiscoveryCandidate, InstanceDiscoveryResult
from platform_core.contracts.aiops import ManagedResourcePatch, TargetConnectionTest, TargetConnectionTestResult
from platform_core.identity import uuid7


class DatabaseResourceEditTest(IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 5, tzinfo=UTC)
        self.system_id, self.link_id, self.resource_id, self.target_id, self.source_id = (uuid7() for _ in range(5))
        scope_args = {"domain_id": 100, "principal_id": "PORTAL:aiops", "actor_id": "operator-1", "request_id": "test", "trace_id": "test"}
        if "tenant_id" in ConfigurationScope.__dataclass_fields__:
            scope_args.update(tenant_id=uuid7(), domain_id=uuid7(), actor_id=uuid7())
        self.scope = ConfigurationScope(**scope_args)
        self.system = SimpleNamespace(status="DISABLED")
        self.link = SimpleNamespace(resource_id=self.resource_id, status="ACTIVE")
        self.resource = SimpleNamespace(attributes_json={}, resource_id=self.resource_id, resource_code="DB-001", resource_type_key="DATABASE", display_name="ERP", identity_kind="STABLE", lifecycle_status="ACTIVE", owner_group_id=uuid7(), location_name=None, row_version=3)
        self.endpoint = {"host": "db.example", "port": 1521, "service": "ERP", "database": None, "tls_enabled": False, "tls_profile_ref": None}
        self.target = SimpleNamespace(target_id=self.target_id, db_type="ORACLE", version_code="19c", environment="PROD", deployment_type="SINGLE_INSTANCE", rac_node_number=None, db_role="PRIMARY", connectivity_status="CONNECTED", connectivity_version=2, row_version=7, endpoint_json=dict(self.endpoint), oracle_container_scope="PDB", oracle_pdb_name="ERP", observed_oracle_container_scope="PDB", observed_oracle_container_name="ERP", observed_oracle_database_name="ERPDB", diagnostic_credential_id=uuid7())
        self.locator = {"database_type": "ORACLE", "target_id": "1"*32, "oracle_container_scope": "PDB", "oracle_pdb_name": "ERP", "database_name": "ERPDB"}
        self.source = SimpleNamespace(source_type="OEM", declared_capabilities_json={})
        self.binding = self.make_binding(self.source_id)
        self.other = self.make_binding(uuid7())
        self.other.role = "SUPPLEMENTARY"
        self.bindings = [self.binding, self.other]
        self.systems = SimpleNamespace(get_scoped=AsyncMock(return_value=self.system), get_system_resource_scoped=AsyncMock(return_value=self.link), get_resource_scoped=AsyncMock(return_value=self.resource), get_database_profile=AsyncMock(return_value=SimpleNamespace(target_id=self.target_id)), get_resource_for_target=AsyncMock(return_value=self.resource))
        self.targets = SimpleNamespace(get_scoped=AsyncMock(return_value=self.target), list_source_bindings=AsyncMock(side_effect=lambda **_: list(self.bindings)), get_source_binding_scoped=AsyncMock(return_value=self.binding), get_source_binding_by_locator_any_status=AsyncMock(return_value=None), add_source_binding=AsyncMock())
        self.sources = SimpleNamespace(get_scoped=AsyncMock(return_value=self.source))
        self.uow = SimpleNamespace(business_systems=self.systems, targets=self.targets, diagnostic_sources=self.sources, flush=AsyncMock(), commit=AsyncMock())
        value = self.uow
        class Context:
            async def __aenter__(self):
                return value
            async def __aexit__(self, *args):
                return False
        self.service = object.__new__(AIOpsConfigurationService)
        self.service._uow_factory = Context
        self.service._now = lambda: self.now
        self.service._managed_credentials = SimpleNamespace(read=AsyncMock(return_value={"username": "reader", "password": "fixture-only"}), put=AsyncMock(return_value=SimpleNamespace(credential_id=uuid7())))
        self.service._instance_candidate_refs = Mock()
        self.service._instance_candidate_refs.decode.return_value = {"candidate_kind": "DATABASE", "source_locator_key": "1"*32, "source_locator": self.locator, "db_type": "ORACLE"}
        self.provider = SimpleNamespace(discover_instances=AsyncMock(return_value=InstanceDiscoveryResult(candidates=(InstanceDiscoveryCandidate(source_locator_key="1"*32, source_locator=self.locator, db_type="ORACLE", display_name="ERP"),))))
        self.service._instance_discovery_adapter = AsyncMock(return_value=self.provider)

    def make_binding(self, source_id):
        return ResourceSourceBindingEntity(resource_id=self.resource_id, resource_source_binding_id=uuid7(), target_id=self.target_id, diagnostic_source_id=source_id, source_locator_key="old", source_locator_json={"old": True}, role="PRIMARY", priority=100, status="ACTIVE", health_status="HEALTHY", row_version=4, health_version=1, capability_scope_json=None, mapping_overrides_json=None, query_budget_json=None, created_at=self.now, updated_at=self.now, created_by="operator", updated_by="operator")

    def request(self, **changes):
        return ManagedResourcePatch(database={"target_row_version": 7, "business_system_id": self.system_id, "system_resource_id": self.link_id, **changes})

    def mapping(self, **changes):
        return {"source_id": self.source_id, "candidate_ref": "signed-reference", "binding_id": self.binding.resource_source_binding_id, "binding_row_version": 4, **changes}

    async def save(self, request):
        with patch("aiops_agent.application.configuration.database_resource_edit.add_configuration_event", new_callable=AsyncMock), patch("aiops_agent.application.configuration.business_system_service.add_configuration_event", new_callable=AsyncMock), patch("aiops_agent.application.configuration.diagnostic_source_service.add_configuration_event", new_callable=AsyncMock):
            return await self.service.patch_managed_resource(scope=self.scope, resource_id=self.resource_id, request=request, expected_version=3)

    async def test_existing_target_without_mapping_can_add_oem(self):
        self.bindings.clear()
        self.targets.add_source_binding.side_effect = lambda entity: self.bindings.append(entity)
        result = await self.save(self.request(monitoring=self.mapping(binding_id=None, binding_row_version=None)))
        self.targets.add_source_binding.assert_awaited_once()
        self.assertEqual(self.target_id, result.database.target_id)
        self.assertEqual("1"*32, result.database.monitoring_bindings[0].source_locator_key)
        self.uow.commit.assert_awaited_once()

    async def test_existing_oem_scope_change_requires_consistent_mapping(self):
        self.bindings = [self.binding]
        self.binding.source_locator_json = dict(self.locator)
        with self.assertRaises(AIOpsApplicationError):
            await self.save(self.request(oracle_container_scope="CDB_ROOT", oracle_pdb_name=None))
        self.uow.commit.assert_not_awaited()

    def test_connection_test_rejects_missing_or_ambiguous_credentials(self):
        values = {"db_type": "ORACLE", "version_code": "19c", "endpoint": self.endpoint, "oracle_container_scope": "PDB", "oracle_pdb_name": "ERP"}
        with self.assertRaises(ValidationError):
            TargetConnectionTest(**values)
        with self.assertRaises(ValidationError):
            TargetConnectionTest(**values, saved_target_id=self.target_id, diagnostic_credential={"username": "reader", "password": "fixture-only"})

    async def test_oem_rebind_revalidates_candidate_preserves_identity_and_other_mapping(self):
        old_id = self.binding.resource_source_binding_id
        self.binding.mapping_overrides_json = {"preserved": True}
        result = await self.save(self.request(monitoring=self.mapping(source_id=uuid7())))
        self.assertEqual(old_id, self.binding.resource_source_binding_id)
        self.assertEqual(self.target_id, result.database.target_id)
        self.assertEqual("1"*32, self.binding.source_locator_key)
        self.assertEqual({"old": True}, self.other.source_locator_json)
        self.assertEqual({"preserved": True}, self.binding.mapping_overrides_json)
        self.provider.discover_instances.assert_awaited_once()
        self.targets.add_source_binding.assert_not_awaited()
        self.uow.commit.assert_awaited_once()
        self.service._managed_credentials.put.assert_not_awaited()

    async def test_same_source_can_choose_new_guid(self):
        await self.save(self.request(monitoring=self.mapping()))
        self.assertEqual(self.source_id, self.binding.diagnostic_source_id)
        self.assertEqual(self.locator, self.binding.source_locator_json)

    async def test_stale_binding_target_and_resource_versions_block_writes(self):
        for kind in ("binding", "target", "resource"):
            with self.subTest(kind=kind):
                obj = getattr(self, kind)
                old = obj.row_version
                obj.row_version = old + 1
                with self.assertRaises(AIOpsApplicationError):
                    await self.save(self.request(monitoring=self.mapping()))
                obj.row_version = old
        self.uow.commit.assert_not_awaited()
        self.provider.discover_instances.assert_not_awaited()

    async def test_online_wrong_scope_missing_binding_and_disabled_binding_rejected(self):
        self.system.status = "ENABLED"
        with self.assertRaises(AIOpsApplicationError):
            await self.save(self.request(monitoring=self.mapping()))
        self.system.status = "DISABLED"
        self.link.resource_id = uuid7()
        with self.assertRaises(AIOpsApplicationError):
            await self.save(self.request(monitoring=self.mapping()))
        self.link.resource_id = self.resource_id
        self.targets.get_source_binding_scoped.return_value = None
        with self.assertRaises(AIOpsApplicationError):
            await self.save(self.request(monitoring=self.mapping()))
        self.targets.get_source_binding_scoped.return_value = self.binding
        self.binding.status = "DISABLED"
        with self.assertRaises(AIOpsApplicationError):
            await self.save(self.request(monitoring=self.mapping()))
        self.uow.commit.assert_not_awaited()

    async def test_changed_discovery_identity_and_wrong_pdb_are_rejected(self):
        candidate = InstanceDiscoveryCandidate(source_locator_key="1"*32, source_locator={**self.locator, "oracle_pdb_name": "OTHER"}, db_type="ORACLE", display_name="other")
        self.provider.discover_instances.return_value = InstanceDiscoveryResult(candidates=(candidate,))
        with self.assertRaises(AIOpsApplicationError):
            await self.save(self.request(monitoring=self.mapping()))
        self.service._instance_candidate_refs.decode.return_value["source_locator"] = candidate.source_locator
        with self.assertRaises(AIOpsApplicationError):
            await self.save(self.request(monitoring=self.mapping()))
        self.uow.commit.assert_not_awaited()
        self.assertEqual("old", self.binding.source_locator_key)

    async def test_other_target_locator_and_duplicate_source_are_not_overwritten(self):
        self.targets.get_source_binding_by_locator_any_status.return_value = self.other
        with self.assertRaises(AIOpsApplicationError):
            await self.save(self.request(monitoring=self.mapping()))
        self.targets.get_source_binding_by_locator_any_status.return_value = None
        self.other.diagnostic_source_id = self.source_id
        with self.assertRaises(AIOpsApplicationError):
            await self.save(self.request(monitoring=self.mapping()))
        self.uow.commit.assert_not_awaited()

    async def test_selected_binding_can_keep_its_own_locator(self):
        self.targets.get_source_binding_by_locator_any_status.return_value = self.binding
        await self.save(self.request(monitoring=self.mapping()))
        self.uow.commit.assert_awaited_once()

    async def test_raw_oem_guid_cannot_bypass_discovery(self):
        mapping = {"source_id": self.source_id, "binding_id": self.binding.resource_source_binding_id, "binding_row_version": 4, "source_binding": {"source_id": self.source_id, "source_locator_key": "1"*32, "source_locator": self.locator}}
        with self.assertRaises(AIOpsApplicationError):
            await self.save(self.request(monitoring=mapping))
        self.uow.commit.assert_not_awaited()

    async def test_unchanged_connection_does_not_clear_secret_or_recheck(self):
        await self.save(self.request(endpoint=self.endpoint))
        self.assertEqual("CONNECTED", self.target.connectivity_status)
        self.service._managed_credentials.read.assert_not_awaited()
        self.service._managed_credentials.put.assert_not_awaited()

    async def test_endpoint_edit_preserves_secret_and_schedules_recheck(self):
        self.bindings.clear()
        await self.save(self.request(endpoint={**self.endpoint, "host": "new.example", "port": 1522}))
        self.assertEqual("new.example", self.target.endpoint_json["host"])
        self.assertEqual("CHECKING", self.target.connectivity_status)
        self.assertEqual(3, self.target.connectivity_version)
        self.assertIsNone(self.target.observed_oracle_database_name)
        self.service._managed_credentials.put.assert_not_awaited()

    async def test_username_only_keeps_saved_password_and_password_only_keeps_username(self):
        self.bindings.clear()
        await self.save(self.request(diagnostic_username="new_reader"))
        self.assertEqual({"username": "new_reader", "password": "fixture-only"}, self.service._managed_credentials.put.call_args.kwargs["values"])
        await self.save(self.request(diagnostic_password="new-fixture-only"))
        self.assertEqual({"username": "reader", "password": "new-fixture-only"}, self.service._managed_credentials.put.call_args.kwargs["values"])

    async def test_connection_and_oem_identity_can_change_together(self):
        self.bindings = [self.binding]
        self.locator["oracle_pdb_name"] = "NEWPDB"
        self.provider.discover_instances.return_value = InstanceDiscoveryResult(candidates=(InstanceDiscoveryCandidate(source_locator_key="1"*32, source_locator=self.locator, db_type="ORACLE", display_name="ERP"),))
        await self.save(self.request(endpoint={**self.endpoint, "service": "NEWPDB"}, oracle_pdb_name="NEWPDB", monitoring=self.mapping()))
        self.assertEqual("NEWPDB", self.target.oracle_pdb_name)
        self.assertEqual("NEWPDB", self.binding.source_locator_json["oracle_pdb_name"])

    async def test_cannot_change_deployment_or_rac_node_identity(self):
        for change in ({"deployment_type": "RAC", "rac_node_number": 1}, {"rac_node_number": 2}):
            with self.assertRaises(AIOpsApplicationError):
                await self.save(ManagedResourcePatch(database=change))
        self.uow.commit.assert_not_awaited()

    async def test_existing_connection_test_reuses_credentials_without_persistence(self):
        self.service._target_connection_tester = AsyncMock(return_value=TargetConnectionTestResult(ok=True))
        request = TargetConnectionTest(db_type="ORACLE", version_code="19c", endpoint={**self.endpoint, "host": "draft.example"}, oracle_container_scope="PDB", oracle_pdb_name="ERP", saved_target_id=self.target_id, diagnostic_username="new_reader")
        await self.service.test_target_connection(scope=self.scope, request=request)
        tested = self.service._target_connection_tester.call_args.args[0]
        self.assertEqual("draft.example", tested.endpoint.host)
        self.assertEqual("new_reader", tested.diagnostic_credential.username)
        self.assertEqual("fixture-only", tested.diagnostic_credential.password)
        self.assertEqual("db.example", self.target.endpoint_json["host"])
        self.uow.commit.assert_not_awaited()
        self.service._managed_credentials.put.assert_not_awaited()

    async def test_saved_connection_test_rejects_out_of_scope_or_wrong_engine(self):
        request = TargetConnectionTest(db_type="ORACLE", version_code="19c", endpoint=self.endpoint, oracle_container_scope="PDB", oracle_pdb_name="ERP", saved_target_id=self.target_id)
        self.service._target_connection_tester = AsyncMock()
        self.targets.get_scoped.return_value = None
        with self.assertRaises(AIOpsApplicationError):
            await self.service.test_target_connection(scope=self.scope, request=request)
        self.targets.get_scoped.return_value = self.target
        self.target.db_type = "MYSQL"
        with self.assertRaises(AIOpsApplicationError):
            await self.service.test_target_connection(scope=self.scope, request=request)
        self.service._target_connection_tester.assert_not_awaited()

    def test_patch_requires_context_and_rejects_secret_clearing(self):
        for change in ({"monitoring": self.mapping()}, {"endpoint": self.endpoint}, {"diagnostic_password": "fixture-only"}):
            with self.assertRaises(ValidationError):
                ManagedResourcePatch(database=change)
        for change in ({"diagnostic_password": ""}, {"diagnostic_password": None}, {"monitoring": None}, {"diagnostic_username": ""}):
            with self.assertRaises(ValidationError):
                self.request(**change)
        for change in ({"binding_id": None}, {"binding_row_version": None}):
            with self.assertRaises(ValidationError):
                self.request(monitoring=self.mapping(**change))


if __name__ == "__main__":
    import unittest
    unittest.main()
