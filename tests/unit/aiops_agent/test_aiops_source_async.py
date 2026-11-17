"""监控源保存仅投递检查，Worker 独立回写真实结果且不自动启用。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from aiops_agent.application.configuration import AIOpsConfigurationService, ConfigurationScope
from aiops_agent.application.diagnostic_sources.connectivity_check import DiagnosticSourceConnectivityCheckService
from aiops_agent.adapters.diagnostic_sources import DiagnosticSourceAdapterCatalog
from aiops_agent.entities import DiagnosticSourceEntity
from aiops_agent.ports.diagnostic_source import SourceHealthResult
from platform_core.contracts.aiops import DiagnosticSourcePatch
from platform_core.identity import uuid7


class _Context:
    def __init__(self, uow):
        self.uow = uow

    async def __aenter__(self):
        return self.uow

    async def __aexit__(self, *_args):
        return False


class SourceAsyncConnectivityTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = datetime.now(UTC)
        self.scope = ConfigurationScope(domain_id=100,
            principal_id="PORTAL:test", actor_id="test-user", request_id="test", trace_id="test")
        self.entity = DiagnosticSourceEntity(
            diagnostic_source_id=uuid7(), domain_id=self.scope.domain_id,
            display_name="本地 Loki", source_type="LOKI", adapter_id="loki", adapter_version="1.0.0",
            endpoint="http://old.invalid:3100/", auth_credential_id=uuid7(), webhook_credential_id=None,
            tls_profile_ref=None, declared_capabilities_json={"log.query": {}},
            discovered_capabilities_json={"old": {}}, config_json={}, status="ENABLED",
            connectivity_status="CONNECTED", connectivity_check_request_id=None,
            connectivity_check_requested_at=None, last_connectivity_check_at=self.now-timedelta(minutes=1),
            last_connectivity_success_at=self.now-timedelta(minutes=1), last_error_code=None,
            row_version=3, connectivity_version=4, created_at=self.now, updated_at=self.now,
            created_by=self.scope.actor_id, updated_by=self.scope.actor_id)
        self.repository = AsyncMock()
        self.repository.get_scoped.return_value = self.entity
        self.uow = SimpleNamespace(diagnostic_sources=self.repository, managed_credentials=object(),
            outbox=AsyncMock(), flush=AsyncMock(), session=AsyncMock(), commit=AsyncMock(),
            runs=SimpleNamespace(database_now=AsyncMock(return_value=self.now)))
        self.service = object.__new__(AIOpsConfigurationService)
        self.service._uow_factory = lambda: _Context(self.uow)
        self.service._diagnostic_source_catalog = DiagnosticSourceAdapterCatalog()
        self.service._diagnostic_source_registry = SimpleNamespace(create=AsyncMock())
        self.service._managed_credentials = SimpleNamespace(read=AsyncMock(), put=AsyncMock())

    async def test_patch_commits_pending_check_without_reading_credentials_or_network(self):
        previous_success = self.entity.last_connectivity_success_at
        response = await asyncio.wait_for(self.service.patch_diagnostic_source(
            scope=self.scope, source_id=self.entity.diagnostic_source_id,
            request=DiagnosticSourcePatch(endpoint="http://127.0.0.1:3100/"), expected_version=3), timeout=1)
        self.assertTrue(response.connectivity_check_pending)
        self.assertEqual(response.connectivity_status, "CHECKING")
        self.assertEqual(response.status, "DISABLED")
        self.assertEqual(self.entity.connectivity_version, 5)
        self.assertEqual(self.entity.last_connectivity_success_at, previous_success)
        self.assertIsNone(self.entity.discovered_capabilities_json)
        self.service._managed_credentials.read.assert_not_awaited()
        self.service._diagnostic_source_registry.create.assert_not_called()
        self.uow.commit.assert_awaited_once()
        event = self.uow.outbox.add.await_args_list[-1].args[0]
        self.assertEqual(event.event_type, "SOURCE_CONNECTIVITY_CHECK_REQUESTED")
        self.assertEqual(event.payload_json["details"]["connectivity_check_request_id"],
                         str(self.entity.connectivity_check_request_id))
        self.assertNotIn("password", str(event.payload_json))
        self.assertNotIn("token", str(event.payload_json))

    async def test_name_only_patch_does_not_disable_or_schedule_network(self):
        response = await self.service.patch_diagnostic_source(scope=self.scope,
            source_id=self.entity.diagnostic_source_id,
            request=DiagnosticSourcePatch(display_name="Loki 新名称"), expected_version=3)
        self.assertFalse(response.connectivity_check_pending)
        self.assertEqual(response.status, "ENABLED")
        self.assertEqual(response.connectivity_status, "CONNECTED")
        self.assertEqual(self.entity.connectivity_version, 4)
        self.assertEqual([call.args[0].event_type for call in self.uow.outbox.add.await_args_list],
                         ["DIAGNOSTIC_SOURCE_UPDATED"])

    async def test_outbox_failure_does_not_commit_configuration(self):
        self.uow.outbox.add.side_effect = RuntimeError("模拟投递失败")
        with self.assertRaises(RuntimeError):
            await self.service.patch_diagnostic_source(scope=self.scope,
                source_id=self.entity.diagnostic_source_id,
                request=DiagnosticSourcePatch(endpoint="http://127.0.0.1:3100/"), expected_version=3)
        self.uow.commit.assert_not_awaited()

    async def test_worker_records_success_or_failure_without_enabling_source(self):
        self.entity.status = "DISABLED"
        worker = DiagnosticSourceConnectivityCheckService(uow_factory=lambda: _Context(self.uow),
            diagnostic_source_registry=None, secret_store=None)
        self.repository.update_connectivity.return_value = True
        for healthy in (True, False):
            self.uow.commit.reset_mock()
            changed = await worker._record_result(
                source_id=self.entity.diagnostic_source_id, request_id=uuid7(),
                snapshot={"config_version": 3, "connectivity_version": 5},
                result=SourceHealthResult(healthy=healthy,
                    error_code=None if healthy else "SOURCE_UNREACHABLE",
                    adapter_id="loki", adapter_version="1.0.0"))
            self.assertTrue(changed)
            arguments = self.repository.update_connectivity.await_args.kwargs
            self.assertEqual(arguments["connectivity_status"], "CONNECTED" if healthy else "UNREACHABLE")
            self.assertEqual(arguments["expected_config_version"], 3)
            self.assertEqual(self.entity.status, "DISABLED")
            self.uow.commit.assert_awaited_once()

    async def test_superseded_worker_request_does_not_probe_new_configuration(self):
        self.entity.connectivity_check_request_id = uuid7()
        registry = SimpleNamespace(create=AsyncMock())
        worker = DiagnosticSourceConnectivityCheckService(uow_factory=lambda: _Context(self.uow),
            diagnostic_source_registry=registry, secret_store=None)
        payload = {"aggregate_id": str(self.entity.diagnostic_source_id),
                   "domain_id": str(self.scope.domain_id),
                   "details": {"connectivity_check_request_id": str(uuid7())}}
        await worker.execute(payload)
        registry.create.assert_not_called()
        self.repository.update_connectivity.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
