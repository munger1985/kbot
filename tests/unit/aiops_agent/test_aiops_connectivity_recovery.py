"""AIOps 连通性检查异常恢复测试。"""

import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from aiops_agent.application.diagnostic_sources.connectivity_check import (
    DiagnosticSourceConnectivityCheckService,
)
from aiops_agent.ports.diagnostic_source import SourceHealthResult
from aiops_agent.repositories.monitoring import DiagnosticSourceRepository
from aiops_agent.repositories.target import TargetRepository
from aiops_agent.scheduling.connectivity import AIOpsConnectivityScheduler
from platform_core.identity import uuid7


class _HealthyAdapter:
    async def health_check(self, request):
        del request
        return SourceHealthResult(
            healthy=True,
            adapter_id="alertmanager",
            adapter_version="1.0.0",
            discovered_capabilities=("health.check",),
        )


class _Registry:
    def __init__(self):
        self.create_count = 0

    def create(self, context, *, capability):
        del context, capability
        self.create_count += 1
        return _HealthyAdapter()


class _SecretStore:
    async def resolve(self, reference):
        del reference
        return SimpleNamespace(values={"webhook_secret": "test-secret"})


class _DiagnosticSourceRepository:
    def __init__(self, source):
        self.source = source
        self.read_count = 0
        self.expected_versions = []

    async def get_scoped(self, **kwargs):
        del kwargs
        self.read_count += 1
        self.source.row_version = self.read_count
        return self.source

    async def update_connectivity(self, **kwargs):
        self.expected_versions.append(kwargs["expected_config_version"])
        return len(self.expected_versions) == 2


class _Runs:
    async def database_now(self):
        return datetime(2026, 9, 1, tzinfo=UTC)


class _Uow:
    def __init__(self, repository):
        self.diagnostic_sources = repository
        self.runs = _Runs()
        self.commit_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    async def commit(self):
        self.commit_count += 1


class ConnectivityRecoveryTest(unittest.IsolatedAsyncioTestCase):
    async def test_oracle_claims_use_server_cursor_without_row_limit(self):
        due_before = datetime(2026, 9, 1, tzinfo=UTC)
        pending_before = datetime(2026, 8, 31, tzinfo=UTC)
        for repository_type in (
            DiagnosticSourceRepository,
            TargetRepository,
        ):
            with self.subTest(repository=repository_type.__name__):
                repository = repository_type(AsyncMock())
                repository._claim_oracle_uuid = AsyncMock(
                    return_value=None
                )

                claimed = await repository.claim_due_connectivity(
                    due_before=due_before,
                    pending_before=pending_before,
                )

                self.assertIsNone(claimed)
                arguments = repository._claim_oracle_uuid.await_args.kwargs
                self.assertIn("FOR UPDATE OF", arguments["plsql"])
                self.assertNotIn("FETCH FIRST", arguments["plsql"])
                self.assertEqual(
                    {
                        "due_before": due_before,
                        "pending_before": pending_before,
                    },
                    arguments["parameters"],
                )

    async def test_scheduler_continues_after_one_iteration_fails(self):
        scheduler = AIOpsConnectivityScheduler(
            uow_factory=Mock(),
            scheduler_id="test-scheduler",
            interval_seconds=3600,
            jitter_seconds=0,
            scan_interval_seconds=0.001,
        )
        calls = 0

        async def run_once():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary database failure")
            scheduler.stop()
            return False

        scheduler.run_once = run_once

        with patch(
            "aiops_agent.scheduling.connectivity.logger"
        ) as logger:
            await asyncio.wait_for(scheduler.run_forever(), timeout=1)

        self.assertEqual(2, calls)
        logger.opt.return_value.error.assert_called_once()

    async def test_source_check_retries_after_config_version_changes(self):
        source_id = uuid7()
        request_id = uuid7()
        source = SimpleNamespace(
            diagnostic_source_id=source_id,
            domain_id=1,
            source_type="ALERTMANAGER",
            adapter_id="alertmanager",
            adapter_version="1.0.0",
            row_version=1,
            connectivity_version=1,
            endpoint=None,
            auth_credential_id=None,
            webhook_credential_id=uuid7(),
            connectivity_check_request_id=request_id,
            declared_capabilities_json={},
            config_json={},
        )
        repository = _DiagnosticSourceRepository(source)
        uow = _Uow(repository)
        registry = _Registry()
        service = DiagnosticSourceConnectivityCheckService(
            uow_factory=lambda: uow,
            diagnostic_source_registry=registry,
            secret_store=_SecretStore(),
        )

        await service.execute(
            {
                "aggregate_id": str(source_id),
                "domain_id": 1,
                "trace_id": "test-trace",
                "details": {
                    "connectivity_check_request_id": str(request_id)
                },
            }
        )

        self.assertEqual([1, 2], repository.expected_versions)
        self.assertEqual(2, registry.create_count)
        self.assertEqual(1, uow.commit_count)


if __name__ == "__main__":
    unittest.main()
