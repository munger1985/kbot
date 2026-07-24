"""AIOps 步骤 6 只读诊断目录、Grant 与 Executor 测试。"""

from __future__ import annotations

import hashlib
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

from pydantic import ValidationError
from fastapi.testclient import TestClient

from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.diagnostics.grants import (
    DiagnosticGrantCodec,
    DiagnosticGrantError,
    canonical_sha256,
)
from aiops_agent.diagnostics.validation import validate_readonly_template
from aiops_agent.executor import DiagnosticExecutorService
from aiops_agent.bootstrap.executor import create_aiops_executor
from aiops_agent.config import AIOpsSettings
from aiops_agent.executor.drivers import (
    DiagnosticDriverError,
    DriverQueryResult,
)
from aiops_agent.orchestration import (
    BlueprintRegistry,
    build_database_diagnostic_blueprint,
)
from aiops_agent.ports.secret_store import ResolvedSecret
from platform_core.contracts.aiops.executor import (
    DiagnosticConnectionProfile,
    DiagnosticExecutionGrant,
    DiagnosticLimits,
    ReadDiagnosticRequest,
)
from platform_core.identity import uuid7
from platform_core.security import build_scoped_internal_auth_headers


class FakeDriver:
    db_type = "ORACLE"

    async def execute(self, **kwargs):
        return DriverQueryResult(
            columns=(
                "product",
                "version",
                "instance_name",
                "database_role",
                "server_time",
            ),
            rows=(
                (
                    "Oracle Database",
                    "23.6.0.24.10",
                    "KBOT4",
                    "PRIMARY",
                    datetime(2026, 7, 24, tzinfo=UTC),
                ),
            ),
            truncated=False,
            db_version="23.6.0.24.10",
        )


class FailingDriver:
    db_type = "ORACLE"

    async def execute(self, **kwargs):
        raise DiagnosticDriverError(
            "PRIVILEGE_MISSING", retryable=False
        )


class DiagnosticCatalogTest(unittest.TestCase):
    def test_catalog_contains_oracle_mysql_parity(self) -> None:
        registry = DiagnosticRegistry.load()
        self.assertEqual(12, len(registry.tools))
        pairs = {
            (item.definition.db_type, item.definition.tool_id)
            for item in registry.tools
        }
        for tool_id in (
            "db.instance.identity",
            "db.session.active",
            "db.session.blocking_chain",
            "db.storage.capacity",
        ):
            self.assertIn(("ORACLE", tool_id), pairs)
            self.assertIn(("MYSQL", tool_id), pairs)

    def test_capability_and_entitlement_selection_is_exact(self) -> None:
        registry = DiagnosticRegistry.load()
        with self.assertRaises(LookupError):
            registry.resolve(
                tool_id="db.session.active",
                tool_version="1.0.0",
                db_type="ORACLE",
                db_version="19c",
                capabilities=set(),
                entitlements=set(),
            )
        selected = registry.resolve(
            tool_id="db.session.active",
            tool_version="1.0.0",
            db_type="ORACLE",
            db_version="19c",
            capabilities={"dynamic_performance_views"},
            entitlements=set(),
        )
        self.assertEqual("oracle_19_plus_gv", selected.definition.variant)

    def test_dangerous_or_multiple_statement_template_is_rejected(self) -> None:
        definition = DiagnosticRegistry.load().tools[0].definition
        for sql in (
            "SELECT 1 FROM dual; DROP TABLE x",
            "BEGIN NULL; END",
            "SELECT 1 FROM dual FOR UPDATE",
            "SELECT utl_http.request('https://example') FROM dual",
        ):
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                validate_readonly_template(sql, definition)

    def test_database_blueprint_always_fences_on_identity(self) -> None:
        blueprint = build_database_diagnostic_blueprint(
            ("db.instance.identity", "db.session.active")
        )
        BlueprintRegistry.validate(blueprint, max_tasks=8)
        active = next(
            item
            for item in blueprint.tasks
            if item.task_key == "diagnostic:db.session.active"
        )
        self.assertEqual(
            ("diagnostic:db.instance.identity",), active.depends_on
        )


class DiagnosticGrantTest(unittest.TestCase):
    def setUp(self) -> None:
        self.codec = DiagnosticGrantCodec(
            secret="g" * 32,
            issuer="kbot-aiops-worker",
            audience="kbot-aiops-db-executor",
        )

    def _grant(self, *, expires_at=None):
        now = datetime.now(UTC).replace(microsecond=0)
        tool = DiagnosticRegistry.load().resolve(
            tool_id="db.instance.identity",
            tool_version="1.0.0",
            db_type="ORACLE",
            db_version="23ai",
            capabilities=set(),
            entitlements=set(),
        )
        return DiagnosticExecutionGrant(
            issuer="kbot-aiops-worker",
            audience="kbot-aiops-db-executor",
            grant_id=uuid7(),
            issued_at=now,
            expires_at=expires_at or now + timedelta(seconds=30),
            run_id=uuid7(),
            task_id=uuid7(),
            lease_token_hash="a" * 64,
            target_id=uuid7(),
            target_row_version=1,
            db_type="ORACLE",
            connection_profile=DiagnosticConnectionProfile(
                host="db.internal",
                port=1521,
                service="KBOT4",
            ),
            diagnostic_secret_ref="env://AIOPS_TEST_DATABASE",
            tool_id=tool.definition.tool_id,
            tool_version=tool.definition.version,
            variant=tool.definition.variant,
            template_sha256=tool.definition.template_sha256,
            parameters_sha256=canonical_sha256({}),
            capability_snapshot_hash="b" * 64,
            limits=DiagnosticLimits(
                statement_timeout_seconds=8,
                max_result_rows=1,
                max_result_bytes=65536,
            ),
            trace_id="trace-1",
        )

    def test_signed_grant_rejects_tampering(self) -> None:
        token = self.codec.issue(self._grant())
        head, body, signature = token.split(".")
        tampered = ".".join((head, body[:-1] + "A", signature))
        with self.assertRaises(DiagnosticGrantError):
            self.codec.verify(tampered)

    def test_request_rejects_sql_connection_and_password(self) -> None:
        payload = {
            "executor_request_id": str(uuid7()),
            "grant": self.codec.issue(self._grant()),
            "parameters": {},
            "idempotency_key": "request-1",
            "sql": "SELECT * FROM users",
            "password": "secret",
        }
        with self.assertRaises(ValidationError):
            ReadDiagnosticRequest.model_validate(payload)

    def test_executor_http_maps_unknown_fields_to_safe_400(self) -> None:
        app = create_aiops_executor(AIOpsSettings())
        headers = build_scoped_internal_auth_headers(
            audience="kbot-aiops-db-executor",
            caller_service="kbot-aiops-worker",
            scopes=("db-executor.diagnostic",),
        )
        with TestClient(app) as client:
            response = client.post(
                "/internal/v1/db-executor/diagnostics",
                headers=headers,
                json={
                    "executor_request_id": str(uuid7()),
                    "grant": "x" * 64,
                    "parameters": {},
                    "idempotency_key": "request-1",
                    "sql": "SELECT 1",
                },
            )
        self.assertEqual(400, response.status_code)
        self.assertNotIn("SELECT 1", response.text)


class DiagnosticExecutorTest(unittest.IsolatedAsyncioTestCase):
    async def _execute(self, driver):
        grants = DiagnosticGrantTest()
        grants.setUp()
        grant = grants._grant()
        request = ReadDiagnosticRequest(
            executor_request_id=uuid7(),
            grant=grants.codec.issue(grant),
            parameters={},
            idempotency_key="request-1",
        )
        secret_store = AsyncMock()
        secret_store.resolve.return_value = ResolvedSecret(
            values={"username": "private-user", "password": "hidden"},
            fingerprint="secret-1",
        )
        service = DiagnosticExecutorService(
            registry=DiagnosticRegistry.load(),
            grant_codec=grants.codec,
            secret_store=secret_store,
            drivers=(driver,),
            hard_limits=DiagnosticLimits(
                statement_timeout_seconds=30,
                max_result_rows=100,
                max_result_bytes=1048576,
            ),
            concurrency=2,
        )
        return await service.execute(request)

    async def test_executor_returns_bounded_typed_observation(self) -> None:
        result = await self._execute(FakeDriver())
        self.assertEqual("SUCCEEDED", result.status)
        self.assertIsNotNone(result.observation)
        assert result.observation is not None
        self.assertEqual(1, result.observation.row_count)
        self.assertEqual(64, len(result.observation.result_sha256))
        serialized = result.model_dump_json()
        self.assertNotIn("hidden", serialized)
        self.assertNotIn("private-user", serialized)

    async def test_database_failure_becomes_structured_gap(self) -> None:
        result = await self._execute(FailingDriver())
        self.assertEqual("GAP", result.status)
        self.assertEqual("PRIVILEGE_MISSING", result.error_code)
        self.assertFalse(result.retryable)

    async def test_parameter_hash_mismatch_is_rejected_before_secret(self) -> None:
        grants = DiagnosticGrantTest()
        grants.setUp()
        request = ReadDiagnosticRequest(
            executor_request_id=uuid7(),
            grant=grants.codec.issue(grants._grant()),
            parameters={"unexpected": 1},
            idempotency_key="request-2",
        )
        secret_store = AsyncMock()
        service = DiagnosticExecutorService(
            registry=DiagnosticRegistry.load(),
            grant_codec=grants.codec,
            secret_store=secret_store,
            drivers=(FakeDriver(),),
            hard_limits=DiagnosticLimits(
                statement_timeout_seconds=30,
                max_result_rows=100,
                max_result_bytes=1048576,
            ),
            concurrency=1,
        )
        with self.assertRaises(DiagnosticGrantError):
            await service.execute(request)
        secret_store.resolve.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
