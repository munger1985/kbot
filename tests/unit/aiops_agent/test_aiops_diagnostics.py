"""AIOps 步骤 6 只读诊断目录、Grant 与 Executor 测试。"""

from __future__ import annotations

import hashlib
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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
    OracleDiagnosticDriver,
)
from aiops_agent.ports.secret_store import ResolvedSecret
from aiops_agent.orchestration import (
    BlueprintRegistry,
    build_database_diagnostic_blueprint,
)
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


class _TimeoutCursor:
    description = (("SID",),)

    def __init__(self) -> None:
        self.execute_count = 0

    async def execute(self, _sql, _parameters=None):
        self.execute_count += 1
        if self.execute_count == 2:
            raise TimeoutError

    async def fetchmany(self, _limit):
        return ()

    def close(self):
        return None


class _TimeoutConnection:
    version = "19.24.0.0.0"
    call_timeout = 0
    module = ""
    action = ""

    def __init__(self) -> None:
        self._cursor = _TimeoutCursor()

    def cursor(self):
        return self._cursor

    async def rollback(self):
        return None

    async def close(self):
        return None


class _OracleErrorCursor(_TimeoutCursor):
    def __init__(self, code: int = 1861) -> None:
        super().__init__()
        self.code = code

    async def execute(self, _sql, _parameters=None):
        self.execute_count += 1
        if self.execute_count == 2:
            error = type(
                "OracleErrorInfo",
                (),
                {
                    "code": self.code,
                    "full_code": f"ORA-{self.code:05d}",
                },
            )()
            import oracledb

            raise oracledb.DatabaseError(error)


class _OracleErrorConnection(_TimeoutConnection):
    def __init__(self, code: int = 1861) -> None:
        self._cursor = _OracleErrorCursor(code)


class _FakeOracleLob:
    def __init__(self, value, lob_type) -> None:
        self.value = value
        self.type = lob_type
        self.read_amount = None

    async def read(self, offset=1, amount=None):
        self.read_amount = (offset, amount)
        return self.value[:amount]


class _OracleResultCursor(_TimeoutCursor):
    def __init__(self, value, database_type=None) -> None:
        super().__init__()
        self.value = value
        self.description = (("PLAN_TEXT", database_type),)

    async def execute(self, _sql, _parameters=None):
        self.execute_count += 1

    async def fetchmany(self, _limit):
        return ((self.value,),)


class _OracleResultConnection(_TimeoutConnection):
    def __init__(self, value, database_type=None) -> None:
        self._cursor = _OracleResultCursor(value, database_type)


class OracleDiagnosticDriverTimeoutTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _profile() -> DiagnosticConnectionProfile:
        return DiagnosticConnectionProfile(
            host="db.internal",
            port=1521,
            service="PDB1",
            tls_enabled=False,
        )

    @staticmethod
    def _limits() -> DiagnosticLimits:
        return DiagnosticLimits(
            statement_timeout_seconds=10,
            max_result_rows=10,
            max_result_bytes=65536,
        )

    async def _execute(self):
        return await OracleDiagnosticDriver().execute_dynamic(
            profile=self._profile(),
            secret=ResolvedSecret(
                values={"username": "readonly", "password": "hidden"},
                fingerprint="test",
            ),
            sql="SELECT sid FROM v$session",
            parameters={},
            limits=self._limits(),
            trace_id="trace-timeout",
        )

    async def test_connect_timeout_has_distinct_error_code(self) -> None:
        with patch(
            "aiops_agent.executor.drivers.oracle.oracledb.connect_async",
            AsyncMock(side_effect=TimeoutError),
        ):
            with self.assertRaises(DiagnosticDriverError) as raised:
                await self._execute()

        self.assertEqual("TARGET_CONNECTION_TIMEOUT", raised.exception.code)
        self.assertTrue(raised.exception.retryable)

    async def test_query_timeout_has_distinct_error_code(self) -> None:
        with patch(
            "aiops_agent.executor.drivers.oracle.oracledb.connect_async",
            AsyncMock(return_value=_TimeoutConnection()),
        ):
            with self.assertRaises(DiagnosticDriverError) as raised:
                await self._execute()

        self.assertEqual("QUERY_TIMEOUT", raised.exception.code)
        self.assertTrue(raised.exception.retryable)

    async def test_date_format_error_has_specific_query_code(self) -> None:
        with patch(
            "aiops_agent.executor.drivers.oracle.oracledb.connect_async",
            AsyncMock(return_value=_OracleErrorConnection()),
        ):
            with self.assertRaises(DiagnosticDriverError) as raised:
                await self._execute()

        self.assertEqual("QUERY_VALUE_FORMAT_INVALID", raised.exception.code)
        self.assertFalse(raised.exception.retryable)

    async def test_invalid_column_is_not_reported_as_privilege_error(
        self,
    ) -> None:
        with patch(
            "aiops_agent.executor.drivers.oracle.oracledb.connect_async",
            AsyncMock(return_value=_OracleErrorConnection(904)),
        ):
            with self.assertRaises(DiagnosticDriverError) as raised:
                await self._execute()

        self.assertEqual("QUERY_COLUMN_INVALID", raised.exception.code)

    async def test_missing_object_is_not_assumed_to_be_privilege_error(
        self,
    ) -> None:
        with patch(
            "aiops_agent.executor.drivers.oracle.oracledb.connect_async",
            AsyncMock(return_value=_OracleErrorConnection(942)),
        ):
            with self.assertRaises(DiagnosticDriverError) as raised:
                await self._execute()

        self.assertEqual("QUERY_OBJECT_UNAVAILABLE", raised.exception.code)

    async def test_character_lob_is_materialized_before_connection_closes(
        self,
    ) -> None:
        import oracledb

        lob = _FakeOracleLob(
            "SELECT STATEMENT\n  TABLE ACCESS FULL",
            oracledb.DB_TYPE_CLOB,
        )
        with patch(
            "aiops_agent.executor.drivers.oracle.oracledb.connect_async",
            AsyncMock(return_value=_OracleResultConnection(lob)),
        ):
            result = await self._execute()

        self.assertEqual(
            (("SELECT STATEMENT\n  TABLE ACCESS FULL",),),
            result.rows,
        )
        self.assertFalse(result.truncated)
        self.assertEqual((1, 32769), lob.read_amount)

    async def test_oversized_character_lob_is_bounded_and_marked_truncated(
        self,
    ) -> None:
        import oracledb

        lob = _FakeOracleLob(
            "x" * 40000,
            oracledb.DB_TYPE_NCLOB,
        )
        with patch(
            "aiops_agent.executor.drivers.oracle.oracledb.connect_async",
            AsyncMock(return_value=_OracleResultConnection(lob)),
        ):
            result = await self._execute()

        self.assertEqual(32768, len(result.rows[0][0]))
        self.assertTrue(result.truncated)

    async def test_character_lob_uses_cursor_metadata_when_lob_type_is_absent(
        self,
    ) -> None:
        import oracledb

        lob = _FakeOracleLob("SELECT * FROM orders", None)
        with patch(
            "aiops_agent.executor.drivers.oracle.oracledb.connect_async",
            AsyncMock(
                return_value=_OracleResultConnection(
                    lob, oracledb.DB_TYPE_CLOB
                )
            ),
        ):
            result = await self._execute()

        self.assertEqual((("SELECT * FROM orders",),), result.rows)
        self.assertEqual(("DB_TYPE_CLOB",), result.database_types)

    async def test_raw_value_is_rendered_as_bounded_hex_text(self) -> None:
        import oracledb

        with patch(
            "aiops_agent.executor.drivers.oracle.oracledb.connect_async",
            AsyncMock(
                return_value=_OracleResultConnection(
                    bytes.fromhex("DEADBEEF"), oracledb.DB_TYPE_RAW
                )
            ),
        ):
            result = await self._execute()

        self.assertEqual((("DEADBEEF",),), result.rows)
        self.assertEqual(("DB_TYPE_RAW",), result.database_types)

    async def test_binary_lob_is_not_materialized(self) -> None:
        import oracledb

        lob = _FakeOracleLob(b"binary-plan", oracledb.DB_TYPE_BLOB)
        with patch(
            "aiops_agent.executor.drivers.oracle.oracledb.connect_async",
            AsyncMock(
                return_value=_OracleResultConnection(
                    lob, oracledb.DB_TYPE_BLOB
                )
            ),
        ):
            result = await self._execute()

        self.assertIs(lob, result.rows[0][0])
        self.assertIsNone(lob.read_amount)
        self.assertEqual(("DB_TYPE_BLOB",), result.database_types)


class DiagnosticCatalogTest(unittest.TestCase):
    def test_catalog_contains_three_database_parity(self) -> None:
        registry = DiagnosticRegistry.load()
        self.assertEqual(59, len(registry.tools))
        self.assertTrue(
            all(
                column.sensitivity == "PUBLIC"
                for tool in registry.tools
                for column in tool.definition.output_columns
            )
        )
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
            self.assertIn(("POSTGRESQL", tool_id), pairs)
        self.assertIn(("ORACLE", "db.sql.top_current"), pairs)
        self.assertIn(("ORACLE", "db.sql.plan_monitor"), pairs)
        self.assertIn(("ORACLE", "db.sql.cursor_details"), pairs)
        self.assertIn(("ORACLE", "db.sql.execution_plan"), pairs)
        self.assertIn(("ORACLE", "db.sql.display_cursor"), pairs)
        self.assertIn(("ORACLE", "db.sql.object_statistics"), pairs)
        self.assertIn(
            ("ORACLE", "db.resource.session_utilization"), pairs
        )
        for tool_id in (
            "db.instance.parameters",
            "db.storage.temp_usage",
            "db.storage.undo_usage",
            "db.redo.status",
            "db.alert.recent",
            "db.scheduler.failed_jobs",
            "db.objects.invalid_summary",
            "db.backup.recent_jobs",
            "db.replication.lag",
        ):
            self.assertIn(("ORACLE", tool_id), pairs)

    def test_oracle_top_sql_uses_only_v_sqlstats_columns(self) -> None:
        registry = DiagnosticRegistry.load()
        tool = registry.resolve(
            tool_id="db.sql.top_current",
            tool_version="1.0.0",
            db_type="ORACLE",
            db_version="19c",
            capabilities={"dynamic_performance_views"},
            entitlements=set(),
        )

        self.assertNotIn("parsing_schema_name", tool.sql.lower())
        self.assertNotIn("module", tool.sql.lower())
        self.assertNotIn("action", tool.sql.lower())

    def test_oracle_plan_monitor_uses_documented_cardinality_column(self) -> None:
        registry = DiagnosticRegistry.load()
        tool = registry.resolve(
            tool_id="db.sql.plan_monitor",
            tool_version="1.0.0",
            db_type="ORACLE",
            db_version="19c",
            capabilities={"dynamic_performance_views"},
            entitlements=set(),
        )

        self.assertIn("plan_cardinality", tool.sql.lower())
        self.assertNotIn("\n    cardinality,", tool.sql.lower())
        self.assertEqual(
            ("sql_id", "limit"),
            tuple(parameter.name for parameter in tool.definition.parameters),
        )

    def test_oracle_display_cursor_is_fixed_to_allstats_last(self) -> None:
        tool = DiagnosticRegistry.load().resolve(
            tool_id="db.sql.display_cursor",
            tool_version="1.0.0",
            db_type="ORACLE",
            db_version="19c",
            capabilities={"dynamic_performance_views"},
            entitlements=set(),
        )

        self.assertEqual(("DBMS_XPLAN",), tool.definition.allowed_packages)
        self.assertIn("dbms_xplan.display_cursor", tool.sql.lower())
        self.assertIn("'ALLSTATS LAST'", tool.sql)
        self.assertIn(":sql_id", tool.sql)
        self.assertIn("格式化实际执行计划", tool.definition.description)

    def test_oracle_single_sql_baseline_contracts_are_loadable(self) -> None:
        registry = DiagnosticRegistry.load()
        expected = {
            "db.sql.cursor_details": (
                ("sql_id", "limit"),
                {"child_number", "plan_hash_value", "last_active_time"},
            ),
            "db.sql.execution_plan": (
                ("sql_id", "limit"),
                {"plan_line_id", "cardinality", "access_predicates"},
            ),
            "db.sql.object_statistics": (
                ("sql_id",),
                {"last_analyzed", "stale_stats", "stattype_locked"},
            ),
            "db.sql.display_cursor": (
                ("sql_id",),
                {"plan_table_output"},
            ),
        }
        for tool_id, (parameters, columns) in expected.items():
            with self.subTest(tool_id=tool_id):
                tool = registry.resolve(
                    tool_id=tool_id,
                    tool_version="1.0.0",
                    db_type="ORACLE",
                    db_version="19c",
                    capabilities={
                        "dynamic_performance_views",
                        "dba_catalog_views",
                    },
                    entitlements=set(),
                )
                self.assertEqual(
                    parameters,
                    tuple(
                        parameter.name
                        for parameter in tool.definition.parameters
                    ),
                )
                self.assertTrue(
                    columns
                    <= {
                        column.name
                        for column in tool.definition.output_columns
                    }
                )
                self.assertEqual(64, len(tool.definition.template_sha256))

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
            domain_id=100,
            target_row_version=1,
            db_type="ORACLE",
            connection_profile=DiagnosticConnectionProfile(
                host="db.internal",
                port=1521,
                service="KBOT4",
            ),
            diagnostic_credential_id=uuid7(),
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
        control_plane = AsyncMock()
        control_plane.issue_credential.return_value = SimpleNamespace(
            username="private-user", password="hidden",
        )
        service = DiagnosticExecutorService(
            registry=DiagnosticRegistry.load(),
            grant_codec=grants.codec,
            control_plane=control_plane,
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

    def test_alert_log_text_is_returned_without_business_redaction(self) -> None:
        tool = DiagnosticRegistry.load().resolve(
            tool_id="db.alert.recent",
            tool_version="1.0.0",
            db_type="ORACLE",
            db_version="19c",
            capabilities={"dynamic_performance_views"},
            entitlements=set(),
        )
        observed_at = datetime(2026, 9, 1, 1, 33, 53, tzinfo=UTC)
        message = "ORA-00600: internal error code, arguments: [4194]"
        observation = DiagnosticExecutorService._normalize(
            request=SimpleNamespace(executor_request_id=uuid7()),
            grant=SimpleNamespace(
                target_id=uuid7(),
                tool_id=tool.definition.tool_id,
                tool_version=tool.definition.version,
                variant=tool.definition.variant,
                template_sha256=tool.definition.template_sha256,
                db_type="ORACLE",
                capability_snapshot_hash="b" * 64,
                parameters_sha256=canonical_sha256({}),
            ),
            tool=tool,
            raw=DriverQueryResult(
                columns=tuple(
                    item.name for item in tool.definition.output_columns
                ),
                rows=((observed_at, 1, 1, "ORA 600 [4194]", message),),
                truncated=False,
                db_version="19.24.0.0.0",
            ),
            captured_at=observed_at,
            duration_ms=8,
            limits=DiagnosticLimits(
                statement_timeout_seconds=30,
                max_result_rows=100,
                max_result_bytes=1048576,
            ),
        )

        self.assertEqual("ORA 600 [4194]", observation.rows[0][3])
        self.assertEqual(message, observation.rows[0][4])
        self.assertTrue(
            all(column.sensitivity == "PUBLIC" for column in observation.columns)
        )

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
        control_plane = AsyncMock()
        service = DiagnosticExecutorService(
            registry=DiagnosticRegistry.load(),
            grant_codec=grants.codec,
            control_plane=control_plane,
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
        control_plane.issue_credential.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
