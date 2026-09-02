"""9C3 隔离 Executor 单次变更闸门测试。"""

from __future__ import annotations

import asyncio
import hashlib
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiops_agent.actions import (
    ActionRegistry,
    ActionRenderer,
    MutationGrantCodec,
)
from aiops_agent.executor import (
    MutationExecutionError,
    MutationExecutorService,
)
from aiops_agent.executor.drivers import (
    MutationDriverError,
    MutationDriverResult,
    OracleMutationDriver,
)
from aiops_agent.ports.secret_store import ResolvedSecret
from platform_core.contracts.aiops.executor import (
    DiagnosticConnectionProfile,
    MutationClaimReceipt,
    MutationExecutionGrant,
    MutationExecutionRequest,
)
from platform_core.contracts.aiops.internal import EventReceipt
from platform_core.identity import uuid7


class _ControlPlane:
    def __init__(self, receipt, *, reject_running: bool = False):
        self.receipt = receipt
        self.reject_running = reject_running
        self.events = []

    async def claim_execution(self, execution_id, request, *, trace_id):
        del execution_id, request, trace_id
        return self.receipt

    async def publish_event(self, event, *, trace_id):
        del trace_id
        self.events.append(event)
        accepted = not (
            self.reject_running and event.status == "RUNNING"
        )
        return EventReceipt(
            event_id=event.event_id,
            accepted=accepted,
        )

    async def issue_credential(self, grant, *, trace_id):
        del grant, trace_id
        return SimpleNamespace(username="ops", password="secret")


class _Driver:
    db_type = "MYSQL"

    def __init__(self, error: MutationDriverError | None = None):
        self.error = error
        self.calls = 0

    async def execute_action(self, **kwargs):
        del kwargs
        self.calls += 1
        if self.error is not None:
            raise self.error
        return MutationDriverResult(
            bounded_result={
                "accepted": True,
                "action_template_id": "db.session.terminate",
                "affected_object_count": 1,
            }
        )


class MutationExecutorServiceTest(unittest.TestCase):
    def _fixture(self, *, driver=None, reject_running=False):
        registry = ActionRegistry.load()
        template = registry.resolve(
            action_template_id="db.session.terminate",
            version="1.0.0",
            db_type="MYSQL",
            db_version="8.0.36",
            capabilities={"session_management"},
            entitlements=set(),
            environment="PROD",
        )
        action = ActionRenderer().render(
            template, {"session_id": 42}
        )
        codec = MutationGrantCodec(
            secret="mutation-grant-test-secret-at-least-32-bytes",
            issuer="kbot-aiops-api",
            audience="kbot-aiops-db-executor",
        )
        execution_id = uuid7()
        request_id = uuid7()
        now = datetime.now(UTC)
        grant = MutationExecutionGrant(
            issuer="kbot-aiops-api",
            audience="kbot-aiops-db-executor",
            grant_id=execution_id,
            issued_at=now,
            expires_at=now + timedelta(seconds=30),
            execution_id=execution_id,
            executor_request_id=request_id,
            executor_instance_id="executor-test",
            target_id=uuid7(),
            domain_id=100,
            target_version=1,
            db_type="MYSQL",
            connection_profile={
                "host": "mysql.internal",
                "port": 3306,
                "database": "ops",
                "tls_enabled": True,
            },
            execution_credential_id=uuid7(),
            action_template_id=action.action_template_id,
            action_template_version=action.action_template_version,
            action_template_variant=action.variant,
            renderer_version=action.renderer_version,
            typed_parameters=action.typed_parameters,
            action_template_hash=action.template_hash,
            parameters_hash=action.parameters_hash,
            command_hash=action.command_hash,
            proposal_hash="a" * 64,
            policy_decision_hash="b" * 64,
            approval_token_hash="c" * 64,
            approver_id="portal:user-1",
            action_catalog_hash=registry.catalog_hash,
            statement_timeout_seconds=60,
            trace_id="trace-source",
        )
        receipt = MutationClaimReceipt(
            execution_id=execution_id,
            executor_request_id=request_id,
            status="SUBMITTED",
            grant=codec.issue(grant),
            expires_at=grant.expires_at,
        )
        control = _ControlPlane(
            receipt, reject_running=reject_running
        )
        resolved_driver = driver or _Driver()
        service = MutationExecutorService(
            enabled=True,
            executor_instance_id="executor-test",
            registry=registry,
            grant_codec=codec,
            control_plane=control,
            drivers=(resolved_driver,),
            concurrency=1,
        )
        request = MutationExecutionRequest(
            execution_id=execution_id,
            executor_request_id=request_id,
            idempotency_key=f"execution:{execution_id}:dispatch",
        )
        return service, request, control, resolved_driver

    def test_running_is_persisted_before_exactly_one_database_call(
        self,
    ) -> None:
        service, request, control, driver = self._fixture()
        result = asyncio.run(
            service.execute(request, trace_id="trace-dispatch")
        )
        self.assertEqual(result.status, "SUCCEEDED")
        self.assertEqual(driver.calls, 1)
        self.assertEqual(
            [event.status for event in control.events],
            ["RUNNING", "SUCCEEDED"],
        )
        self.assertEqual(
            [event.status_version for event in control.events],
            [3, 4],
        )

    def test_rejected_running_event_prevents_database_call(self) -> None:
        service, request, _, driver = self._fixture(
            reject_running=True
        )
        with self.assertRaises(MutationExecutionError) as caught:
            asyncio.run(
                service.execute(request, trace_id="trace-dispatch")
            )
        self.assertEqual(caught.exception.code, "RUNNING_EVENT_REJECTED")
        self.assertEqual(driver.calls, 0)

    def test_uncertain_driver_outcome_is_reported_as_unknown(self) -> None:
        driver = _Driver(
            MutationDriverError(
                "EXECUTION_OUTCOME_UNKNOWN",
                outcome_unknown=True,
            )
        )
        service, request, control, _ = self._fixture(driver=driver)
        result = asyncio.run(
            service.execute(request, trace_id="trace-dispatch")
        )
        self.assertEqual(result.status, "UNKNOWN")
        terminal = control.events[-1]
        self.assertEqual(terminal.status, "UNKNOWN")
        self.assertEqual(
            terminal.result_hash,
            hashlib.sha256(
                b'{"accepted":false,"action_template_id":'
                b'"db.session.terminate","outcome_unknown":true}'
            ).hexdigest(),
        )


class _OracleMutationCursor:
    def __init__(self, *, present=True, execution_error=False):
        self.present = present
        self.execution_error = execution_error
        self.calls = []

    async def execute(self, sql, parameters=None):
        self.calls.append((" ".join(sql.split()), parameters))
        if self.execution_error and str(sql).startswith("ALTER INDEX"):
            raise TimeoutError

    async def fetchone(self):
        return (1,) if self.present else None

    def close(self):
        return None


class _OracleMutationConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.call_timeout = 0
        self.module = ""
        self.action = ""

    def cursor(self):
        return self._cursor

    async def close(self):
        return None


class OracleIndexMutationDriverTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        registry = ActionRegistry.load()
        template = registry.resolve(
            action_template_id="db.index.rebuild",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dba_catalog_views", "index_maintenance"},
            entitlements=set(),
            environment="PROD",
        )
        self.action = ActionRenderer().render(
            template,
            {
                "index_ref": {
                    "schema": "APP",
                    "object_type": "INDEX",
                    "object_name": "IX_ORDERS",
                },
                "online": True,
            },
        )
        self.profile = DiagnosticConnectionProfile(
            host="db.internal",
            port=1521,
            service="PDB1",
            tls_enabled=False,
        )
        self.secret = ResolvedSecret(
            values={"username": "ops", "password": "hidden"},
            fingerprint="test",
        )

    async def _execute(self, cursor):
        with patch(
            "aiops_agent.executor.drivers.oracle.oracledb.connect_async",
            AsyncMock(return_value=_OracleMutationConnection(cursor)),
        ):
            return await OracleMutationDriver().execute_action(
                profile=self.profile,
                secret=self.secret,
                action=self.action,
                trace_id="trace-index",
            )

    async def test_rechecks_exact_index_before_rebuild(self):
        cursor = _OracleMutationCursor()
        result = await self._execute(cursor)

        self.assertTrue(result.bounded_result["accepted"])
        self.assertIn("FROM DBA_INDEXES", cursor.calls[0][0])
        self.assertEqual(
            {"schema_name": "APP", "index_name": "IX_ORDERS"},
            cursor.calls[0][1],
        )
        self.assertEqual(
            'ALTER INDEX "APP"."IX_ORDERS" REBUILD ONLINE',
            cursor.calls[1][0],
        )

    async def test_rechecks_exact_partition_before_partition_rebuild(self):
        template = ActionRegistry.load().resolve(
            action_template_id="db.index.partition.rebuild",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dba_catalog_views", "index_maintenance"},
            entitlements=set(),
            environment="PROD",
        )
        self.action = ActionRenderer().render(
            template,
            {
                "index_ref": {
                    "schema": "APP",
                    "object_type": "INDEX",
                    "object_name": "IX_ORDERS",
                    "partition": "P_202609",
                },
                "partition_name": "P_202609",
                "online": True,
            },
        )
        cursor = _OracleMutationCursor()
        result = await self._execute(cursor)

        self.assertTrue(result.bounded_result["accepted"])
        self.assertIn("FROM DBA_IND_PARTITIONS", cursor.calls[0][0])
        self.assertEqual(
            {
                "schema_name": "APP",
                "index_name": "IX_ORDERS",
                "partition_name": "P_202609",
            },
            cursor.calls[0][1],
        )
        self.assertEqual(
            'ALTER INDEX "APP"."IX_ORDERS" REBUILD PARTITION "P_202609" ONLINE',
            cursor.calls[1][0],
        )

    async def test_rechecks_exact_sql_before_cancelling(self):
        template = ActionRegistry.load().resolve(
            action_template_id="db.session.cancel_sql",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dynamic_performance_views", "session_management"},
            entitlements=set(),
            environment="PROD",
        )
        self.action = ActionRenderer().render(
            template,
            {
                "session_id": 42,
                "serial_number": 9,
                "instance_id": 1,
                "sql_id": "0abc123def456",
            },
        )
        cursor = _OracleMutationCursor()
        result = await self._execute(cursor)

        self.assertTrue(result.bounded_result["accepted"])
        self.assertIn("SQL_ID = :sql_id", cursor.calls[0][0])
        self.assertEqual(
            {
                "instance_id": 1,
                "session_id": 42,
                "serial_number": 9,
                "sql_id": "0abc123def456",
            },
            cursor.calls[0][1],
        )
        self.assertEqual(
            "ALTER SYSTEM CANCEL SQL '42,9,@1,0abc123def456' IMMEDIATE",
            cursor.calls[1][0],
        )

    async def test_rechecks_exact_invalid_object_before_compile(self):
        template = ActionRegistry.load().resolve(
            action_template_id="db.object.compile",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dba_catalog_views"},
            entitlements=set(),
            environment="PROD",
        )
        self.action = ActionRenderer().render(
            template,
            {
                "object_type": "PACKAGE",
                "object_ref": {
                    "schema": "APP",
                    "object_type": "PACKAGE",
                    "object_name": "PKG_ORDERS",
                },
            },
        )
        cursor = _OracleMutationCursor()

        result = await self._execute(cursor)

        self.assertTrue(result.bounded_result["accepted"])
        self.assertIn("FROM DBA_OBJECTS", cursor.calls[0][0])
        self.assertIn("STATUS = 'INVALID'", cursor.calls[0][0])
        self.assertEqual(
            {
                "schema_name": "APP",
                "object_name": "PKG_ORDERS",
                "object_type": "PACKAGE",
            },
            cursor.calls[0][1],
        )
        self.assertEqual(
            'ALTER PACKAGE "APP"."PKG_ORDERS" COMPILE',
            cursor.calls[1][0],
        )

    async def test_rechecks_stale_unlocked_table_before_gathering_stats(self):
        template = ActionRegistry.load().resolve(
            action_template_id="db.statistics.gather",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dba_catalog_views"},
            entitlements=set(),
            environment="PROD",
        )
        self.action = ActionRenderer().render(
            template,
            {
                "table_ref": {
                    "schema": "APP",
                    "object_type": "TABLE",
                    "object_name": "ORDERS",
                }
            },
        )
        cursor = _OracleMutationCursor()

        result = await self._execute(cursor)

        self.assertTrue(result.bounded_result["accepted"])
        self.assertIn("FROM DBA_TABLES", cursor.calls[0][0])
        self.assertIn("s.STATTYPE_LOCKED IS NULL", cursor.calls[0][0])
        self.assertIn("s.STALE_STATS = 'YES'", cursor.calls[0][0])
        self.assertEqual(
            {"schema_name": "APP", "table_name": "ORDERS"},
            cursor.calls[0][1],
        )
        self.assertTrue(
            cursor.calls[1][0].startswith(
                "BEGIN DBMS_STATS.GATHER_TABLE_STATS("
            )
        )

    async def test_rechecks_statistics_lock_state_before_change(self):
        cases = (
            ("db.statistics.lock", 0, "LOCK_TABLE_STATS"),
            ("db.statistics.unlock", 1, "UNLOCK_TABLE_STATS"),
        )
        for action_id, require_locked, operation in cases:
            with self.subTest(action_id=action_id):
                template = ActionRegistry.load().resolve(
                    action_template_id=action_id,
                    version="1.0.0",
                    db_type="ORACLE",
                    db_version="19.0.0",
                    capabilities={"dba_catalog_views"},
                    entitlements=set(),
                    environment="PROD",
                )
                self.action = ActionRenderer().render(
                    template,
                    {
                        "table_ref": {
                            "schema": "APP",
                            "object_type": "TABLE",
                            "object_name": "ORDERS",
                        }
                    },
                )
                cursor = _OracleMutationCursor()

                result = await self._execute(cursor)

                self.assertTrue(result.bounded_result["accepted"])
                self.assertEqual(
                    {
                        "schema_name": "APP",
                        "table_name": "ORDERS",
                        "require_locked": require_locked,
                    },
                    cursor.calls[0][1],
                )
                self.assertIn(operation, cursor.calls[1][0])

    async def test_rechecks_job_state_and_counts_before_running(self):
        template = ActionRegistry.load().resolve(
            action_template_id="db.scheduler.job.run",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dba_catalog_views"},
            entitlements=set(),
            environment="PROD",
        )
        self.action = ActionRenderer().render(
            template,
            {
                "job_ref": {
                    "schema": "APP",
                    "object_type": "SCHEDULER_JOB",
                    "object_name": "NIGHTLY_JOB",
                },
                "previous_run_count": 7,
                "previous_failure_count": 1,
            },
        )
        cursor = _OracleMutationCursor()

        result = await self._execute(cursor)

        self.assertTrue(result.bounded_result["accepted"])
        self.assertIn("FROM DBA_SCHEDULER_JOBS", cursor.calls[0][0])
        self.assertIn("STATE = 'SCHEDULED'", cursor.calls[0][0])
        self.assertEqual(
            {
                "schema_name": "APP",
                "job_name": "NIGHTLY_JOB",
                "previous_run_count": 7,
                "previous_failure_count": 1,
            },
            cursor.calls[0][1],
        )
        self.assertEqual(
            "BEGIN DBMS_SCHEDULER.RUN_JOB(job_name => "
            "'\"APP\".\"NIGHTLY_JOB\"', use_current_session => FALSE); END;",
            cursor.calls[1][0],
        )

    async def test_rechecks_scheduler_state_before_state_change(self):
        cases = (
            ("db.scheduler.job.enable", "FALSE", "DISABLED"),
            ("db.scheduler.job.disable", "TRUE", "SCHEDULED"),
            ("db.scheduler.job.stop", "TRUE", "RUNNING"),
        )
        for action_id, enabled, state in cases:
            with self.subTest(action_id=action_id):
                template = ActionRegistry.load().resolve(
                    action_template_id=action_id,
                    version="1.0.0",
                    db_type="ORACLE",
                    db_version="19.0.0",
                    capabilities={"dba_catalog_views"},
                    entitlements=set(),
                    environment="PROD",
                )
                self.action = ActionRenderer().render(
                    template,
                    {
                        "job_ref": {
                            "schema": "APP",
                            "object_type": "SCHEDULER_JOB",
                            "object_name": "NIGHTLY_JOB",
                        }
                    },
                )
                cursor = _OracleMutationCursor()

                result = await self._execute(cursor)

                self.assertTrue(result.bounded_result["accepted"])
                self.assertEqual(
                    {
                        "schema_name": "APP",
                        "job_name": "NIGHTLY_JOB",
                        "expected_enabled": enabled,
                        "expected_state": state,
                    },
                    cursor.calls[0][1],
                )
                self.assertIn("DBMS_SCHEDULER", cursor.calls[1][0])

    async def test_rechecks_local_application_user_before_state_change(self):
        cases = (
            ("db.user.lock", 0, "ACCOUNT LOCK"),
            ("db.user.unlock", 1, "ACCOUNT UNLOCK"),
        )
        for action_id, require_locked, operation in cases:
            with self.subTest(action_id=action_id):
                template = ActionRegistry.load().resolve(
                    action_template_id=action_id,
                    version="1.0.0",
                    db_type="ORACLE",
                    db_version="19.0.0",
                    capabilities={"dba_catalog_views"},
                    entitlements=set(),
                    environment="PROD",
                )
                self.action = ActionRenderer().render(
                    template,
                    {
                        "user_ref": {
                            "schema": "APPUSER",
                            "object_type": "USER",
                            "object_name": "APPUSER",
                        }
                    },
                )
                cursor = _OracleMutationCursor()

                result = await self._execute(cursor)

                self.assertTrue(result.bounded_result["accepted"])
                self.assertIn("FROM DBA_USERS", cursor.calls[0][0])
                self.assertIn("ORACLE_MAINTAINED = 'N'", cursor.calls[0][0])
                self.assertEqual(
                    {
                        "username": "APPUSER",
                        "require_locked": require_locked,
                    },
                    cursor.calls[0][1],
                )
                self.assertIn(operation, cursor.calls[1][0])

    async def test_rechecks_unexpired_user_before_password_expiry(self):
        template = ActionRegistry.load().resolve(
            action_template_id="db.user.password.expire",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dba_catalog_views"},
            entitlements=set(),
            environment="PROD",
        )
        self.action = ActionRenderer().render(
            template,
            {
                "user_ref": {
                    "schema": "APPUSER",
                    "object_type": "USER",
                    "object_name": "APPUSER",
                }
            },
        )
        cursor = _OracleMutationCursor()

        result = await self._execute(cursor)

        self.assertTrue(result.bounded_result["accepted"])
        self.assertIn("ACCOUNT_STATUS NOT LIKE '%EXPIRED%'", cursor.calls[0][0])
        self.assertEqual({"username": "APPUSER"}, cursor.calls[0][1])
        self.assertIn("PASSWORD EXPIRE", cursor.calls[1][0])

    async def test_missing_index_rejects_before_mutation(self):
        cursor = _OracleMutationCursor(present=False)
        with self.assertRaises(MutationDriverError) as raised:
            await self._execute(cursor)

        self.assertEqual("PRECONDITION_CHANGED", raised.exception.code)
        self.assertEqual(1, len(cursor.calls))

    async def test_timeout_after_rebuild_starts_is_unknown(self):
        cursor = _OracleMutationCursor(execution_error=True)
        with self.assertRaises(MutationDriverError) as raised:
            await self._execute(cursor)

        self.assertEqual("EXECUTION_OUTCOME_UNKNOWN", raised.exception.code)
        self.assertTrue(raised.exception.outcome_unknown)


if __name__ == "__main__":
    unittest.main()
