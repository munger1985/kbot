"""AIOps Target 创建前连接测试。"""

import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from aiops_agent.application.configuration.connection_test import (
    test_target_connection as run_connection_test,
)
from aiops_agent.application.targets.connectivity_check import (
    TargetConnectivityCheckService,
)
from platform_core.contracts.aiops import (
    TargetConnectionTest,
    TargetConnectionTestResult,
)
from platform_core.identity import uuid7


def _request(db_type: str) -> TargetConnectionTest:
    endpoint = {
        "host": "db.internal",
        "port": {"ORACLE": 1521, "MYSQL": 3306, "POSTGRESQL": 5432}[db_type],
        "tls_enabled": False,
    }
    endpoint["service" if db_type == "ORACLE" else "database"] = (
        "ORCLPDB1" if db_type == "ORACLE" else "app"
    )
    payload = {
        "db_type": db_type,
        "endpoint": endpoint,
        "diagnostic_credential": {
            "username": "diag",
            "password": "secret",
        },
    }
    if db_type == "ORACLE":
        payload.update(
            oracle_container_scope="PDB",
            oracle_pdb_name="ORCLPDB1",
        )
    return TargetConnectionTest.model_validate(payload)


class _OracleCursor:
    async def execute(self, _sql):
        return None

    async def fetchone(self):
        return ("ORCLPDB1", 3, "YES", "ORCLCDB")

    def close(self):
        return None


class _OracleConnection:
    version = "19.24.0.0.0"
    call_timeout = 0

    def cursor(self):
        return _OracleCursor()

    async def close(self):
        return None


class _MySQLConnection:
    def cursor(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def execute(self, _sql):
        return None

    async def fetchone(self):
        return (1,)

    def get_server_info(self):
        return "8.4.0"

    def close(self):
        return None


class _PostgreSQLConnection:
    def __init__(self):
        self.calls = 0

    async def fetchval(self, _sql):
        self.calls += 1
        return 1 if self.calls == 1 else "16.4"

    async def close(self):
        return None


class _ConnectivityUow:
    def __init__(self) -> None:
        self.target_id = uuid7()
        self.request_id = uuid7()
        self.target = SimpleNamespace(
            domain_id=7,
            target_id=self.target_id,
            db_type="ORACLE",
            oracle_container_scope="PDB",
            oracle_pdb_name="ORCLPDB1",
            endpoint_json={
                "host": "db.internal",
                "port": 1521,
                "service": "ORCLPDB1",
            },
            diagnostic_credential_id=uuid7(),
            row_version=4,
            connectivity_version=2,
            connectivity_check_request_id=self.request_id,
        )
        self.update = None
        self.committed = False
        self.targets = SimpleNamespace(
            get_scoped=self._get_scoped,
            update_connectivity=self._update_connectivity,
        )
        self.runs = SimpleNamespace(database_now=self._database_now)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def _get_scoped(self, **_kwargs):
        return self.target

    async def _update_connectivity(self, **kwargs):
        self.update = kwargs
        return True

    async def _database_now(self):
        return datetime(2026, 9, 11, tzinfo=UTC)

    async def commit(self):
        self.committed = True


class AIOpsTargetConnectionTest(unittest.IsolatedAsyncioTestCase):
    async def test_three_database_types_execute_minimal_connection(self):
        cases = (
            ("ORACLE", "oracledb.connect_async", _OracleConnection(), "19.24.0.0.0"),
            ("MYSQL", "aiomysql.connect", _MySQLConnection(), "8.4.0"),
            (
                "POSTGRESQL",
                "asyncpg.connect",
                _PostgreSQLConnection(),
                "16.4",
            ),
        )
        for db_type, target, connection, expected_version in cases:
            with self.subTest(db_type=db_type), patch(
                f"aiops_agent.application.configuration.connection_test.{target}",
                AsyncMock(return_value=connection),
            ):
                result = await asyncio.wait_for(
                    run_connection_test(_request(db_type)), timeout=1
                )
            self.assertTrue(result.ok)
            self.assertEqual(expected_version, result.database_version)
            self.assertIsNone(result.error_code)
            if db_type == "ORACLE":
                self.assertEqual("PDB", result.oracle_container_scope)
                self.assertEqual("ORCLPDB1", result.oracle_container_name)
                self.assertEqual(3, result.oracle_container_number)
                self.assertEqual("ORCLCDB", result.oracle_database_name)

    async def test_timeout_returns_stable_error_without_exception_detail(self):
        with patch(
            "aiops_agent.application.configuration.connection_test."
            "oracledb.connect_async",
            AsyncMock(side_effect=TimeoutError),
        ), patch(
            "aiops_agent.application.configuration.connection_test."
            "logger.warning"
        ) as warning:
            result = await run_connection_test(_request("ORACLE"))

        self.assertFalse(result.ok)
        self.assertEqual("TIMEOUT", result.error_code)
        self.assertNotIn("secret", result.model_dump_json())
        logged = " ".join(str(item) for item in warning.call_args.args)
        self.assertIn("TIMEOUT", logged)
        self.assertIn("操作超时", logged)
        self.assertNotIn("secret", logged)
        self.assertNotIn("diag", logged)

    async def test_connection_failure_logs_safe_driver_reason(self):
        failure = ConnectionRefusedError(
            111, "diag secret listener refused"
        )
        with patch(
            "aiops_agent.application.configuration.connection_test."
            "oracledb.connect_async",
            AsyncMock(side_effect=failure),
        ), patch(
            "aiops_agent.application.configuration.connection_test."
            "logger.warning"
        ) as warning:
            result = await run_connection_test(_request("ORACLE"))

        self.assertFalse(result.ok)
        self.assertEqual("TARGET_UNREACHABLE", result.error_code)
        logged = " ".join(str(item) for item in warning.call_args.args)
        self.assertIn("ConnectionRefusedError", logged)
        self.assertIn("111", logged)
        self.assertIn("listener refused", logged)
        self.assertNotIn("secret", logged)
        self.assertNotIn("diag", logged)

    def test_oracle_rejects_database_name(self):
        payload = _request("ORACLE").model_dump(mode="json")
        payload["endpoint"]["database"] = "ORCL"

        with self.assertRaises(ValidationError):
            TargetConnectionTest.model_validate(payload)

    async def test_oracle_rejects_connected_container_mismatch(self):
        request = _request("ORACLE").model_copy(
            update={"oracle_pdb_name": "EXPECTED_PDB"}
        )
        with patch(
            "aiops_agent.application.configuration.connection_test."
            "oracledb.connect_async",
            AsyncMock(return_value=_OracleConnection()),
        ):
            result = await run_connection_test(request)

        self.assertFalse(result.ok)
        self.assertEqual("ORACLE_CONTAINER_MISMATCH", result.error_code)
        self.assertEqual("ORCLPDB1", result.oracle_container_name)

    async def test_connectivity_check_persists_oracle_container_observation(self):
        uow = _ConnectivityUow()
        service = TargetConnectivityCheckService(
            uow_factory=lambda: uow,
            managed_credentials=SimpleNamespace(
                read=AsyncMock(
                    return_value={"username": "diag", "password": "secret"}
                )
            ),
        )
        observed = TargetConnectionTestResult(
            ok=True,
            database_version="19.24.0.0.0",
            oracle_container_scope="PDB",
            oracle_container_name="ORCLPDB1",
            oracle_container_number=3,
            oracle_database_name="ORCLCDB",
        )
        with patch(
            "aiops_agent.application.targets.connectivity_check."
            "test_target_connection",
            AsyncMock(return_value=observed),
        ):
            await service.execute(
                {
                    "domain_id": str(uow.target.domain_id),
                    "aggregate_id": str(uow.target_id),
                    "details": {
                        "connectivity_check_request_id": str(uow.request_id)
                    },
                }
            )

        self.assertEqual("CONNECTED", uow.update["connectivity_status"])
        self.assertEqual(
            {
                "observed_oracle_container_scope": "PDB",
                "observed_oracle_container_name": "ORCLPDB1",
                "observed_oracle_container_number": 3,
                "observed_oracle_database_name": "ORCLCDB",
            },
            uow.update["oracle_observation"],
        )
        self.assertTrue(uow.committed)

    async def test_connectivity_check_marks_container_mismatch_misconfigured(self):
        uow = _ConnectivityUow()
        service = TargetConnectivityCheckService(
            uow_factory=lambda: uow,
            managed_credentials=SimpleNamespace(
                read=AsyncMock(
                    return_value={"username": "diag", "password": "secret"}
                )
            ),
        )
        mismatch = TargetConnectionTestResult(
            ok=False,
            database_version="19.24.0.0.0",
            oracle_container_scope="CDB_ROOT",
            oracle_container_name="CDB$ROOT",
            oracle_container_number=1,
            oracle_database_name="ORCLCDB",
            error_code="ORACLE_CONTAINER_MISMATCH",
        )
        with patch(
            "aiops_agent.application.targets.connectivity_check."
            "test_target_connection",
            AsyncMock(return_value=mismatch),
        ):
            await service.execute(
                {
                    "domain_id": str(uow.target.domain_id),
                    "aggregate_id": str(uow.target_id),
                    "details": {
                        "connectivity_check_request_id": str(uow.request_id)
                    },
                }
            )

        self.assertEqual("MISCONFIGURED", uow.update["connectivity_status"])
        self.assertEqual(
            "CDB_ROOT",
            uow.update["oracle_observation"][
                "observed_oracle_container_scope"
            ],
        )
        self.assertEqual(
            "ORACLE_CONTAINER_MISMATCH", uow.update["last_error_code"]
        )


if __name__ == "__main__":
    unittest.main()
