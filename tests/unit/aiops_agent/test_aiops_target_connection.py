"""AIOps Target 创建前连接测试。"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from aiops_agent.application.configuration.connection_test import (
    test_target_connection as run_connection_test,
)
from platform_core.contracts.aiops import TargetConnectionTest


def _request(db_type: str) -> TargetConnectionTest:
    endpoint = {
        "host": "db.internal",
        "port": {"ORACLE": 1521, "MYSQL": 3306, "POSTGRESQL": 5432}[db_type],
        "tls_enabled": False,
    }
    endpoint["service" if db_type == "ORACLE" else "database"] = (
        "ORCLPDB1" if db_type == "ORACLE" else "app"
    )
    return TargetConnectionTest.model_validate({
        "db_type": db_type,
        "endpoint": endpoint,
        "diagnostic_credential": {
            "username": "diag",
            "password": "secret",
        },
    })


class _OracleCursor:
    async def execute(self, _sql):
        return None

    async def fetchone(self):
        return (1,)

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

    async def test_timeout_returns_stable_error_without_exception_detail(self):
        with patch(
            "aiops_agent.application.configuration.connection_test."
            "oracledb.connect_async",
            AsyncMock(side_effect=TimeoutError),
        ):
            result = await run_connection_test(_request("ORACLE"))

        self.assertFalse(result.ok)
        self.assertEqual("TIMEOUT", result.error_code)
        self.assertNotIn("secret", result.model_dump_json())

    def test_oracle_rejects_database_name(self):
        payload = _request("ORACLE").model_dump(mode="json")
        payload["endpoint"]["database"] = "ORCL"

        with self.assertRaises(ValidationError):
            TargetConnectionTest.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
