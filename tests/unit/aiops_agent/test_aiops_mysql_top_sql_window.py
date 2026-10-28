"""MySQL Top SQL 两次采样、增量与数据质量回归。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.executor.drivers.base import DiagnosticDriverError
from aiops_agent.executor.drivers.mysql import (
    MySQLDiagnosticDriver,
    _mysql_top_sql_delta,
    _TOP_SQL_SAMPLE_COLUMNS,
)
from aiops_agent.ports.secret_store import ResolvedSecret
from platform_core.contracts.aiops.executor import (
    DiagnosticConnectionProfile,
    DiagnosticLimits,
)


START = datetime(2026, 10, 27, 10, 0, 0)
END = START + timedelta(seconds=5)


def sample(digest, at, executions, elapsed, examined=0, sent=0, *, first=None, text=None, schema="app"):
    return (
        10, at, schema, digest, text, "SELECT ?", executions, elapsed,
        examined, sent, first or START - timedelta(hours=1), 0, "YES",
    )


def empty(at):
    return (10, at, None, None, None, None, None, None, None, None, None, 0, "YES")


class MySQLTopSqlDeltaTest(unittest.TestCase):
    def test_orders_by_window_delta_and_uses_sample_or_digest(self):
        before = [
            sample("old_total", START, 1000, 100_000_000_000_000, 100, 10),
            sample("hot_now", START, 2, 10_000_000_000, 5, 1),
        ]
        after = [
            sample("old_total", END, 1001, 100_100_000_000_000, 105, 12),
            sample("hot_now", END, 6, 2_010_000_000_000, 25, 9, text="SELECT 42"),
        ]
        rows, incomplete = _mysql_top_sql_delta(before, after, sample_limit=2001, result_limit=10)
        self.assertFalse(incomplete)
        self.assertEqual(["hot_now", "old_total"], [row[1] for row in rows])
        self.assertEqual(("SELECT 42", "SELECT ?", 4, Decimal("2"), 20, 8, Decimal("5.0")), rows[0][2:])
        self.assertEqual("SELECT ?", rows[1][2])
        limited, _ = _mysql_top_sql_delta(before, after, sample_limit=2001, result_limit=1)
        self.assertEqual(1, len(limited))

    def test_new_digest_and_empty_baseline(self):
        first = START + timedelta(seconds=1)
        rows, incomplete = _mysql_top_sql_delta(
            [empty(START)], [sample("new", END, 3, 3_000_000_000_000, first=first)],
            sample_limit=2001, result_limit=10,
        )
        self.assertFalse(incomplete)
        self.assertEqual(("new", "SELECT ?", "SELECT ?", 3, Decimal("3")), rows[0][1:6])

    def test_reset_missing_and_unknown_baseline_are_not_fabricated(self):
        before = [
            sample("reset", START, 10, 100),
            sample("evicted", START, 10, 100),
            sample("changed_first_seen", START, 10, 100),
        ]
        after = [
            sample("reset", END, 2, 5),
            sample("unknown", END, 20, 200),
            sample("changed_first_seen", END, 30, 300, first=START + timedelta(seconds=1)),
            sample("new", END, 1, 500, first=START + timedelta(seconds=1)),
        ]
        rows, incomplete = _mysql_top_sql_delta(before, after, sample_limit=2001, result_limit=10)
        self.assertTrue(incomplete)
        self.assertEqual(["new"], [row[1] for row in rows])

    def test_sampling_bound_is_marked_incomplete(self):
        before = [sample("a", START, 1, 1), sample("b", START, 1, 1), sample("c", START, 1, 1)]
        after = [sample("a", END, 2, 2), sample("b", END, 2, 2), sample("c", END, 2, 2)]
        rows, incomplete = _mysql_top_sql_delta(before, after, sample_limit=3, result_limit=10)
        self.assertTrue(incomplete)
        self.assertEqual(["a", "b"], [row[1] for row in rows])

    def test_overflow_is_marked_incomplete_and_disabled_digest_is_a_gap(self):
        before = [sample("a", START, 1, 1)]
        after_row = list(sample("a", END, 2, 2))
        after_row[11] = 1
        rows, incomplete = _mysql_top_sql_delta(before, [tuple(after_row)], sample_limit=2001, result_limit=10)
        self.assertTrue(incomplete)
        self.assertEqual(["a"], [row[1] for row in rows])
        after_row[12] = "NO"
        with self.assertRaisesRegex(DiagnosticDriverError, "CAPABILITY_UNAVAILABLE"):
            _mysql_top_sql_delta(before, [tuple(after_row)], sample_limit=2001, result_limit=10)


class _Cursor:
    def __init__(self):
        self.description = tuple((name,) for name in _TOP_SQL_SAMPLE_COLUMNS)
        self.statements = []
        self.samples = [
            [sample("digest", START, 7, 2_000_000_000_000)],
            [sample("digest", END, 9, 5_000_000_000_000, text="SELECT 1")],
        ]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def execute(self, statement, parameters=None):
        self.statements.append((statement, parameters))

    async def fetchmany(self, limit):
        assert limit == 2001
        return self.samples.pop(0)


class MySQLTopSqlDriverTest(unittest.TestCase):
    def test_driver_collects_twice_and_returns_catalog_schema(self):
        registry = DiagnosticRegistry.load()
        tool = registry.resolve(
            tool_id="db.sql.top_current", tool_version="1.0.0", db_type="MYSQL",
            db_version="8.4", capabilities={"statement_digest"}, entitlements=set(),
        )
        cursor = _Cursor()
        connection = Mock()
        connection.cursor.return_value = cursor
        connection.rollback = AsyncMock()
        connection.get_server_info.return_value = "8.4"
        with patch("aiops_agent.executor.drivers.mysql.aiomysql.connect", new=AsyncMock(return_value=connection)), patch(
            "aiops_agent.executor.drivers.mysql.OwnedMySQLExecution.pause", new=AsyncMock()
        ) as sleep:
            result = asyncio.run(MySQLDiagnosticDriver().execute(
                profile=DiagnosticConnectionProfile(host="localhost", port=3306, tls_enabled=False),
                secret=ResolvedSecret(values={"username": "test", "password": "test"}, fingerprint="test"),
                tool=tool, parameters=registry.validate_parameters(tool, {"limit": 10}),
                limits=DiagnosticLimits(statement_timeout_seconds=20, max_result_rows=200, max_result_bytes=1048576),
                trace_id="test",
            ))
        self.assertEqual(tuple(column.name for column in tool.definition.output_columns), result.columns)
        self.assertEqual(2, result.rows[0][4])
        self.assertEqual(Decimal("3"), result.rows[0][5])
        self.assertEqual(2, sum("events_statements_summary_by_digest" in sql for sql, _ in cursor.statements))
        sleep.assert_awaited_once_with(5)
        connection.rollback.assert_awaited_once()
        connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
