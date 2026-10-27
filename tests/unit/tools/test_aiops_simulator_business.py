"""多业务流、事务边界和 Oracle 连接池限制的回归测试。"""

from __future__ import annotations

import asyncio
import random
import sys
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "aiops_simulator"))

from aiops_simulator.adapters.mysql import MySQLAdapter  # noqa: E402
from aiops_simulator.adapters.oracle import OracleAdapter, _resource_limit_seconds  # noqa: E402
from aiops_simulator.adapters.postgresql import PostgreSQLAdapter  # noqa: E402
from aiops_simulator.engine import DatabaseRunner, _driver_error_code  # noqa: E402
from aiops_simulator.modes.base import OperationKind  # noqa: E402


class Context:
    def __init__(self, value):
        self.value = value
        self.exit_args = None

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *args):
        self.exit_args = args
        return False


class SimulatorBusinessTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = SimpleNamespace(statement_timeout_seconds=10, historical_year=2025, timezone=ZoneInfo("Asia/Shanghai"))
        self.database = SimpleNamespace(schema="mes", traffic=SimpleNamespace(max_inflight=4))
        self.now = datetime(2026, 10, 27, 14, 0, tzinfo=self.config.timezone)
        self.run_id = str(uuid.uuid4())

    def oracle(self):
        adapter = OracleAdapter(self.database, self.config, random.Random(7), self.run_id)
        cursor = MagicMock()
        cursor.__enter__.return_value = cursor
        cursor.execute = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=("PLANT-01", "MAT-1", "ORDER-1"))
        cursor.fetchall = AsyncMock(return_value=[])
        connection = MagicMock()
        connection.cursor.return_value = cursor
        connection.commit = AsyncMock()
        connection.rollback = AsyncMock()
        pool = SimpleNamespace(acquire=lambda: Context(connection), max_lifetime_session=0, ping_interval=60)
        adapter._pool = pool
        adapter._driver = SimpleNamespace(Error=RuntimeError)
        return adapter, connection, cursor

    def mysql(self):
        adapter = MySQLAdapter(self.database, self.config, random.Random(7), self.run_id)
        cursor = SimpleNamespace(execute=AsyncMock(), fetchone=AsyncMock(return_value=(1, 2, "LOT-1", 3)), fetchall=AsyncMock(return_value=[]), rowcount=2)
        connection = SimpleNamespace(cursor=lambda: Context(cursor), commit=AsyncMock(), rollback=AsyncMock())
        adapter._pool = SimpleNamespace(acquire=lambda: Context(connection))
        return adapter, connection, cursor

    def postgresql(self):
        adapter = PostgreSQLAdapter(self.database, self.config, random.Random(7), self.run_id)
        tx = Context(None)
        connection = SimpleNamespace(
            transaction=lambda: tx,
            fetchrow=AsyncMock(return_value={"production_order_no": "ORDER-1", "plant_id": 1, "material_code": "MAT-1", "equipment_id": 2}),
            fetch=AsyncMock(return_value=[]), execute=AsyncMock(return_value="INSERT 0 1"),
        )
        adapter._pool = SimpleNamespace(acquire=lambda: Context(connection))
        return adapter, connection, tx

    async def test_three_scenarios_per_engine_write_multiple_tables_and_read_joins(self):
        for make in (self.oracle, self.postgresql, self.mysql):
            adapter, connection, cursor_or_tx = make()
            self.assertEqual(3, len(set(adapter.business_scenarios)))
            for scenario in adapter.business_scenarios:
                with self.subTest(engine=adapter.engine, scenario=scenario):
                    await adapter.write(self.now, scenario)
                    if adapter.engine == "postgresql":
                        writes = [call.args[0].lower() for call in connection.execute.await_args_list[-3:]]
                    else:
                        writes = [call.args[0].lower() for call in cursor_or_tx.execute.await_args_list[-3:]]
                    self.assertIn("insert into", writes[0])
                    self.assertIn(adapter.business_table.lower(), writes[0])
                    self.assertIn(adapter.runtime_table.lower(), writes[1])
                    self.assertIn("update", writes[2])
                    self.assertIn(adapter.business_table.lower(), writes[2])
                    with patch.object(adapter.randomizer, "choice", return_value=5):
                        await adapter.read(self.now, scenario)
                    if adapter.engine == "postgresql":
                        sql = connection.fetch.await_args.args[0].lower()
                    else:
                        sql = cursor_or_tx.execute.await_args.args[0].lower()
                    self.assertIn(" join ", " ".join(sql.split()))
                    self.assertIn(adapter.business_table.lower(), sql)
                    self.assertIn(adapter.runtime_table.lower(), sql)

    async def test_oracle_rollback_failure_does_not_hide_original_error(self):
        adapter, connection, cursor = self.oracle()
        original = RuntimeError("original")
        cursor.execute.side_effect = [None, None, original]
        connection.rollback.side_effect = RuntimeError("rollback failed")
        with self.assertRaises(RuntimeError) as caught:
            await adapter.write(self.now, "material_issue")
        self.assertIs(original, caught.exception)
        connection.rollback.assert_awaited_once()
        connection.commit.assert_not_awaited()

    async def test_mysql_and_postgresql_roll_back_failed_business_transaction(self):
        mysql, connection, cursor = self.mysql()
        cursor.execute.side_effect = [None, None, RuntimeError("failed")]
        with self.assertRaisesRegex(RuntimeError, "failed"):
            await mysql.write(self.now, "receipt")
        connection.rollback.assert_awaited_once()
        connection.commit.assert_not_awaited()

        postgres, connection, tx = self.postgresql()
        connection.execute.side_effect = [None, RuntimeError("failed")]
        with self.assertRaisesRegex(RuntimeError, "failed"):
            await postgres.write(self.now, "signal_ingest")
        self.assertIs(tx.exit_args[0], RuntimeError)

    async def test_cleanup_covers_both_simulation_tables(self):
        for make in (self.oracle, self.mysql, self.postgresql):
            adapter, connection, cursor_or_tx = make()
            if adapter.engine == "postgresql":
                connection.execute.side_effect = ["DELETE 0", "DELETE 4"]
            result = await adapter.cleanup(self.now, 50)
            self.assertGreater(result, 0)
            if adapter.engine == "postgresql":
                sql = " ".join(call.args[0].lower() for call in connection.execute.await_args_list)
            else:
                sql = " ".join(call.args[0].lower() for call in cursor_or_tx.execute.await_args_list)
            self.assertIn(adapter.runtime_table.lower(), sql)
            self.assertIn(adapter.business_table.lower(), sql)

    async def test_validation_rejects_unprepared_business_table(self):
        for make in (self.oracle, self.mysql, self.postgresql):
            adapter, _, _ = make()
            async def columns(name):
                if name == adapter.runtime_table:
                    return adapter._runtime_columns
                return adapter._base_columns.get(name, set())
            adapter._columns = AsyncMock(side_effect=columns)
            if adapter.engine == "mysql":
                adapter._validate_schema_account = AsyncMock()
            else:
                adapter._validate_schema_owner = AsyncMock()
            with self.subTest(engine=adapter.engine):
                with self.assertRaisesRegex(RuntimeError, "模拟业务表"):
                    await adapter.validate(require_runtime_table=True)

    async def test_prepare_creates_only_simulator_tables_and_indexes(self):
        for make in (self.oracle, self.mysql, self.postgresql):
            adapter, connection, cursor_or_tx = make()
            adapter._columns = AsyncMock(return_value=set())
            if adapter.engine == "oracle":
                cursor_or_tx.fetchone.return_value = (0,)
            await adapter.prepare()
            if adapter.engine == "postgresql":
                calls = connection.execute.await_args_list
            else:
                calls = cursor_or_tx.execute.await_args_list
            sql = " ".join(call.args[0].lower() for call in calls)
            with self.subTest(engine=adapter.engine):
                self.assertIn("create table", sql)
                self.assertIn(adapter.business_table.lower(), sql)
                self.assertIn(adapter.runtime_table.lower(), sql)
                self.assertIn("business", sql)
                self.assertNotIn("delete from", sql)

    async def test_oracle_pool_uses_actual_profile_limits(self):
        adapter, _, cursor = self.oracle()
        cursor.fetchall.return_value = [("CONNECT_TIME", "120"), ("IDLE_TIME", "1")]
        await adapter._configure_resource_limits()
        self.assertEqual(5760, adapter._pool.max_lifetime_session)
        self.assertEqual(30, adapter._pool.ping_interval)

        adapter, _, cursor = self.oracle()
        cursor.fetchall.return_value = [("CONNECT_TIME", "UNLIMITED"), ("IDLE_TIME", "UNLIMITED")]
        await adapter._configure_resource_limits()
        self.assertEqual(0, adapter._pool.max_lifetime_session)
        self.assertEqual(60, adapter._pool.ping_interval)

    async def test_profile_failure_is_logged_without_exception_details(self):
        adapter, _, cursor = self.oracle()
        cursor.execute.side_effect = RuntimeError("secret-value")
        with self.assertLogs("aiops_simulator", level="WARNING") as captured:
            await adapter._configure_resource_limits()
        self.assertNotIn("secret-value", " ".join(captured.output))
        self.assertEqual(0, adapter._pool.max_lifetime_session)

    async def test_oracle_default_resource_limit_is_not_assumed(self):
        adapter, _, cursor = self.oracle()
        cursor.fetchall.return_value = [("CONNECT_TIME", "DEFAULT"), ("IDLE_TIME", "UNLIMITED")]
        with self.assertLogs("aiops_simulator", level="WARNING") as captured:
            await adapter._configure_resource_limits()
        self.assertIn("无法直接确定", " ".join(captured.output))
        self.assertEqual(0, adapter._pool.max_lifetime_session)

    async def test_streams_overlap_without_multiplying_total_rate(self):
        seen, rates = set(), []
        all_started, release, stop = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Adapter:
            display_name = "test"
            business_scenarios = ("one", "two", "three")

            async def read(self, now, scenario):
                seen.add(scenario)
                if len(seen) == 3:
                    all_started.set()
                await release.wait()
                return scenario

        class Mode:
            def operation_mix(self, randomizer):
                return SimpleNamespace(next=lambda: OperationKind.READ)

            def rate_at(self, database, now):
                return 6.0

        class Randomizer:
            def expovariate(self, rate):
                rates.append(rate)
                return 0.001

        runner = DatabaseRunner(self.config, self.database, Adapter(), Mode(), Randomizer(), stop)
        task = asyncio.create_task(runner.run())
        try:
            await asyncio.wait_for(all_started.wait(), 2)
            self.assertEqual({"one", "two", "three"}, seen)
            self.assertTrue(all(rate == 2.0 for rate in rates))
        finally:
            stop.set()
            release.set()
            await asyncio.wait_for(task, 2)
        self.assertEqual({"one", "two", "three"}, set(runner.stats.scenarios))

    def test_resource_limits_and_error_codes_are_safe(self):
        self.assertEqual(7200, _resource_limit_seconds("120"))
        for value in (None, "DEFAULT", "UNLIMITED", "0", "-1", "１２"):
            self.assertIsNone(_resource_limit_seconds(value))
        self.assertEqual("ORA-02399", _driver_error_code(Exception(SimpleNamespace(full_code="ORA-02399"))))
        self.assertEqual("DPY-1001", _driver_error_code(Exception(SimpleNamespace(full_code="DPY-1001"))))
        self.assertEqual("未提供", _driver_error_code(Exception(SimpleNamespace(full_code="secret"))))


if __name__ == "__main__":
    unittest.main()
