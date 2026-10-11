"""PostgreSQL可恢复故障实现。"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

from aiops_simulator.faults.base import CONNECTION_SURGE_SIZE, FaultAdapter


class PostgreSQLFaultAdapter(FaultAdapter):
    """使用独立连接制造可观测且有时限的故障。"""

    fault_table = "aiops_sim_fault_account"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._driver: Any = None
        self._connections: list[Any] = []
        self._children: set[asyncio.Task[Any]] = set()

    @property
    def qualified_fault_table(self) -> str:
        return f"{self.database.schema}.{self.fault_table}"

    async def _connect(self, label: str) -> Any:
        if self._driver is None:
            self._driver = importlib.import_module("asyncpg")
        connection = await self._driver.connect(
            host=self.database.host,
            port=self.database.port,
            database=self.database.database,
            user=self.database.username,
            password=self.database.password,
            timeout=self.simulator.statement_timeout_seconds,
            server_settings={
                "application_name": f"KBot Fault {label} {self.run_id[:8]}",
                "statement_timeout": "200000",
                "lock_timeout": "200000",
            },
        )
        self._connections.append(connection)
        return connection

    async def run(self) -> None:
        handler = getattr(self, f"_run_{self.fault_type}")
        await handler()

    async def _run_slow_query(self) -> None:
        connection = await self._connect("slow-query")
        self.ready_event.set()
        while not self.stop_event.is_set():
            await connection.fetchval(
                f"""
                SELECT /* KBot故障模拟 slow_query */ count(*)
                  FROM {self.database.schema}.mes_equipment_signal
                 WHERE (sample_id % 97) = $1
                """,
                37,
            )

    async def _run_temp_pressure(self) -> None:
        connection = await self._connect("temp-pressure")
        await connection.execute("SET work_mem TO '1MB'")
        self.ready_event.set()
        while not self.stop_event.is_set():
            await connection.fetchval(
                f"""
                SELECT /* KBot故障模拟 temp_pressure */ count(*)
                  FROM (
                        SELECT sample_id
                          FROM {self.database.schema}.mes_equipment_signal
                         ORDER BY metric_value * sample_id, observed_at
                         LIMIT 500000
                  ) sorted_signal
                """
            )

    async def _begin(self, connection: Any) -> None:
        await connection.execute("BEGIN")

    async def _run_long_transaction(self) -> None:
        connection = await self._connect("long-transaction")
        await self._begin(connection)
        await connection.execute(
            f"UPDATE {self.qualified_fault_table} "
            "SET balance = balance + 1, updated_at = clock_timestamp(), "
            "run_id = $1 WHERE account_id = 1",
            self.run_id,
        )
        self.ready_event.set()
        await self.wait_until_stopped()

    async def _run_blocking_lock(self) -> None:
        blocker = await self._connect("lock-blocker")
        waiter = await self._connect("lock-waiter")
        await self._begin(blocker)
        await blocker.execute(
            f"UPDATE {self.qualified_fault_table} SET run_id = $1 "
            "WHERE account_id = 1",
            self.run_id,
        )
        await self._begin(waiter)
        child = asyncio.create_task(
            waiter.execute(
                f"UPDATE {self.qualified_fault_table} "
                "SET updated_at = clock_timestamp(), run_id = $1 "
                "WHERE account_id = 1",
                self.run_id,
            )
        )
        self._children.add(child)
        self.ready_event.set()
        await self.wait_until_stopped()

    async def _run_deadlock(self) -> None:
        first = await self._connect("deadlock-a")
        second = await self._connect("deadlock-b")
        self.ready_event.set()
        while not self.stop_event.is_set():
            await self._begin(first)
            await self._begin(second)
            try:
                await first.execute(
                    f"UPDATE {self.qualified_fault_table} SET run_id = $1 "
                    "WHERE account_id = 1",
                    self.run_id,
                )
                await second.execute(
                    f"UPDATE {self.qualified_fault_table} SET run_id = $1 "
                    "WHERE account_id = 2",
                    self.run_id,
                )
                await asyncio.gather(
                    first.execute(
                        f"UPDATE {self.qualified_fault_table} "
                        "SET updated_at = clock_timestamp() WHERE account_id = 2"
                    ),
                    second.execute(
                        f"UPDATE {self.qualified_fault_table} "
                        "SET updated_at = clock_timestamp() WHERE account_id = 1"
                    ),
                    return_exceptions=True,
                )
            finally:
                await asyncio.gather(
                    first.execute("ROLLBACK"),
                    second.execute("ROLLBACK"),
                    return_exceptions=True,
                )
            if await self.pause(3):
                break

    async def _run_connection_surge(self) -> None:
        for index in range(CONNECTION_SURGE_SIZE):
            connection = await self._connect(f"connection-{index + 1}")
            await connection.fetchval("SELECT 1")
        self.ready_event.set()
        await self.wait_until_stopped()

    async def close(self) -> None:
        for child in self._children:
            child.cancel()
        await asyncio.gather(*self._children, return_exceptions=True)
        for connection in reversed(self._connections):
            try:
                if connection.is_in_transaction():
                    await connection.execute("ROLLBACK")
            except Exception:
                pass
            try:
                await connection.close(timeout=5)
            except Exception:
                try:
                    connection.terminate()
                except Exception:
                    pass
        self._connections.clear()
