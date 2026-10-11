"""MySQL可恢复故障实现。"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

from aiops_simulator.faults.base import CONNECTION_SURGE_SIZE, FaultAdapter


class MySQLFaultAdapter(FaultAdapter):
    """以模拟程序自有连接和表构造故障。"""

    fault_table = "aiops_sim_fault_account"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._driver: Any = None
        self._connections: list[Any] = []
        self._children: set[asyncio.Task[Any]] = set()

    async def _connect(self, label: str) -> Any:
        if self._driver is None:
            self._driver = importlib.import_module("aiomysql")
        connection = await self._driver.connect(
            host=self.database.host,
            port=self.database.port,
            db=self.database.database,
            user=self.database.username,
            password=self.database.password,
            autocommit=False,
            connect_timeout=self.simulator.statement_timeout_seconds,
            charset="utf8mb4",
        )
        self._connections.append(connection)
        async with connection.cursor() as cursor:
            await cursor.execute("SET SESSION innodb_lock_wait_timeout = 200")
            await cursor.execute("SET SESSION MAX_EXECUTION_TIME = 200000")
            await cursor.execute(
                "SET @kbot_fault_run_id = %s, @kbot_fault_label = %s",
                (self.run_id, label),
            )
        return connection

    async def run(self) -> None:
        handler = getattr(self, f"_run_{self.fault_type}")
        await handler()

    async def _run_slow_query(self) -> None:
        connection = await self._connect("slow-query")
        self.ready_event.set()
        while not self.stop_event.is_set():
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT /* KBot故障模拟 slow_query */ COUNT(*)
                      FROM wms_inventory_movement
                     WHERE MOD(movement_id, 97) = %s
                    """,
                    (37,),
                )
                await cursor.fetchone()
            await connection.rollback()

    async def _run_temp_pressure(self) -> None:
        connection = await self._connect("temp-pressure")
        self.ready_event.set()
        while not self.stop_event.is_set():
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT /* KBot故障模拟 temp_pressure */ COUNT(*)
                      FROM (
                            SELECT movement_id
                              FROM wms_inventory_movement
                             ORDER BY quantity * movement_id, movement_time
                             LIMIT 500000
                      ) AS sorted_movement
                    """
                )
                await cursor.fetchone()
            await connection.rollback()

    async def _execute(self, connection: Any, sql: str, parameters: tuple[Any, ...]) -> None:
        async with connection.cursor() as cursor:
            await cursor.execute(sql, parameters)

    async def _run_long_transaction(self) -> None:
        connection = await self._connect("long-transaction")
        await connection.begin()
        await self._execute(
            connection,
            f"UPDATE {self.fault_table} "
            "SET balance = balance + 1, updated_at = NOW(6), run_id = %s "
            "WHERE account_id = 1",
            (self.run_id,),
        )
        self.ready_event.set()
        await self.wait_until_stopped()

    async def _run_blocking_lock(self) -> None:
        blocker = await self._connect("lock-blocker")
        waiter = await self._connect("lock-waiter")
        await blocker.begin()
        await self._execute(
            blocker,
            f"UPDATE {self.fault_table} SET run_id = %s WHERE account_id = 1",
            (self.run_id,),
        )
        await waiter.begin()
        child = asyncio.create_task(
            self._execute(
                waiter,
                f"UPDATE {self.fault_table} "
                "SET updated_at = NOW(6), run_id = %s WHERE account_id = 1",
                (self.run_id,),
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
            await first.begin()
            await second.begin()
            try:
                await self._execute(
                    first,
                    f"UPDATE {self.fault_table} SET run_id = %s "
                    "WHERE account_id = 1",
                    (self.run_id,),
                )
                await self._execute(
                    second,
                    f"UPDATE {self.fault_table} SET run_id = %s "
                    "WHERE account_id = 2",
                    (self.run_id,),
                )
                await asyncio.gather(
                    self._execute(
                        first,
                        f"UPDATE {self.fault_table} SET updated_at = NOW(6) "
                        "WHERE account_id = 2",
                        (),
                    ),
                    self._execute(
                        second,
                        f"UPDATE {self.fault_table} SET updated_at = NOW(6) "
                        "WHERE account_id = 1",
                        (),
                    ),
                    return_exceptions=True,
                )
            finally:
                await asyncio.gather(
                    first.rollback(), second.rollback(), return_exceptions=True
                )
            if await self.pause(3):
                break

    async def _run_connection_surge(self) -> None:
        for index in range(CONNECTION_SURGE_SIZE):
            connection = await self._connect(f"connection-{index + 1}")
            async with connection.cursor() as cursor:
                await cursor.execute("SELECT 1")
                await cursor.fetchone()
        self.ready_event.set()
        await self.wait_until_stopped()

    async def close(self) -> None:
        for child in self._children:
            child.cancel()
        await asyncio.gather(*self._children, return_exceptions=True)
        for connection in reversed(self._connections):
            try:
                await connection.rollback()
            except Exception:
                pass
            try:
                connection.close()
            except Exception:
                pass
        self._connections.clear()
