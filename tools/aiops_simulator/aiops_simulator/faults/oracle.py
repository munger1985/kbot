"""Oracle可恢复故障实现。"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

from aiops_simulator.faults.base import CONNECTION_SURGE_SIZE, FaultAdapter


class OracleFaultAdapter(FaultAdapter):
    """只操作基准查询表和模拟程序自有故障表。"""

    fault_table = "AIOPS_SIM_FAULT_ACCOUNT"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._driver: Any = None
        self._connections: list[Any] = []
        self._children: set[asyncio.Task[Any]] = set()
        self._principal_validated = False

    async def _connect(self, action: str) -> Any:
        if self._driver is None:
            self._driver = importlib.import_module("oracledb")
        dsn = self._driver.makedsn(
            self.database.host,
            self.database.port,
            service_name=self.database.database,
        )
        connection = await self._driver.connect_async(
            user=self.database.username,
            password=self.database.password,
            dsn=dsn,
        )
        connection.call_timeout = 200_000
        connection.module = "KBot AIOps Fault Simulator"
        connection.action = action
        self._connections.append(connection)
        if not self._principal_validated:
            with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT SYS_CONTEXT('USERENV', 'CURRENT_USER'),
                           SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')
                      FROM DUAL
                    """
                )
                row = await cursor.fetchone()
            expected = self.database.schema.upper()
            if (
                row is None
                or str(row[0]).upper() != expected
                or str(row[1]).upper() != expected
            ):
                raise RuntimeError(
                    "Oracle故障模拟必须使用业务Schema所有者账号，不能使用诊断账号"
                )
            self._principal_validated = True
        return connection

    async def run(self) -> None:
        handler = getattr(self, f"_run_{self.fault_type}")
        await handler()

    async def _run_slow_query(self) -> None:
        connection = await self._connect("fault-slow-query")
        self.ready_event.set()
        while not self.stop_event.is_set():
            with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT /* KBot故障模拟 slow_query */ COUNT(*)
                      FROM ERP_INVENTORY_TRANSACTION
                     WHERE MOD(TRANSACTION_ID, 97) = :bucket
                    """,
                    bucket=37,
                )
                await cursor.fetchone()

    async def _run_temp_pressure(self) -> None:
        connection = await self._connect("fault-temp-pressure")
        self.ready_event.set()
        while not self.stop_event.is_set():
            with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT /* KBot故障模拟 temp_pressure */ COUNT(*)
                      FROM (
                            SELECT TRANSACTION_ID
                              FROM ERP_INVENTORY_TRANSACTION
                             ORDER BY QUANTITY * TRANSACTION_ID, MATERIAL_CODE
                             FETCH FIRST 500000 ROWS ONLY
                      )
                    """
                )
                await cursor.fetchone()

    async def _run_long_transaction(self) -> None:
        connection = await self._connect("fault-long-transaction")
        with connection.cursor() as cursor:
            await cursor.execute(
                f"UPDATE {self.fault_table} "
                "SET BALANCE = BALANCE + 1, UPDATED_AT = SYSTIMESTAMP, "
                "RUN_ID = :run_id WHERE ACCOUNT_ID = 1",
                run_id=self.run_id,
            )
        self.ready_event.set()
        await self.wait_until_stopped()

    async def _run_blocking_lock(self) -> None:
        blocker = await self._connect("fault-lock-blocker")
        waiter = await self._connect("fault-lock-waiter")
        with blocker.cursor() as cursor:
            await cursor.execute(
                f"UPDATE {self.fault_table} "
                "SET BALANCE = BALANCE + 1, UPDATED_AT = SYSTIMESTAMP, "
                "RUN_ID = :run_id WHERE ACCOUNT_ID = 1",
                run_id=self.run_id,
            )

        async def wait_for_lock() -> None:
            with waiter.cursor() as cursor:
                await cursor.execute(
                    f"UPDATE {self.fault_table} "
                    "SET BALANCE = BALANCE - 1, UPDATED_AT = SYSTIMESTAMP, "
                    "RUN_ID = :run_id WHERE ACCOUNT_ID = 1",
                    run_id=self.run_id,
                )

        child = asyncio.create_task(wait_for_lock())
        self._children.add(child)
        self.ready_event.set()
        await self.wait_until_stopped()

    async def _run_deadlock(self) -> None:
        first = await self._connect("fault-deadlock-a")
        second = await self._connect("fault-deadlock-b")
        self.ready_event.set()
        while not self.stop_event.is_set():
            try:
                with first.cursor() as cursor:
                    await cursor.execute(
                        f"UPDATE {self.fault_table} SET RUN_ID = :run_id "
                        "WHERE ACCOUNT_ID = 1",
                        run_id=self.run_id,
                    )
                with second.cursor() as cursor:
                    await cursor.execute(
                        f"UPDATE {self.fault_table} SET RUN_ID = :run_id "
                        "WHERE ACCOUNT_ID = 2",
                        run_id=self.run_id,
                    )

                async def opposite(connection: Any, account_id: int) -> None:
                    with connection.cursor() as cursor:
                        await cursor.execute(
                            f"UPDATE {self.fault_table} SET UPDATED_AT = SYSTIMESTAMP "
                            "WHERE ACCOUNT_ID = :account_id",
                            account_id=account_id,
                        )

                await asyncio.gather(
                    opposite(first, 2), opposite(second, 1), return_exceptions=True
                )
            finally:
                await asyncio.gather(
                    first.rollback(), second.rollback(), return_exceptions=True
                )
            if await self.pause(3):
                break

    async def _run_connection_surge(self) -> None:
        for index in range(CONNECTION_SURGE_SIZE):
            connection = await self._connect(f"fault-connection-{index + 1}")
            with connection.cursor() as cursor:
                await cursor.execute("SELECT 1 FROM DUAL")
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
                await connection.close()
            except Exception:
                pass
        self._connections.clear()
