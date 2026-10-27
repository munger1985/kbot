"""MySQL WMS日常流量适配器。"""

from __future__ import annotations

import importlib
import uuid
from datetime import datetime, timedelta
from typing import Any

from aiops_simulator.adapters.base import DatabaseAdapter


class MySQLAdapter(DatabaseAdapter):
    engine = "mysql"
    display_name = "MySQL WMS"
    business_scenarios = ("receipt", "pick", "shipment")
    runtime_table = "aiops_sim_wms_activity"
    business_table = "aiops_sim_wms_business"
    _business_columns = {"event_id", "run_id", "business_type", "business_key", "started_at", "completed_at", "quantity", "status_code"}
    _runtime_columns = {
        "event_id",
        "run_id",
        "occurred_at",
        "warehouse_id",
        "location_id",
        "material_id",
        "lot_no",
        "document_no",
        "activity_type",
        "quantity",
        "status_code",
    }

    _base_columns = {
        "wms_inventory_movement": {
            "movement_id",
            "movement_time",
            "warehouse_id",
            "location_id",
            "material_id",
            "lot_no",
            "movement_type",
            "quantity",
        },
        "wms_inventory_balance": {
            "inventory_id",
            "location_id",
            "material_id",
            "lot_no",
            "quantity",
            "reserved_quantity",
        },
        "wms_outbound_order": {
            "outbound_order_id",
            "outbound_order_no",
            "warehouse_id",
            "sales_order_no",
            "status_code",
        },
        "wms_warehouse_task": {
            "task_id",
            "task_no",
            "task_type",
            "material_id",
            "status_code",
        },
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._driver: Any = None
        self._pool: Any = None

    async def open(self) -> None:
        self._driver = importlib.import_module("aiomysql")
        self._pool = await self._driver.create_pool(
            host=self.database.host,
            port=self.database.port,
            db=self.database.database,
            user=self.database.username,
            password=self.database.password,
            minsize=self.database.min_pool_size,
            maxsize=self.database.max_pool_size,
            autocommit=False,
            connect_timeout=self.simulator.statement_timeout_seconds,
            charset="utf8mb4",
        )

    async def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    async def _columns(self, table_name: str) -> set[str]:
        async with self._pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT column_name
                      FROM information_schema.columns
                     WHERE table_schema = %s AND table_name = %s
                    """,
                    (self.database.database, table_name),
                )
                return {str(row[0]).lower() for row in await cursor.fetchall()}

    async def _validate_schema_account(self) -> None:
        required = {"SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "INDEX"}
        async with self._pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT privilege_type
                      FROM information_schema.schema_privileges
                     WHERE table_schema = %s
                       AND grantee = CONCAT(
                           QUOTE(SUBSTRING_INDEX(CURRENT_USER(), '@', 1)),
                           '@',
                           QUOTE(SUBSTRING_INDEX(CURRENT_USER(), '@', -1))
                       )
                    """,
                    (self.database.database,),
                )
                actual = {str(row[0]).upper() for row in await cursor.fetchall()}
        if not required.issubset(actual):
            raise RuntimeError(
                "MySQL流量模拟必须使用业务Schema应用账号，不能使用诊断账号"
            )

    async def validate(self, *, require_runtime_table: bool) -> None:
        await self._validate_schema_account()
        for table_name, expected in self._base_columns.items():
            actual = await self._columns(table_name)
            missing = expected - actual
            if missing:
                raise RuntimeError(
                    f"MySQL表{table_name}缺少列：{','.join(sorted(missing))}"
                )
        if require_runtime_table:
            actual = await self._columns(self.runtime_table)
            if not actual:
                raise RuntimeError("MySQL实时活动表尚未准备，请先执行prepare")
            missing = self._runtime_columns - actual
            if missing:
                raise RuntimeError(
                    "MySQL实时活动表缺少列：" + ",".join(sorted(missing))
                )
            actual = await self._columns(self.business_table)
            if self._business_columns - actual:
                raise RuntimeError("MySQL模拟业务表未准备或列不完整，请先执行prepare")

    async def prepare(self) -> None:
        runtime_table_exists = bool(await self._columns(self.runtime_table))
        business_table_exists = bool(await self._columns(self.business_table))
        async with self._pool.acquire() as connection:
            try:
                async with connection.cursor() as cursor:
                    if not runtime_table_exists:
                        await cursor.execute(
                            f"""
                            CREATE TABLE {self.runtime_table} (
                                event_id char(36) PRIMARY KEY,
                                run_id char(36) NOT NULL,
                                occurred_at datetime(6) NOT NULL,
                                warehouse_id int NOT NULL,
                                location_id int NOT NULL,
                                material_id int NOT NULL,
                                lot_no varchar(32) NOT NULL,
                                document_no varchar(32) NOT NULL,
                                activity_type varchar(24) NOT NULL,
                                quantity decimal(14,3) NOT NULL,
                                status_code varchar(16) NOT NULL,
                                INDEX aiops_sim_wms_activity_time_idx
                                    (occurred_at, activity_type)
                            ) TABLESPACE aiops_demo_ts ENGINE=InnoDB
                            """
                        )
                    if not business_table_exists:
                        await cursor.execute(f"""
                            CREATE TABLE {self.business_table} (
                                event_id char(36) PRIMARY KEY,
                                run_id char(36) NOT NULL,
                                business_type varchar(24) NOT NULL,
                                business_key varchar(64) NOT NULL,
                                started_at datetime(6) NOT NULL,
                                completed_at datetime(6),
                                quantity decimal(14,3) NOT NULL,
                                status_code varchar(16) NOT NULL,
                                INDEX aiops_sim_wms_business_time_idx (business_type, started_at),
                                INDEX aiops_sim_wms_business_ret_idx (started_at)
                            ) TABLESPACE aiops_demo_ts ENGINE=InnoDB
                        """)
                    await cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS aiops_sim_fault_account (
                            account_id int PRIMARY KEY,
                            balance decimal(14,2) NOT NULL,
                            updated_at datetime(6) NOT NULL,
                            run_id varchar(36) NOT NULL
                        ) TABLESPACE aiops_demo_ts ENGINE=InnoDB
                        """
                    )
                    await cursor.execute(
                        """
                        INSERT INTO aiops_sim_fault_account (
                            account_id, balance, updated_at, run_id
                        )
                        SELECT source.account_id, 10000, NOW(6), 'READY'
                          FROM (
                              SELECT 1 AS account_id
                              UNION ALL SELECT 2 AS account_id
                          ) source
                         WHERE NOT EXISTS (
                             SELECT 1 FROM aiops_sim_fault_account target
                              WHERE target.account_id = source.account_id
                         )
                        """
                    )
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise

    async def read(self, now: datetime, scenario: str) -> str:
        history = self.historical_time(now)
        templates = {"receipt": (0, 4, 5), "pick": (1, 3, 5), "shipment": (2, 3, 5)}
        template = self.randomizer.choice(templates[scenario])
        async with self._pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SET SESSION MAX_EXECUTION_TIME = %s",
                    (self.simulator.statement_timeout_seconds * 1000,),
                )
                if template == 0:
                    name = "wms_inventory_movement_window"
                    await cursor.execute(
                        """
                        SELECT movement_time, movement_type, quantity, lot_no,
                               reference_document_no
                          FROM wms_inventory_movement
                         WHERE warehouse_id = %s AND material_id = %s
                           AND movement_time >= %s AND movement_time < %s
                         ORDER BY movement_time DESC
                         LIMIT 100
                        """,
                        (
                            self.randomizer.randint(1, 10),
                            self.randomizer.randint(1, 5000),
                            history - timedelta(days=7),
                            history + timedelta(days=1),
                        ),
                    )
                elif template == 1:
                    name = "wms_inventory_balance_lookup"
                    await cursor.execute(
                        """
                        SELECT inventory_id, location_id, material_id, lot_no,
                               quantity, reserved_quantity, last_movement_at
                          FROM wms_inventory_balance
                         WHERE inventory_id = %s
                        """,
                        (self.randomizer.randint(1, 200000),),
                    )
                elif template == 2:
                    name = "wms_outbound_status"
                    await cursor.execute(
                        """
                        SELECT outbound_order_no, sales_order_no, customer_code,
                               planned_ship_at, shipped_at, status_code
                          FROM wms_outbound_order
                         WHERE status_code = %s
                           AND planned_ship_at >= %s AND planned_ship_at < %s
                         ORDER BY planned_ship_at
                         LIMIT 100
                        """,
                        (
                            self.randomizer.choice(["SHIPPED", "BACKLOG"]),
                            history - timedelta(days=30),
                            history + timedelta(days=1),
                        ),
                    )
                elif template == 3:
                    name = "wms_task_status"
                    await cursor.execute(
                        """
                        SELECT task_no, task_type, document_no, material_id,
                               planned_quantity, completed_quantity, status_code
                          FROM wms_warehouse_task
                         WHERE status_code = %s
                           AND assigned_at >= %s AND assigned_at < %s
                         ORDER BY assigned_at DESC
                         LIMIT 100
                        """,
                        (
                            self.randomizer.choice(["COMPLETED", "PENDING"]),
                            history - timedelta(days=30),
                            history + timedelta(days=1),
                        ),
                    )
                elif template == 4:
                    name = "wms_movement_summary"
                    await cursor.execute(
                        """
                        SELECT movement_type, COUNT(*), SUM(quantity)
                          FROM wms_inventory_movement
                         WHERE warehouse_id = %s AND material_id = %s
                           AND movement_time >= %s AND movement_time < %s
                         GROUP BY movement_type
                        """,
                        (
                            self.randomizer.randint(1, 10),
                            self.randomizer.randint(1, 5000),
                            history - timedelta(days=30),
                            history + timedelta(days=1),
                        ),
                    )
                else:
                    name = f"wms_{scenario}_recent_business"
                    await cursor.execute(
                        f"""SELECT b.business_key, b.started_at, b.quantity, a.warehouse_id, a.material_id
                              FROM {self.business_table} b JOIN {self.runtime_table} a ON a.event_id = b.event_id
                             WHERE b.business_type = %s AND b.started_at >= %s
                             ORDER BY b.started_at DESC LIMIT 50""",
                        (scenario.upper(), now.replace(tzinfo=None) - timedelta(days=1)),
                    )
                await cursor.fetchall()
                await connection.rollback()
        return name

    async def write(self, now: datetime, scenario: str) -> str:
        inventory_id = self.randomizer.randint(1, 200000)
        event_id = str(uuid.uuid4())
        async with self._pool.acquire() as connection:
            try:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        """
                        SELECT i.location_id, i.material_id, i.lot_no,
                               ((l.zone_id - 1) MOD 10) + 1 AS warehouse_id
                          FROM wms_inventory_balance i
                          JOIN wms_location l ON l.location_id = i.location_id
                         WHERE i.inventory_id = %s
                        """,
                        (inventory_id,),
                    )
                    row = await cursor.fetchone()
                    if row is None:
                        raise RuntimeError("MySQL库存余额不存在")
                    quantity = self.randomizer.randint(1, 100)
                    timestamp = now.replace(tzinfo=None)
                    await cursor.execute(
                        f"""INSERT INTO {self.business_table}
                            (event_id, run_id, business_type, business_key, started_at, quantity, status_code)
                            VALUES (%s, %s, %s, %s, %s, %s, 'STARTED')""",
                        (event_id, self.run_id, scenario.upper(), f"{row[3]}:{row[0]}:{row[1]}", timestamp, quantity),
                    )
                    await cursor.execute(
                        f"""
                        INSERT INTO {self.runtime_table} (
                            event_id, run_id, occurred_at, warehouse_id,
                            location_id, material_id, lot_no, document_no,
                            activity_type, quantity, status_code
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            event_id,
                            self.run_id,
                            timestamp,
                            int(row[3]),
                            int(row[0]),
                            int(row[1]),
                            str(row[2]),
                            f"SIM-{event_id[:12]}",
                            scenario.upper(),
                            quantity,
                            "COMPLETED",
                        ),
                    )
                    await cursor.execute(
                        f"UPDATE {self.business_table} SET status_code = 'COMPLETED', completed_at = %s WHERE event_id = %s",
                        (timestamp, event_id),
                    )
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise
        return f"wms_{scenario}_transaction"

    async def cleanup(self, cutoff: datetime, batch_size: int) -> int:
        async with self._pool.acquire() as connection:
            try:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        f"""
                        DELETE FROM {self.runtime_table}
                         WHERE occurred_at < %s
                         ORDER BY occurred_at
                         LIMIT %s
                        """,
                        (cutoff.replace(tzinfo=None), batch_size),
                    )
                    deleted = int(cursor.rowcount or 0)
                    await cursor.execute(
                        f"DELETE FROM {self.business_table} WHERE started_at < %s ORDER BY started_at LIMIT %s",
                        (cutoff.replace(tzinfo=None), batch_size),
                    )
                    business_deleted = int(cursor.rowcount or 0)
                await connection.commit()
                return max(deleted, business_deleted)
            except Exception:
                await connection.rollback()
                raise
