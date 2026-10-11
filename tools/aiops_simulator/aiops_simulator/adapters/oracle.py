"""Oracle ERP日常流量适配器。"""

from __future__ import annotations

import importlib
import uuid
from datetime import datetime, timedelta
from typing import Any

from aiops_simulator.adapters.base import DatabaseAdapter


class OracleAdapter(DatabaseAdapter):
    engine = "oracle"
    display_name = "Oracle ERP"
    runtime_table = "AIOPS_SIM_ERP_ACTIVITY"
    _runtime_columns = {
        "EVENT_ID",
        "RUN_ID",
        "OCCURRED_AT",
        "PLANT_CODE",
        "MATERIAL_CODE",
        "PRODUCTION_ORDER_NO",
        "ACTIVITY_TYPE",
        "QUANTITY",
        "STATUS_CODE",
    }

    _base_columns = {
        "ERP_INVENTORY_TRANSACTION": {
            "TRANSACTION_ID",
            "TRANSACTION_TIME",
            "PLANT_CODE",
            "MATERIAL_CODE",
            "LOT_NO",
            "MOVEMENT_TYPE",
            "QUANTITY",
            "REFERENCE_DOCUMENT_NO",
        },
        "ERP_PRODUCTION_ORDER": {
            "PRODUCTION_ORDER_ID",
            "PRODUCTION_ORDER_NO",
            "PLANT_ID",
            "MATERIAL_ID",
            "LOT_NO",
            "STATUS_CODE",
        },
        "ERP_SALES_ORDER": {
            "SALES_ORDER_ID",
            "SALES_ORDER_NO",
            "ORDER_DATE",
            "STATUS_CODE",
        },
        "ERP_MATERIAL": {"MATERIAL_ID", "MATERIAL_CODE", "STANDARD_COST"},
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._driver: Any = None
        self._pool: Any = None

    async def open(self) -> None:
        self._driver = importlib.import_module("oracledb")
        dsn = self._driver.makedsn(
            self.database.host,
            self.database.port,
            service_name=self.database.database,
        )
        self._pool = self._driver.create_pool_async(
            user=self.database.username,
            password=self.database.password,
            dsn=dsn,
            min=self.database.min_pool_size,
            max=self.database.max_pool_size,
            increment=1,
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close(force=True)
            self._pool = None

    async def _columns(self, table_name: str) -> set[str]:
        async with self._pool.acquire() as connection:
            with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT COLUMN_NAME FROM USER_TAB_COLUMNS WHERE TABLE_NAME = :name",
                    name=table_name.upper(),
                )
                return {str(row[0]).upper() for row in await cursor.fetchall()}

    async def validate(self, *, require_runtime_table: bool) -> None:
        for table_name, expected in self._base_columns.items():
            actual = await self._columns(table_name)
            missing = expected - actual
            if missing:
                raise RuntimeError(
                    f"Oracle表{table_name}缺少列：{','.join(sorted(missing))}"
                )
        if require_runtime_table:
            actual = await self._columns(self.runtime_table)
            if not actual:
                raise RuntimeError("Oracle实时活动表尚未准备，请先执行prepare")
            missing = self._runtime_columns - actual
            if missing:
                raise RuntimeError(
                    "Oracle实时活动表缺少列：" + ",".join(sorted(missing))
                )

    async def prepare(self) -> None:
        table_exists = bool(await self._columns(self.runtime_table))
        ddl = f"""
            CREATE TABLE {self.runtime_table} (
                EVENT_ID VARCHAR2(36) PRIMARY KEY,
                RUN_ID VARCHAR2(36) NOT NULL,
                OCCURRED_AT TIMESTAMP NOT NULL,
                PLANT_CODE VARCHAR2(16) NOT NULL,
                MATERIAL_CODE VARCHAR2(24) NOT NULL,
                PRODUCTION_ORDER_NO VARCHAR2(32) NOT NULL,
                ACTIVITY_TYPE VARCHAR2(24) NOT NULL,
                QUANTITY NUMBER(14,3) NOT NULL,
                STATUS_CODE VARCHAR2(16) NOT NULL
            ) TABLESPACE AIOPS_DEMO_TS
        """
        async with self._pool.acquire() as connection:
            with connection.cursor() as cursor:
                if not table_exists:
                    await cursor.execute(ddl)
                await cursor.execute(
                    "SELECT COUNT(*) FROM USER_INDEXES WHERE INDEX_NAME = :name",
                    name="AIOPS_SIM_ERP_ACTIVITY_TIME_IDX",
                )
                index_exists = int((await cursor.fetchone())[0]) > 0
                if not index_exists:
                    await cursor.execute(
                        f"CREATE INDEX AIOPS_SIM_ERP_ACTIVITY_TIME_IDX "
                        f"ON {self.runtime_table} (OCCURRED_AT, ACTIVITY_TYPE) "
                        "TABLESPACE AIOPS_DEMO_TS"
                    )
                await cursor.execute(
                    "SELECT COUNT(*) FROM USER_TABLES WHERE TABLE_NAME = :name",
                    name="AIOPS_SIM_FAULT_ACCOUNT",
                )
                fault_table_exists = int((await cursor.fetchone())[0]) > 0
                if not fault_table_exists:
                    await cursor.execute(
                        """
                        CREATE TABLE AIOPS_SIM_FAULT_ACCOUNT (
                            ACCOUNT_ID NUMBER PRIMARY KEY,
                            BALANCE NUMBER(14,2) NOT NULL,
                            UPDATED_AT TIMESTAMP NOT NULL,
                            RUN_ID VARCHAR2(36) NOT NULL
                        ) TABLESPACE AIOPS_DEMO_TS
                        """
                    )
                await cursor.execute(
                    """
                    MERGE INTO AIOPS_SIM_FAULT_ACCOUNT target
                    USING (
                        SELECT 1 AS ACCOUNT_ID FROM DUAL
                        UNION ALL
                        SELECT 2 AS ACCOUNT_ID FROM DUAL
                    ) source
                       ON (target.ACCOUNT_ID = source.ACCOUNT_ID)
                     WHEN NOT MATCHED THEN INSERT (
                         ACCOUNT_ID, BALANCE, UPDATED_AT, RUN_ID
                     ) VALUES (
                         source.ACCOUNT_ID, 10000, SYSTIMESTAMP, 'READY'
                     )
                    """
                )
                await connection.commit()

    def _set_context(self, connection: Any, action: str) -> None:
        connection.call_timeout = self.simulator.statement_timeout_seconds * 1000
        connection.module = "KBot AIOps Simulator"
        connection.action = action

    async def read(self, now: datetime) -> str:
        history = self.historical_time(now)
        template = self.randomizer.randrange(5)
        async with self._pool.acquire() as connection:
            self._set_context(connection, "daily-read")
            with connection.cursor() as cursor:
                if template == 0:
                    name = "erp_inventory_window"
                    await cursor.execute(
                        """
                        SELECT TRANSACTION_ID, TRANSACTION_TIME, MOVEMENT_TYPE,
                               QUANTITY, REFERENCE_DOCUMENT_NO
                          FROM ERP_INVENTORY_TRANSACTION
                         WHERE PLANT_CODE = :plant_code
                           AND MATERIAL_CODE = :material_code
                           AND TRANSACTION_TIME >= :start_at
                           AND TRANSACTION_TIME < :end_at
                         FETCH FIRST 50 ROWS ONLY
                        """,
                        plant_code=f"PLANT-{self.randomizer.randint(1, 5):02d}",
                        material_code=f"MAT-{self.randomizer.randint(1, 5000):06d}",
                        start_at=history - timedelta(days=7),
                        end_at=history + timedelta(days=1),
                    )
                elif template == 1:
                    name = "erp_inventory_summary"
                    await cursor.execute(
                        """
                        SELECT MOVEMENT_TYPE, COUNT(*), SUM(QUANTITY)
                          FROM ERP_INVENTORY_TRANSACTION
                         WHERE PLANT_CODE = :plant_code
                           AND MATERIAL_CODE = :material_code
                           AND TRANSACTION_TIME >= :start_at
                           AND TRANSACTION_TIME < :end_at
                         GROUP BY MOVEMENT_TYPE
                        """,
                        plant_code=f"PLANT-{self.randomizer.randint(1, 5):02d}",
                        material_code=f"MAT-{self.randomizer.randint(1, 5000):06d}",
                        start_at=history - timedelta(days=30),
                        end_at=history + timedelta(days=1),
                    )
                elif template == 2:
                    name = "erp_sales_order_status"
                    await cursor.execute(
                        """
                        SELECT SALES_ORDER_NO, ORDER_DATE, ORDER_AMOUNT, STATUS_CODE
                          FROM ERP_SALES_ORDER
                         WHERE STATUS_CODE = :status_code
                           AND ORDER_DATE >= :start_at
                           AND ORDER_DATE < :end_at
                         FETCH FIRST 50 ROWS ONLY
                        """,
                        status_code=self.randomizer.choice(
                            ["COMPLETED", "OPEN", "DELAYED"]
                        ),
                        start_at=history - timedelta(days=30),
                        end_at=history + timedelta(days=1),
                    )
                elif template == 3:
                    name = "erp_production_order_status"
                    await cursor.execute(
                        """
                        SELECT PRODUCTION_ORDER_NO, LOT_NO, PLANNED_QUANTITY,
                               COMPLETED_QUANTITY, STATUS_CODE
                          FROM ERP_PRODUCTION_ORDER
                         WHERE STATUS_CODE = :status_code
                           AND PLANNED_START_DATE >= :start_at
                           AND PLANNED_START_DATE < :end_at
                         FETCH FIRST 50 ROWS ONLY
                        """,
                        status_code=self.randomizer.choice(
                            ["COMPLETED", "RUNNING", "DELAYED"]
                        ),
                        start_at=history - timedelta(days=30),
                        end_at=history + timedelta(days=1),
                    )
                else:
                    name = "erp_material_lookup"
                    await cursor.execute(
                        """
                        SELECT MATERIAL_CODE, MATERIAL_NAME, MATERIAL_GROUP,
                               STANDARD_COST, STATUS_CODE
                          FROM ERP_MATERIAL
                         WHERE MATERIAL_ID = :material_id
                        """,
                        material_id=self.randomizer.randint(1, 5000),
                    )
                await cursor.fetchall()
        return name

    async def write(self, now: datetime) -> str:
        production_order_id = self.randomizer.randint(1, 100000)
        event_id = str(uuid.uuid4())
        async with self._pool.acquire() as connection:
            self._set_context(connection, "daily-write")
            try:
                with connection.cursor() as cursor:
                    await cursor.execute(
                        """
                        SELECT p.PLANT_CODE, m.MATERIAL_CODE,
                               o.PRODUCTION_ORDER_NO
                          FROM ERP_PRODUCTION_ORDER o
                          JOIN ERP_PLANT p ON p.PLANT_ID = o.PLANT_ID
                          JOIN ERP_MATERIAL m ON m.MATERIAL_ID = o.MATERIAL_ID
                         WHERE o.PRODUCTION_ORDER_ID = :order_id
                        """,
                        order_id=production_order_id,
                    )
                    row = await cursor.fetchone()
                    if row is None:
                        raise RuntimeError("Oracle生产订单不存在")
                    await cursor.execute(
                        f"""
                        INSERT INTO {self.runtime_table} (
                            EVENT_ID, RUN_ID, OCCURRED_AT, PLANT_CODE,
                            MATERIAL_CODE, PRODUCTION_ORDER_NO, ACTIVITY_TYPE,
                            QUANTITY, STATUS_CODE
                        ) VALUES (
                            :event_id, :run_id, :occurred_at, :plant_code,
                            :material_code, :order_no, :activity_type,
                            :quantity, :status_code
                        )
                        """,
                        event_id=event_id,
                        run_id=self.run_id,
                        occurred_at=now.replace(tzinfo=None),
                        plant_code=row[0],
                        material_code=row[1],
                        order_no=row[2],
                        activity_type=self.randomizer.choice(
                            ["ORDER_QUERY", "MATERIAL_ISSUE", "FINISH_RECEIPT"]
                        ),
                        quantity=self.randomizer.randint(1, 100),
                        status_code="POSTED",
                    )
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise
        return "erp_live_activity"

    async def cleanup(self, cutoff: datetime, batch_size: int) -> int:
        async with self._pool.acquire() as connection:
            self._set_context(connection, "daily-cleanup")
            try:
                with connection.cursor() as cursor:
                    await cursor.execute(
                        f"""
                        DELETE FROM {self.runtime_table}
                         WHERE ROWID IN (
                             SELECT ROWID FROM {self.runtime_table}
                              WHERE OCCURRED_AT < :cutoff
                                AND ROWNUM <= :batch_size
                         )
                        """,
                        cutoff=cutoff.replace(tzinfo=None),
                        batch_size=batch_size,
                    )
                    deleted = int(cursor.rowcount or 0)
                await connection.commit()
                return deleted
            except Exception:
                await connection.rollback()
                raise
