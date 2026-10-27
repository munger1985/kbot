"""Oracle ERP日常流量适配器。"""

from __future__ import annotations

import importlib
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

from aiops_simulator.adapters.base import DatabaseAdapter

LOGGER = logging.getLogger("aiops_simulator")


def _resource_limit_seconds(value: object) -> int | None:
    text = str(value).strip()
    return int(text) * 60 if text.isascii() and text.isdecimal() and int(text) > 0 else None


class OracleAdapter(DatabaseAdapter):
    engine = "oracle"
    display_name = "Oracle ERP"
    business_scenarios = ("production_report", "material_issue", "finish_receipt")
    runtime_table = "AIOPS_SIM_ERP_ACTIVITY"
    business_table = "AIOPS_SIM_ERP_BUSINESS"
    _business_columns = {"EVENT_ID", "RUN_ID", "BUSINESS_TYPE", "BUSINESS_KEY", "STARTED_AT", "COMPLETED_AT", "QUANTITY", "STATUS_CODE"}
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
        try:
            await self._configure_resource_limits()
        except BaseException:
            await self._pool.close(force=True)
            self._pool = None
            raise

    async def _configure_resource_limits(self) -> None:
        """让连接池在账号 Profile 限制到期前回收或检测连接。"""
        try:
            async with self._pool.acquire() as connection:
                with connection.cursor() as cursor:
                    await cursor.execute(
                        "SELECT resource_name, limit FROM user_resource_limits "
                        "WHERE resource_name IN ('CONNECT_TIME', 'IDLE_TIME')"
                    )
                    limits = {str(name).upper(): value for name, value in await cursor.fetchall()}
        except self._driver.Error as exc:
            LOGGER.warning("Oracle连接池无法读取账号资源限制，使用驱动默认策略，错误类型=%s", type(exc).__name__)
            return

        connect_seconds = _resource_limit_seconds(limits.get("CONNECT_TIME"))
        if connect_seconds is not None:
            # 提前回收，并为当前事务及驱动超时留出余量。
            lifetime = max(1, min(connect_seconds * 4 // 5, connect_seconds - self.simulator.statement_timeout_seconds - 2))
            self._pool.max_lifetime_session = lifetime
            LOGGER.info("Oracle连接池按账号连接时限提前回收，最大生命周期=%d秒", lifetime)
        idle_seconds = _resource_limit_seconds(limits.get("IDLE_TIME"))
        if idle_seconds is not None:
            ping_interval = max(1, idle_seconds // 2)
            current = self._pool.ping_interval
            if current < 0 or current > ping_interval:
                self._pool.ping_interval = ping_interval
            LOGGER.info("Oracle连接池空闲连接检查间隔=%d秒", self._pool.ping_interval)
        for name, value in limits.items():
            if str(value).upper() == "DEFAULT":
                LOGGER.warning("Oracle账号%s继承默认限制，无法直接确定连接池时限", name)

    async def _rollback_after_failure(self, connection: Any) -> None:
        try:
            await connection.rollback()
        except Exception as exc:
            LOGGER.warning("Oracle事务回滚失败，错误类型=%s", type(exc).__name__)

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

    async def _validate_schema_owner(self) -> None:
        async with self._pool.acquire() as connection:
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
        if row is None or str(row[0]).upper() != expected or str(row[1]).upper() != expected:
            raise RuntimeError(
                "Oracle流量模拟必须使用业务Schema所有者账号，不能使用诊断账号"
            )

    async def validate(self, *, require_runtime_table: bool) -> None:
        await self._validate_schema_owner()
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
            actual = await self._columns(self.business_table)
            missing = self._business_columns - actual
            if missing:
                raise RuntimeError("Oracle模拟业务表未准备或列不完整，请先执行prepare")

    async def prepare(self) -> None:
        table_exists = bool(await self._columns(self.runtime_table))
        business_exists = bool(await self._columns(self.business_table))
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
                if not business_exists:
                    await cursor.execute(f"""
                        CREATE TABLE {self.business_table} (
                            EVENT_ID VARCHAR2(36) PRIMARY KEY,
                            RUN_ID VARCHAR2(36) NOT NULL,
                            BUSINESS_TYPE VARCHAR2(24) NOT NULL,
                            BUSINESS_KEY VARCHAR2(64) NOT NULL,
                            STARTED_AT TIMESTAMP NOT NULL,
                            COMPLETED_AT TIMESTAMP,
                            QUANTITY NUMBER(14,3) NOT NULL,
                            STATUS_CODE VARCHAR2(16) NOT NULL
                        ) TABLESPACE AIOPS_DEMO_TS
                    """)
                await cursor.execute(
                    "SELECT COUNT(*) FROM USER_INDEXES WHERE INDEX_NAME = :name",
                    name="AIOPS_SIM_ERP_BUS_TIME_IDX",
                )
                if int((await cursor.fetchone())[0]) == 0:
                    await cursor.execute(
                        f"CREATE INDEX AIOPS_SIM_ERP_BUS_TIME_IDX ON {self.business_table} "
                        "(BUSINESS_TYPE, STARTED_AT) TABLESPACE AIOPS_DEMO_TS"
                    )
                await cursor.execute(
                    "SELECT COUNT(*) FROM USER_INDEXES WHERE INDEX_NAME = :name",
                    name="AIOPS_SIM_ERP_BUS_RET_IDX",
                )
                if int((await cursor.fetchone())[0]) == 0:
                    await cursor.execute(
                        f"CREATE INDEX AIOPS_SIM_ERP_BUS_RET_IDX ON {self.business_table} "
                        "(STARTED_AT) TABLESPACE AIOPS_DEMO_TS"
                    )
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

    async def read(self, now: datetime, scenario: str) -> str:
        history = self.historical_time(now)
        templates = {"production_report": (2, 3, 5), "material_issue": (0, 1, 5), "finish_receipt": (2, 4, 5)}
        template = self.randomizer.choice(templates[scenario])
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
                elif template == 4:
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
                else:
                    name = f"erp_{scenario}_recent_business"
                    await cursor.execute(
                        f"""
                        SELECT b.BUSINESS_KEY, b.STARTED_AT, b.QUANTITY, a.PLANT_CODE, a.MATERIAL_CODE
                          FROM {self.business_table} b JOIN {self.runtime_table} a ON a.EVENT_ID = b.EVENT_ID
                         WHERE b.BUSINESS_TYPE = :business_type AND b.STARTED_AT >= :start_at
                         ORDER BY b.STARTED_AT DESC FETCH FIRST 50 ROWS ONLY
                        """,
                        business_type=scenario.upper(), start_at=now.replace(tzinfo=None) - timedelta(days=1),
                    )
                await cursor.fetchall()
        return name

    async def write(self, now: datetime, scenario: str) -> str:
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
                    quantity = self.randomizer.randint(1, 100)
                    timestamp = now.replace(tzinfo=None)
                    await cursor.execute(
                        f"""INSERT INTO {self.business_table}
                            (EVENT_ID, RUN_ID, BUSINESS_TYPE, BUSINESS_KEY, STARTED_AT, QUANTITY, STATUS_CODE)
                            VALUES (:event_id, :run_id, :business_type, :business_key, :started_at, :quantity, 'STARTED')""",
                        event_id=event_id, run_id=self.run_id, business_type=scenario.upper(), business_key=row[2], started_at=timestamp, quantity=quantity,
                    )
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
                        occurred_at=timestamp,
                        plant_code=row[0],
                        material_code=row[1],
                        order_no=row[2],
                        activity_type=scenario.upper(),
                        quantity=quantity,
                        status_code="POSTED",
                    )
                    await cursor.execute(
                        f"UPDATE {self.business_table} SET STATUS_CODE = 'COMPLETED', COMPLETED_AT = :completed_at WHERE EVENT_ID = :event_id",
                        completed_at=timestamp, event_id=event_id,
                    )
                await connection.commit()
            except Exception:
                await self._rollback_after_failure(connection)
                raise
        return f"erp_{scenario}_transaction"

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
                    await cursor.execute(
                        f"""DELETE FROM {self.business_table} WHERE ROWID IN (
                            SELECT ROWID FROM {self.business_table}
                            WHERE STARTED_AT < :cutoff AND ROWNUM <= :batch_size
                        )""",
                        cutoff=cutoff.replace(tzinfo=None), batch_size=batch_size,
                    )
                    business_deleted = int(cursor.rowcount or 0)
                await connection.commit()
                return max(deleted, business_deleted)
            except Exception:
                await self._rollback_after_failure(connection)
                raise
