"""PostgreSQL MES日常流量适配器。"""

from __future__ import annotations

import importlib
import uuid
from datetime import datetime, timedelta
from typing import Any

from aiops_simulator.adapters.base import DatabaseAdapter


class PostgreSQLAdapter(DatabaseAdapter):
    engine = "postgresql"
    display_name = "PostgreSQL MES"
    runtime_table = "aiops_sim_mes_activity"
    _runtime_columns = {
        "event_id",
        "run_id",
        "occurred_at",
        "plant_code",
        "equipment_id",
        "material_code",
        "production_order_no",
        "activity_type",
        "metric_value",
        "status_code",
    }

    _base_columns = {
        "mes_equipment_signal": {
            "sample_id",
            "equipment_id",
            "metric_id",
            "observed_at",
            "metric_value",
            "quality_code",
            "production_order_no",
        },
        "mes_production_order": {
            "production_order_id",
            "production_order_no",
            "plant_id",
            "material_code",
            "lot_no",
            "status_code",
        },
        "mes_quality_inspection": {
            "inspection_id",
            "production_order_id",
            "inspected_at",
            "result_code",
        },
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._driver: Any = None
        self._pool: Any = None

    @property
    def qualified_runtime_table(self) -> str:
        return f"{self.database.schema}.{self.runtime_table}"

    async def open(self) -> None:
        self._driver = importlib.import_module("asyncpg")
        self._pool = await self._driver.create_pool(
            host=self.database.host,
            port=self.database.port,
            database=self.database.database,
            user=self.database.username,
            password=self.database.password,
            min_size=self.database.min_pool_size,
            max_size=self.database.max_pool_size,
            command_timeout=self.simulator.statement_timeout_seconds,
            server_settings={"application_name": "KBot AIOps Simulator"},
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _columns(self, table_name: str) -> set[str]:
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT column_name
                  FROM information_schema.columns
                 WHERE table_schema = $1 AND table_name = $2
                """,
                self.database.schema,
                table_name,
            )
            return {str(row["column_name"]).lower() for row in rows}

    async def validate(self, *, require_runtime_table: bool) -> None:
        for table_name, expected in self._base_columns.items():
            actual = await self._columns(table_name)
            missing = expected - actual
            if missing:
                raise RuntimeError(
                    f"PostgreSQL表{self.database.schema}.{table_name}缺少列："
                    f"{','.join(sorted(missing))}"
                )
        if require_runtime_table:
            actual = await self._columns(self.runtime_table)
            if not actual:
                raise RuntimeError("PostgreSQL实时活动表尚未准备，请先执行prepare")
            missing = self._runtime_columns - actual
            if missing:
                raise RuntimeError(
                    "PostgreSQL实时活动表缺少列：" + ",".join(sorted(missing))
                )

    async def prepare(self) -> None:
        runtime_table_exists = bool(await self._columns(self.runtime_table))
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                if not runtime_table_exists:
                    await connection.execute(
                        f"""
                        CREATE TABLE {self.qualified_runtime_table} (
                            event_id uuid PRIMARY KEY,
                            run_id uuid NOT NULL,
                            occurred_at timestamp NOT NULL,
                            plant_code varchar(16) NOT NULL,
                            equipment_id integer NOT NULL,
                            material_code varchar(24) NOT NULL,
                            production_order_no varchar(32) NOT NULL,
                            activity_type varchar(24) NOT NULL,
                            metric_value numeric(14,3) NOT NULL,
                            status_code varchar(16) NOT NULL
                        ) TABLESPACE aiops_demo_ts
                        """
                    )
                    await connection.execute(
                        f"""
                        CREATE INDEX aiops_sim_mes_activity_time_idx
                            ON {self.qualified_runtime_table}
                            (occurred_at, activity_type)
                            TABLESPACE aiops_demo_ts
                        """
                    )
                fault_table = f"{self.database.schema}.aiops_sim_fault_account"
                await connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {fault_table} (
                        account_id integer PRIMARY KEY,
                        balance numeric(14,2) NOT NULL,
                        updated_at timestamp NOT NULL,
                        run_id varchar(36) NOT NULL
                    ) TABLESPACE aiops_demo_ts
                    """
                )
                await connection.execute(
                    f"""
                    INSERT INTO {fault_table} (
                        account_id, balance, updated_at, run_id
                    )
                    SELECT source.account_id, 10000, clock_timestamp(), 'READY'
                      FROM (VALUES (1), (2)) AS source(account_id)
                     WHERE NOT EXISTS (
                         SELECT 1 FROM {fault_table} target
                          WHERE target.account_id = source.account_id
                     )
                    """
                )

    async def read(self, now: datetime) -> str:
        history = self.historical_time(now)
        template = self.randomizer.randrange(5)
        schema = self.database.schema
        async with self._pool.acquire() as connection:
            if template == 0:
                name = "mes_equipment_recent_signal"
                await connection.fetch(
                    f"""
                    SELECT observed_at, metric_value, quality_code,
                           production_order_no
                      FROM {schema}.mes_equipment_signal
                     WHERE equipment_id = $1 AND metric_id = $2
                       AND observed_at >= $3 AND observed_at < $4
                     ORDER BY observed_at DESC
                     LIMIT 100
                    """,
                    self.randomizer.randint(1, 500),
                    self.randomizer.randint(1, 20),
                    history - timedelta(days=1),
                    history + timedelta(hours=1),
                )
            elif template == 1:
                name = "mes_equipment_metric_summary"
                await connection.fetch(
                    f"""
                    SELECT date_trunc('hour', observed_at) AS bucket,
                           avg(metric_value), min(metric_value), max(metric_value)
                      FROM {schema}.mes_equipment_signal
                     WHERE equipment_id = $1 AND metric_id = $2
                       AND observed_at >= $3 AND observed_at < $4
                     GROUP BY date_trunc('hour', observed_at)
                     ORDER BY bucket
                    """,
                    self.randomizer.randint(1, 500),
                    self.randomizer.randint(1, 20),
                    history - timedelta(days=1),
                    history + timedelta(hours=1),
                )
            elif template == 2:
                name = "mes_production_order_lookup"
                await connection.fetch(
                    f"""
                    SELECT production_order_no, material_code, lot_no,
                           planned_quantity, completed_quantity, status_code
                      FROM {schema}.mes_production_order
                     WHERE production_order_id = $1
                    """,
                    self.randomizer.randint(1, 100000),
                )
            elif template == 3:
                name = "mes_production_order_status"
                await connection.fetch(
                    f"""
                    SELECT production_order_no, material_code, planned_start_at,
                           planned_end_at, status_code
                      FROM {schema}.mes_production_order
                     WHERE status_code = $1
                       AND planned_start_at >= $2 AND planned_start_at < $3
                     ORDER BY planned_start_at
                     LIMIT 100
                    """,
                    self.randomizer.choice(["COMPLETED", "RUNNING", "DELAYED"]),
                    history - timedelta(days=30),
                    history + timedelta(days=1),
                )
            else:
                name = "mes_quality_exception"
                await connection.fetch(
                    f"""
                    SELECT production_order_id, equipment_id, inspected_at,
                           sample_quantity, defect_quantity, defect_code
                      FROM {schema}.mes_quality_inspection
                     WHERE result_code = 'FAIL'
                       AND inspected_at >= $1 AND inspected_at < $2
                     ORDER BY inspected_at DESC
                     LIMIT 100
                    """,
                    history - timedelta(days=30),
                    history + timedelta(days=1),
                )
        return name

    async def write(self, now: datetime) -> str:
        event_id = uuid.uuid4()
        order_id = self.randomizer.randint(1, 100000)
        schema = self.database.schema
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    f"""
                    SELECT o.production_order_no, o.material_code, o.plant_id,
                           ((o.production_order_id - 1) % 500) + 1 AS equipment_id
                      FROM {schema}.mes_production_order o
                     WHERE o.production_order_id = $1
                    """,
                    order_id,
                )
                if row is None:
                    raise RuntimeError("PostgreSQL生产订单不存在")
                await connection.execute(
                    f"""
                    INSERT INTO {self.qualified_runtime_table} (
                        event_id, run_id, occurred_at, plant_code,
                        equipment_id, material_code, production_order_no,
                        activity_type, metric_value, status_code
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                    event_id,
                    uuid.UUID(self.run_id),
                    now.replace(tzinfo=None),
                    f"PLANT-{int(row['plant_id']):02d}",
                    int(row["equipment_id"]),
                    str(row["material_code"]),
                    str(row["production_order_no"]),
                    self.randomizer.choice(
                        ["SIGNAL_INGEST", "OPERATION_REPORT", "QUALITY_REPORT"]
                    ),
                    float(self.randomizer.randint(1000, 9000)) / 100,
                    "ACCEPTED",
                )
        return "mes_live_activity"

    async def cleanup(self, cutoff: datetime, batch_size: int) -> int:
        async with self._pool.acquire() as connection:
            status = await connection.execute(
                f"""
                WITH expired AS (
                    SELECT ctid FROM {self.qualified_runtime_table}
                     WHERE occurred_at < $1
                     ORDER BY occurred_at
                     LIMIT $2
                )
                DELETE FROM {self.qualified_runtime_table} target
                 USING expired
                 WHERE target.ctid = expired.ctid
                """,
                cutoff.replace(tzinfo=None),
                batch_size,
            )
            return int(status.rsplit(" ", 1)[-1])
