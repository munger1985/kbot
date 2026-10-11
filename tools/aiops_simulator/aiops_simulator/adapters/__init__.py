"""数据库适配器工厂。"""

from __future__ import annotations

from random import Random

from aiops_simulator.adapters.base import DatabaseAdapter
from aiops_simulator.adapters.mysql import MySQLAdapter
from aiops_simulator.adapters.oracle import OracleAdapter
from aiops_simulator.adapters.postgresql import PostgreSQLAdapter
from aiops_simulator.config import DatabaseConfig, SimulatorConfig


def create_adapter(
    database: DatabaseConfig,
    simulator: SimulatorConfig,
    randomizer: Random,
    run_id: str,
) -> DatabaseAdapter:
    """按显式数据库类型创建适配器，不做环境猜测。"""

    adapters: dict[str, type[DatabaseAdapter]] = {
        OracleAdapter.engine: OracleAdapter,
        PostgreSQLAdapter.engine: PostgreSQLAdapter,
        MySQLAdapter.engine: MySQLAdapter,
    }
    adapter_type = adapters.get(database.engine)
    if adapter_type is None:
        raise ValueError(f"不支持的数据库类型：{database.engine}")
    return adapter_type(database, simulator, randomizer, run_id)


__all__ = ["create_adapter", "DatabaseAdapter"]
