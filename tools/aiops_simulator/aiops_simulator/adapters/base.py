"""数据库适配器公共接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from calendar import monthrange
from datetime import datetime, timedelta
from random import Random

from aiops_simulator.config import DatabaseConfig, SimulatorConfig


class DatabaseAdapter(ABC):
    """日常模式数据库操作的最小扩展边界。"""

    engine: str
    display_name: str

    def __init__(
        self,
        database: DatabaseConfig,
        simulator: SimulatorConfig,
        randomizer: Random,
        run_id: str,
    ) -> None:
        self.database = database
        self.simulator = simulator
        self.randomizer = randomizer
        self.run_id = run_id

    def historical_time(self, now: datetime, days_back: int = 30) -> datetime:
        """把真实业务时间映射到历史基准年，并保留日内流量特征。"""

        last_day = monthrange(self.simulator.historical_year, now.month)[1]
        mapped = now.replace(
            year=self.simulator.historical_year,
            day=min(now.day, last_day),
            tzinfo=None,
        )
        mapped -= timedelta(days=self.randomizer.randint(0, days_back))
        year_start = datetime(self.simulator.historical_year, 1, 1)
        return max(mapped, year_start)

    @abstractmethod
    async def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def validate(self, *, require_runtime_table: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    async def prepare(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def read(self, now: datetime) -> str:
        """执行一个只读业务事务，并返回模板名称。"""

    @abstractmethod
    async def write(self, now: datetime) -> str:
        """执行一个写业务事务，并返回模板名称。"""

    @abstractmethod
    async def cleanup(self, cutoff: datetime, batch_size: int) -> int:
        """小批量清理过期模拟数据，并返回删除行数。"""
