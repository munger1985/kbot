"""故障类型合同与数据库适配器公共接口。"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from aiops_simulator.config import DatabaseConfig, SimulatorConfig


FAULT_TTL_SECONDS = 600
CONNECTION_SURGE_SIZE = 12
SUPPORTED_FAULTS = (
    "slow_query",
    "blocking_lock",
    "deadlock",
    "long_transaction",
    "connection_surge",
    "temp_pressure",
)
DATABASE_ALIASES = {
    "oracle": "oracle",
    "postgres": "postgresql",
    "mysql": "mysql",
}


def normalize_database_name(name: str) -> str:
    """把公开命令名转换为内部数据库引擎名。"""

    try:
        return DATABASE_ALIASES[name]
    except KeyError as exc:
        supported = "、".join(DATABASE_ALIASES)
        raise ValueError(f"数据库必须是：{supported}") from exc


def validate_fault_type(fault_type: str) -> str:
    """校验公开故障类型并返回原值。"""

    if fault_type not in SUPPORTED_FAULTS:
        supported = "、".join(SUPPORTED_FAULTS)
        raise ValueError(f"故障类型必须是：{supported}")
    return fault_type


class FaultAdapter(ABC):
    """单数据库故障实现的受控运行边界。"""

    def __init__(
        self,
        database: DatabaseConfig,
        simulator: SimulatorConfig,
        run_id: str,
        fault_type: str,
        stop_event: asyncio.Event,
    ) -> None:
        self.database = database
        self.simulator = simulator
        self.run_id = run_id
        self.fault_type = fault_type
        self.stop_event = stop_event
        self.ready_event = asyncio.Event()

    async def wait_until_stopped(self) -> None:
        await self.stop_event.wait()

    async def pause(self, seconds: float) -> bool:
        """等待指定时间；收到停止信号时返回真。"""

        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=seconds)
            return True
        except TimeoutError:
            return False

    @abstractmethod
    async def run(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """回滚并关闭本次故障创建的全部连接。"""
