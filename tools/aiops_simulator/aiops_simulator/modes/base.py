"""流量模式扩展接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import StrEnum
from random import Random

from aiops_simulator.config import DatabaseConfig


class OperationKind(StrEnum):
    """单次逻辑业务事务类型。"""

    READ = "read"
    WRITE = "write"


class RiskLevel(StrEnum):
    """模式风险级别，为后续故障功能保留强制边界。"""

    NORMAL = "normal"
    ABNORMAL = "abnormal"
    DESTRUCTIVE = "destructive"


class OperationMix:
    """按固定窗口精确生成读写比例，并打散窗口内顺序。"""

    def __init__(self, randomizer: Random, reads: int, writes: int) -> None:
        if reads <= 0 or writes <= 0:
            raise ValueError("读写权重必须大于0")
        self._randomizer = randomizer
        self._template = [OperationKind.READ] * reads + [OperationKind.WRITE] * writes
        self._current: list[OperationKind] = []

    def next(self) -> OperationKind:
        if not self._current:
            self._current = list(self._template)
            self._randomizer.shuffle(self._current)
        return self._current.pop()


class TrafficMode(ABC):
    """流量模式必须声明风险并提供速率与操作组合。"""

    name: str
    risk_level: RiskLevel

    @abstractmethod
    def operation_mix(self, randomizer: Random) -> OperationMix:
        raise NotImplementedError

    @abstractmethod
    def rate_at(self, database: DatabaseConfig, now: datetime) -> float:
        raise NotImplementedError
