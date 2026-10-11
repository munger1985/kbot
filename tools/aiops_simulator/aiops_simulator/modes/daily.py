"""真实时间日常流量模式。"""

from __future__ import annotations

from datetime import datetime
from random import Random

from aiops_simulator.config import DatabaseConfig
from aiops_simulator.modes.base import OperationMix, RiskLevel, TrafficMode


class DailyTrafficMode(TrafficMode):
    """长期常驻的安全日常流量，固定保持8比2读写比例。"""

    name = "daily"
    risk_level = RiskLevel.NORMAL

    def operation_mix(self, randomizer: Random) -> OperationMix:
        return OperationMix(randomizer, reads=80, writes=20)

    def rate_at(self, database: DatabaseConfig, now: datetime) -> float:
        minute_of_day = now.hour * 60 + now.minute
        return database.traffic.rate_at(minute_of_day, now.weekday())
