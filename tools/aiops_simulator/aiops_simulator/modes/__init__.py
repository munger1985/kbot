"""流量模式注册表。"""

from aiops_simulator.modes.base import RiskLevel, TrafficMode
from aiops_simulator.modes.daily import DailyTrafficMode


def load_mode(name: str) -> TrafficMode:
    """只加载已经完整实现并明确风险级别的模式。"""

    modes: dict[str, type[TrafficMode]] = {DailyTrafficMode.name: DailyTrafficMode}
    mode_type = modes.get(name)
    if mode_type is None:
        raise ValueError(f"未实现的流量模式：{name}")
    mode = mode_type()
    if mode.risk_level is not RiskLevel.NORMAL:
        raise ValueError("当前入口只允许启动日常安全流量模式")
    return mode


__all__ = ["load_mode", "RiskLevel", "TrafficMode"]
