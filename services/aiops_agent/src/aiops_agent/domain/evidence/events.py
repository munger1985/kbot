"""外部诊断源信号事件的领域枚举。"""

from enum import StrEnum


class SignalEventStatus(StrEnum):
    FIRING = "FIRING"
    RESOLVED = "RESOLVED"
    INFORMATIONAL = "INFORMATIONAL"


class SignalSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
