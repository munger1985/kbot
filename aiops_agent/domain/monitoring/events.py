"""监控事件的领域枚举。"""

from enum import StrEnum


class MonitorEventStatus(StrEnum):
    FIRING = "FIRING"
    RESOLVED = "RESOLVED"
    INFORMATIONAL = "INFORMATIONAL"


class MonitorSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
