"""Oracle/MySQL 只读 Driver 注册。"""

from .base import (
    DiagnosticDriverError,
    DriverQueryResult,
    ReadonlyDatabaseDriver,
)
from .mysql import MySQLDiagnosticDriver
from .oracle import OracleDiagnosticDriver

__all__ = [
    "DiagnosticDriverError",
    "DriverQueryResult",
    "MySQLDiagnosticDriver",
    "OracleDiagnosticDriver",
    "ReadonlyDatabaseDriver",
]
