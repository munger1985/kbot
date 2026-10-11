"""数据库故障模拟入口与安全边界。"""

from aiops_simulator.faults.base import (
    FAULT_TTL_SECONDS,
    SUPPORTED_FAULTS,
    normalize_database_name,
    validate_fault_type,
)
from aiops_simulator.faults.runner import run_fault
from aiops_simulator.faults.state import print_fault_status

__all__ = [
    "FAULT_TTL_SECONDS",
    "SUPPORTED_FAULTS",
    "normalize_database_name",
    "print_fault_status",
    "run_fault",
    "validate_fault_type",
]
