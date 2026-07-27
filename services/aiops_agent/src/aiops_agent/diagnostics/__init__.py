"""Oracle/MySQL 版本化只读诊断目录。"""

from .registry import (
    DiagnosticRegistry,
    ResolvedDiagnosticTool,
    database_major_version,
)
from .runtime import (
    create_diagnostic_grant_codec,
    create_diagnostic_registry,
)

__all__ = [
    "DiagnosticRegistry",
    "ResolvedDiagnosticTool",
    "database_major_version",
    "create_diagnostic_grant_codec",
    "create_diagnostic_registry",
]
