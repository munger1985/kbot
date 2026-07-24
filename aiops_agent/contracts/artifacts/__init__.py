"""AIOps Artifact 类型化内容。"""

from .kernel import (
    KernelReport,
    ObservationSet,
    ScopeResult,
)
from .monitoring import (
    MonitorObservationSet,
    MonitorScopeResult,
    ObserveReport,
)
from .database import (
    DatabaseDiagnosticReport,
    DatabaseDiagnosticResult,
    DatabaseObservationAggregate,
    DatabaseScopeResult,
    EvidenceGap,
)

__all__ = [
    "KernelReport",
    "MonitorObservationSet",
    "MonitorScopeResult",
    "ObservationSet",
    "ObserveReport",
    "ScopeResult",
    "DatabaseDiagnosticReport",
    "DatabaseDiagnosticResult",
    "DatabaseObservationAggregate",
    "DatabaseScopeResult",
    "EvidenceGap",
]
