"""AIOps 独立拥有的诊断源 Adapter。"""

from .registry import (
    DiagnosticSourceAdapterCatalog,
    DiagnosticSourceAdapterRegistry,
)

__all__ = [
    "DiagnosticSourceAdapterCatalog",
    "DiagnosticSourceAdapterRegistry",
]
