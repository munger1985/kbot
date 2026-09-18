"""证据、假设、根因与解决建议用例。"""

from .findings import compile_findings
from .policy import (
    freeze_automatic_entry_intent,
    is_automatic_entry,
    is_diagnosis_turn,
)

__all__ = [
    "compile_findings",
    "freeze_automatic_entry_intent",
    "is_automatic_entry",
    "is_diagnosis_turn",
]
