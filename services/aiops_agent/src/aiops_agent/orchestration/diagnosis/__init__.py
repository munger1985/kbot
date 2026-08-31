"""AIOps 数据库 Prompt 资产。"""

from .prompts import (
    DIAGNOSIS_PROMPT_IDS,
    TURN_PROMPT_IDS,
    AIOpsPromptRegistry,
    PromptAsset,
)

__all__ = [
    "AIOpsPromptRegistry",
    "DIAGNOSIS_PROMPT_IDS",
    "PromptAsset",
    "TURN_PROMPT_IDS",
]
