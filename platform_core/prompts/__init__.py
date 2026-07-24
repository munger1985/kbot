"""共享 Prompt Catalog、持久化和解析接口。"""

from .catalog import (
    PromptCatalog,
    PromptCatalogEntry,
    PromptCatalogError,
    load_prompt_catalog,
)
from .entities import PlatformPromptEntity, PlatformPromptVersionEntity
from .repository import PlatformPromptRepository
from .resolver import (
    PromptIntegrityError,
    PromptNotFoundError,
    PromptResolver,
    ResolvedPrompt,
    StrictPromptRenderer,
)

__all__ = [
    "PlatformPromptEntity",
    "PlatformPromptRepository",
    "PlatformPromptVersionEntity",
    "PromptCatalog",
    "PromptCatalogEntry",
    "PromptCatalogError",
    "PromptIntegrityError",
    "PromptNotFoundError",
    "PromptResolver",
    "ResolvedPrompt",
    "StrictPromptRenderer",
    "load_prompt_catalog",
]
