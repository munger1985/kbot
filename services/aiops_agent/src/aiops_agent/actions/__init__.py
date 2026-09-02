"""版本化数据库 Action Catalog 与确定性命令渲染。"""

from .contracts import (
    ActionParameter,
    ActionTemplateDefinition,
    RenderedAction,
    ResolvedActionTemplate,
)
from .registry import ActionRegistry, DESTRUCTIVE_EFFECT_CLASSES
from .rendering import ActionRenderer
from .grants import MutationGrantCodec, MutationGrantError
from .runtime import create_mutation_grant_codec
from .compiler import ActionCompilerRegistry, CompiledActionParameters

__all__ = [
    "ActionParameter",
    "ActionRegistry",
    "ActionRenderer",
    "DESTRUCTIVE_EFFECT_CLASSES",
    "MutationGrantCodec",
    "MutationGrantError",
    "create_mutation_grant_codec",
    "ActionTemplateDefinition",
    "RenderedAction",
    "ResolvedActionTemplate",
    "ActionCompilerRegistry",
    "CompiledActionParameters",
]
