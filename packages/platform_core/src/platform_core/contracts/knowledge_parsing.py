"""知识库模型绑定与文档解析档位的共享产品合同。"""

from collections.abc import Mapping
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

ParseStrategy = Literal["TEXT", "AUTO", "VISUAL"]
ParseModelRole = Literal["ocr", "vlm"]
REQUIRED_PARSE_MODELS: Mapping[ParseStrategy, tuple[ParseModelRole, ...]] = {
    "TEXT": (), "AUTO": ("ocr",), "VISUAL": ("ocr", "vlm"),
}


class LibraryParseModels(BaseModel):
    """角色可以不绑定；可选档位由审批时的有效绑定决定。"""

    model_config = ConfigDict(extra="forbid")
    ocr: UUID | None = None
    vlm: UUID | None = None


class LibraryIndexModels(BaseModel):
    """文本与视觉表示独立于解析档位；目录模型身份由各库保存。"""

    model_config = ConfigDict(extra="forbid")
    embedding: UUID | None = None
    visual_embedding: UUID | None = None


class ParseStrategyOption(BaseModel):
    strategy: ParseStrategy
    selectable: bool
    unavailable_roles: tuple[ParseModelRole, ...] = ()
    executor_unavailable: bool = False


def parse_strategy_options(available: Mapping[ParseModelRole, object], *, executor_ready=True) -> list[ParseStrategyOption]:
    """只传已授权、类型匹配且可用的模型；配置存在不等于可用。"""

    options = []
    for strategy, required in REQUIRED_PARSE_MODELS.items():
        missing = tuple(role for role in required if not available.get(role))
        unavailable = bool(required) and not executor_ready
        options.append(ParseStrategyOption(strategy=strategy, selectable=not missing and not unavailable,
            unavailable_roles=missing, executor_unavailable=unavailable))
    return options
