"""运维知识库的公开与内部管理合同。"""

from typing import Annotated, Literal
import re
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from ..knowledge_parsing import LibraryIndexModels, LibraryParseModels


class KnowledgeBaseWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
    description: str | None = Field(default=None, max_length=1000)
    tagging_model_id: UUID | None = None
    parse_models: LibraryParseModels | None = None
    index_models: LibraryIndexModels | None = None
    expected_row_version: int | None = Field(default=None, ge=1)


CapabilityState = Literal["NOT_ENABLED", "PENDING_VERIFICATION", "AVAILABLE", "DEGRADED"]


class KnowledgeSemanticDirection(BaseModel):
    direction: Literal["TEXT_TO_TEXT", "TEXT_TO_IMAGE", "IMAGE_TO_TEXT", "IMAGE_TO_IMAGE"]
    state: CapabilityState


class KnowledgeBaseCapabilities(BaseModel):
    native_parse: CapabilityState
    ocr: CapabilityState
    visual_enrichment: CapabilityState
    lexical: CapabilityState
    semantic_directions: list[KnowledgeSemanticDirection]


class KnowledgeBaseIndexStatus(BaseModel):
    state: Literal["EMPTY", "NOT_READY", "PARTIAL", "READY", "ERROR"]


class KnowledgeBaseTaggingStatus(BaseModel):
    state: Literal["NOT_CONFIGURED", "PENDING_VERIFICATION", "AVAILABLE", "DEGRADED"]
    model_id: UUID | None
    vocabulary_ready: bool


class KnowledgeBaseUsage(BaseModel):
    assets: int = Field(ge=0)
    current_agent_versions: int = Field(ge=0)
    historical_agent_versions: int = Field(ge=0)


KnowledgeBaseBlocker = Literal["ASSET_RECORDS", "CURRENT_AGENT_VERSION", "HISTORICAL_AGENT_VERSION"]


def parse_delete_version(if_match: str) -> int:
    match = re.fullmatch(r'"rv-([1-9][0-9]{0,18})"', if_match)
    if not match:
        raise ValueError('If-Match 必须使用 "rv-N" 格式')
    return int(match[1])


class KnowledgeBaseView(BaseModel):
    # 保留平台已有库标识、审计和使用情况字段；下面是统一的管理能力合同。
    model_config = ConfigDict(extra="allow")
    knowledge_base_id: UUID
    display_name: str
    description: str | None
    status: str
    row_version: int
    capabilities: KnowledgeBaseCapabilities
    index: KnowledgeBaseIndexStatus
    tagging: KnowledgeBaseTaggingStatus
    parse_models: LibraryParseModels | None = None
    index_models: LibraryIndexModels | None = None
    usage: KnowledgeBaseUsage
    blockers: list[KnowledgeBaseBlocker] = Field(description="资料及 Agent 版本引用不阻止归档，当前返回空列表。")


class KnowledgeBaseList(BaseModel):
    items: list[KnowledgeBaseView]
