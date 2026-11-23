"""模型目录与推理服务共享契约。"""

import re
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.json_schema import SkipJsonSchema


NATIVE_VISUAL_MAX_IMAGE_BYTES = 8 * 1024 * 1024
NATIVE_VISUAL_IMAGE_INPUT_UNSUPPORTED = "IMAGE_INPUT_UNSUPPORTED"


class _ModelContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


ModelLifecycleStatus = Literal["DRAFT", "ACTIVE", "ARCHIVED"]


def is_grok_provider_model_name(value: object) -> bool:
    """根据上游模型标识判断是否为 Grok 模型家族。"""
    normalized = str(value or "").strip().lower()
    if not normalized:
        return False
    parts = [part for part in re.split(r"[./:_]+", normalized) if part]
    return any(part == "grok" or part.startswith("grok-") for part in parts)


class ModelProviderOption(_ModelContract):
    category: int = Field(ge=1, le=8)
    provider: str = Field(min_length=1, max_length=64)
    required_fields: tuple[str, ...] = ()
    secret_fields: tuple[str, ...] = ()
    allowed_model_params: tuple[str, ...] = ()
    supports_tool_calling: bool = False
    max_context_tokens: int | None = Field(default=None, ge=1)


class ModelCreateRequest(_ModelContract):
    display_name: str = Field(min_length=1, max_length=256)
    provider_model_name: str = Field(min_length=1, max_length=256)
    category: int = Field(ge=1, le=8)
    provider: str = Field(min_length=1, max_length=64)
    api_endpoint: str | None = Field(default=None, max_length=1024)
    api_key: str | None = Field(default=None, max_length=4096)
    status: ModelLifecycleStatus = "DRAFT"
    model_params: dict[str, Any] = Field(default_factory=dict)
    description: str | None = Field(default=None, max_length=512)


class ModelUpdateRequest(_ModelContract):
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    provider_model_name: str | SkipJsonSchema[None] = Field(default=None, min_length=1, max_length=256)
    api_endpoint: str | None = Field(default=None, max_length=1024)
    api_key: str | None = Field(default=None, max_length=4096)
    model_params: dict[str, Any] | None = None
    description: str | None = Field(default=None, max_length=512)

    @field_validator("provider_model_name", mode="before")
    @classmethod
    def validate_provider_model_name(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError("厂商模型名称或 ID 不能为空")
        return value.strip() if isinstance(value, str) else value


class ModelArchiveRequest(_ModelContract):
    expected_row_version: int = Field(ge=1)


class ModelStatusRequest(_ModelContract):
    expected_row_version: int = Field(ge=1)
    status: Literal["DRAFT", "ACTIVE"]


class ModelDeleteRequest(_ModelContract):
    expected_row_version: int = Field(ge=1)


class ModelConnectionTestResult(_ModelContract):
    latency_ms: int = Field(ge=0)


class ModelCatalogItem(_ModelContract):
    model_id: UUID
    display_name: str
    provider_model_name: str
    category: int = Field(ge=1, le=8)
    provider: str
    api_endpoint: str | None = None
    status: ModelLifecycleStatus
    model_params: dict[str, Any] = Field(default_factory=dict)
    effective_max_tokens: int | None = Field(default=None, ge=1)
    description: str | None = None
    row_version: int = Field(ge=1)


class ModelReference(_ModelContract):
    service: str
    resource_type: str
    resource_id: str
    usage: str


class ModelReferenceSummary(_ModelContract):
    model_id: UUID
    references: tuple[ModelReference, ...] = ()
    unavailable_services: tuple[str, ...] = ()


class EmbeddingDataItem(BaseModel):
    """One vector returned by the embedding service."""

    embedding: list[float] = Field(..., description="Embedding vector")
    index: int = Field(..., description="Index position in the batch")
    object: str = Field("embedding", description="Object type")
