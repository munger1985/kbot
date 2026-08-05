"""模型目录与推理服务共享契约。"""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class _ModelContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


ModelLifecycleStatus = Literal["DRAFT", "ACTIVE", "ARCHIVED"]


class ModelProviderOption(_ModelContract):
    category: int = Field(ge=1, le=5)
    provider: str = Field(min_length=1, max_length=64)
    required_fields: tuple[str, ...] = ()
    secret_fields: tuple[str, ...] = ()
    allowed_model_params: tuple[str, ...] = ()
    supports_tool_calling: bool = False
    max_context_tokens: int | None = Field(default=None, ge=1)


class ModelCreateRequest(_ModelContract):
    served_model_name: str = Field(
        min_length=1, max_length=128,
        pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$",
    )
    display_name: str = Field(min_length=1, max_length=256)
    provider_model_name: str = Field(min_length=1, max_length=256)
    category: int = Field(ge=1, le=5)
    provider: str = Field(min_length=1, max_length=64)
    api_endpoint: str | None = Field(default=None, max_length=1024)
    api_key: str | None = Field(default=None, max_length=4096)
    status: ModelLifecycleStatus = "DRAFT"
    model_params: dict[str, Any] = Field(default_factory=dict)
    description: str | None = Field(default=None, max_length=512)


class ModelUpdateRequest(_ModelContract):
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    api_endpoint: str | None = Field(default=None, max_length=1024)
    api_key: str | None = Field(default=None, max_length=4096)
    model_params: dict[str, Any] | None = None
    description: str | None = Field(default=None, max_length=512)


class ModelArchiveRequest(_ModelContract):
    expected_row_version: int = Field(ge=1)


class ModelStatusRequest(_ModelContract):
    expected_row_version: int = Field(ge=1)
    status: Literal["DRAFT", "ACTIVE"]


class ModelDeleteRequest(_ModelContract):
    expected_row_version: int = Field(ge=1)


class ModelCatalogItem(_ModelContract):
    model_id: UUID
    served_model_name: str
    display_name: str
    provider_model_name: str
    category: int = Field(ge=1, le=5)
    provider: str
    api_endpoint: str | None = None
    status: ModelLifecycleStatus
    model_params: dict[str, Any] = Field(default_factory=dict)
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
