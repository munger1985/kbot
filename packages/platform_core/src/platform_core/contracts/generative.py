"""研究检索和图片生成的供应商中立内部契约。"""

from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _GenerativeContract(BaseModel):
    """禁止把供应商专有参数、凭据或未约束字段透传到下游。"""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ResearchRequest(_GenerativeContract):
    """由业务 App 发给模型服务的一次受控外部研究请求。"""

    schema_version: Literal["ResearchRequest.v1"] = "ResearchRequest.v1"
    model_id: UUID
    input: str = Field(min_length=1, max_length=32000)
    from_date: date | None = None
    to_date: date | None = None
    allowed_x_handles: tuple[str, ...] = Field(default=(), max_length=20)
    excluded_x_handles: tuple[str, ...] = Field(default=(), max_length=20)
    enable_image_understanding: bool = False
    enable_video_understanding: bool = False
    search_context_size: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    trace_id: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_source_filters(self) -> "ResearchRequest":
        """保持 X 原厂账号过滤的互斥规则，并避免无意义的日期范围。"""
        if self.allowed_x_handles and self.excluded_x_handles:
            raise ValueError("allowed_x_handles 与 excluded_x_handles 不能同时设置")
        if len(set(self.allowed_x_handles)) != len(self.allowed_x_handles):
            raise ValueError("allowed_x_handles 不能包含重复账号")
        if len(set(self.excluded_x_handles)) != len(self.excluded_x_handles):
            raise ValueError("excluded_x_handles 不能包含重复账号")
        if self.from_date and self.to_date and self.from_date > self.to_date:
            raise ValueError("from_date 不能晚于 to_date")
        return self


class ResearchCitation(_GenerativeContract):
    """外部来源的最小可审计投影，不保存不可信的完整原文。"""

    provider_citation_id: str = Field(min_length=1, max_length=256)
    canonical_url: str = Field(min_length=1, max_length=2048)
    title: str | None = Field(default=None, max_length=512)
    excerpt: str | None = Field(default=None, max_length=4000)
    author_handle: str | None = Field(default=None, max_length=128)
    published_at: datetime | None = None


class ModelUsage(_GenerativeContract):
    """供应商返回的可审计用量投影，缺失字段保持为空而非推测。"""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    provider_usage: dict[str, int | float | str] = Field(default_factory=dict)


class ResearchResult(_GenerativeContract):
    """模型服务返回给业务 App 的外部研究终态。"""

    schema_version: Literal["ResearchResult.v1"] = "ResearchResult.v1"
    status: Literal["COMPLETED", "FAILED"]
    answer: str | None = Field(default=None, max_length=128000)
    citations: tuple[ResearchCitation, ...] = ()
    usage: ModelUsage | None = None
    provider_request_id: str | None = Field(default=None, max_length=256)
    error_code: str | None = Field(default=None, max_length=128)


class ImageGenerationRequest(_GenerativeContract):
    """由业务 App 发给模型服务的一次受控文生图请求。"""

    schema_version: Literal["ImageGenerationRequest.v1"] = "ImageGenerationRequest.v1"
    model_id: UUID
    prompt: str = Field(min_length=1, max_length=32000)
    aspect_ratio: str = Field(default="1:1", pattern=r"^[1-9][0-9]?:[1-9][0-9]?$")
    count: int = Field(default=1, ge=1, le=4)
    trace_id: str = Field(min_length=1, max_length=128)


class GeneratedImageArtifact(_GenerativeContract):
    """仅在服务间传输图片字节；业务 App 必须立即转存至对象存储。"""

    provider_artifact_id: str | None = Field(default=None, max_length=256)
    mime_type: str = Field(pattern=r"^image/(png|jpeg|webp)$")
    content: bytes = Field(min_length=1, max_length=32 * 1024 * 1024)
    width: int | None = Field(default=None, ge=1, le=16384)
    height: int | None = Field(default=None, ge=1, le=16384)


class ImageGenerationResult(_GenerativeContract):
    """模型服务返回给业务 App 的文生图终态。"""

    schema_version: Literal["ImageGenerationResult.v1"] = "ImageGenerationResult.v1"
    status: Literal["COMPLETED", "FAILED"]
    artifacts: tuple[GeneratedImageArtifact, ...] = ()
    usage: ModelUsage | None = None
    provider_request_id: str | None = Field(default=None, max_length=256)
    error_code: str | None = Field(default=None, max_length=128)
