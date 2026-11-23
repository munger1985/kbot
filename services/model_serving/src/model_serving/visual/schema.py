"""视觉嵌入微服务模型定义。"""

from uuid import UUID

from pydantic import BaseModel
from pydantic import Field
from typing import Literal
from platform_core.contracts.model import NATIVE_VISUAL_MAX_IMAGE_BYTES


class VisualEmbeddingRequest(BaseModel):
    """视觉嵌入请求"""
    model_id: UUID
    image_base64: str                  # base64 编码的图片（JPEG/PNG）


class VisualEmbeddingResponse(BaseModel):
    """视觉嵌入响应"""
    embedding: list[float]
    dimension: int
    model_id: UUID

class NativeVisualRequest(BaseModel):
    """方向必须来自已批准模型能力，不接受任意向量覆盖。"""
    model_id: UUID
    input_kind: Literal["TEXT", "IMAGE"]
    direction: Literal["TEXT_TO_IMAGE", "IMAGE_TO_IMAGE", "IMAGE_TO_TEXT", "TEXT_TO_TEXT"]
    expected_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    text: str | None = Field(default=None, max_length=16000)
    image_base64: str | None = Field(default=None, max_length=4 * ((NATIVE_VISUAL_MAX_IMAGE_BYTES + 2) // 3) + 8)
    model_config = {"extra": "forbid"}


class NativeVisualResponse(BaseModel):
    vectors: list[list[float]]
    mask: list[bool]
    dimension: int
    model_revision: str
    processor_revision: str
    input_template: str
    normalization: Literal["L2"]
    similarity: Literal["DOT"]
    directions: list[str]
