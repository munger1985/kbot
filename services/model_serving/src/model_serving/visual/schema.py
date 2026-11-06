"""视觉嵌入微服务模型定义。"""

from pydantic import BaseModel
from pydantic import Field
from typing import Literal


class VisualEmbeddingRequest(BaseModel):
    """视觉嵌入请求"""
    served_model_name: str
    image_base64: str                  # base64 编码的图片（JPEG/PNG）


class VisualEmbeddingResponse(BaseModel):
    """视觉嵌入响应"""
    embedding: list[float]
    dimension: int
    served_model_name: str

class NativeVisualRequest(BaseModel):
    """方向必须来自已批准模型能力，不接受任意向量覆盖。"""
    served_model_name: str
    input_kind: Literal["TEXT", "IMAGE"]
    direction: Literal["TEXT_TO_IMAGE", "IMAGE_TO_IMAGE", "IMAGE_TO_TEXT", "TEXT_TO_TEXT"]
    text: str | None = Field(default=None, max_length=16000)
    image_base64: str | None = Field(default=None, max_length=11184820)
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
