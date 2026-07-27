"""视觉嵌入微服务模型定义。"""

from pydantic import BaseModel


class VisualEmbeddingRequest(BaseModel):
    """视觉嵌入请求"""
    served_model_name: str
    image_base64: str                  # base64 编码的图片（JPEG/PNG）


class VisualEmbeddingResponse(BaseModel):
    """视觉嵌入响应"""
    embedding: list[float]
    dimension: int
    served_model_name: str
