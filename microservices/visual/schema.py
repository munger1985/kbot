"""视觉嵌入微服务模型定义。"""

from pydantic import BaseModel


class VisualEmbeddingRequest(BaseModel):
    """视觉嵌入请求"""
    model_name: str = ""               # 模型名称（空则取第一个激活模型）
    image_base64: str                  # base64 编码的图片（JPEG/PNG）


class VisualEmbeddingResponse(BaseModel):
    """视觉嵌入响应"""
    embedding: list[float]
    dimension: int
    model_name: str
