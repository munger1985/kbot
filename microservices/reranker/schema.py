from pydantic import BaseModel, Field
from typing import Any


# 定义请求模型
class RerankerRequest(BaseModel):
    model_name: str = Field(..., description="Reranker 模型唯一名称")
    query: str = Field(..., description="查询文本")
    documents: list[str] = Field(..., description="需要重排序的文档列表")
    top_k: int | None = Field(10, description="返回的顶部文档数量（None 表示返回所有）")

class ToggleModelRequest(BaseModel):
    """启用或禁用模型请求表单。"""
    model_name: str = Field(..., description="模型唯一名称")
    operation: str = Field(..., description="操作类型，'load' 或 'unload'")

# 定义响应模型
class RerankerResponse(BaseModel):
    rerankers: list[dict[str, Any]] = Field(..., description="重排序后的文档列表")
