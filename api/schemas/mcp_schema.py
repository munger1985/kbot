from pydantic import BaseModel, Field
from typing import Any
from core.dictionary import KBSearchType


class KBSearchRequest(BaseModel):
    """知识库搜索请求"""
    vector_search_question: str = Field(..., description="语义搜索问题")
    full_text_question: list[str] = Field(..., description="全文搜索问题列表")
    security: int = Field(..., description="安全级别", ge=0)
    tags: list[str]|None = Field(None, description="标签列表")
    search_type: KBSearchType = Field(..., description="搜索类型")
    tool_id: str = Field(..., description="知识库工具ID")
    threshold: float = Field(0.7, description="相似度阈值", ge=0.0, le=1.0)
    top_k: int = Field(10, description="返回结果数量", ge=1, le=100)
    tool_weight: float = Field(1.0, description="工具权重", ge=0.0, le=1.0)


class KBResultResponse(BaseModel):
    """知识库搜索结果响应"""
    file_id: str
    chunk_type: int
    page_num: int
    content: str
    similarity: float
    weight: float


class KBSearchResponse(BaseModel):
    """知识库搜索响应"""
    success: bool
    results: list[KBResultResponse]
    total_count: int
    message: str|None = None


class StreamChunk(BaseModel):
    """流式响应数据块"""
    type: str = Field(..., description="数据类型: result|complete|error")
    data: dict[str, Any]|None = None
    message: str|None = None
    total: int|None = None
    current: int|None = None