from pydantic import BaseModel, Field

# 定义嵌入请求模型
class EmbeddingRequest(BaseModel):
    """嵌入请求参数模型。"""
    
    model_id: int = Field(..., description="模型唯一标识符")
    texts: list[str] = Field(..., description="待嵌入的文本列表")
    batch_size: int | None = Field(32, description="批处理大小")
    is_query: bool = Field(True, description="是否为查询文本")

class ToggleModelRequest(BaseModel):
    """启用或禁用模型请求表单。"""
    model_id: int = Field(..., description="模型唯一标识符")
    operation: str = Field(..., description="操作类型，'load' 或 'unload'")