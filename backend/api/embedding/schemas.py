from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class EmbeddingRequest(BaseModel):
    """嵌入请求模型"""
    texts: List[str] = Field(..., description="要生成嵌入的文本列表")
    model_id: str = Field(..., description="要使用的模型ID")

class EmbeddingResponse(BaseModel):
    """嵌入响应模型"""
    embeddings: List[List[float]] = Field(..., description="生成的嵌入向量列表")
    model_id: str = Field(..., description="使用的模型ID")
    dimensions: int = Field(..., description="嵌入向量的维度")

class ModelConfigUpdate(BaseModel):
    """模型配置更新请求"""
    config: Dict[str, Any] = Field(..., description="模型配置")
    version: Optional[str] = Field(None, description="配置版本，用于版本控制")

class ModelConfigUpdateResponse(BaseModel):
    """模型配置更新响应"""
    updated: bool = Field(..., description="配置是否已更新")
    model_id: str = Field(..., description="模型ID")

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态：healthy, degraded, unhealthy")
    timestamp: str = Field(..., description="检查时间戳")
    models: Dict[str, Any] = Field(..., description="各模型的健康状态")

class StatsResponse(BaseModel):
    """统计信息响应"""
    models: Dict[str, Dict[str, Any]] = Field(..., description="各模型的使用统计")
    instance_id: str = Field(..., description="服务实例ID")
    cpu_usage: float = Field(..., description="CPU使用率")
    memory_usage: float = Field(..., description="内存使用率")