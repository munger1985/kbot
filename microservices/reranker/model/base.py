from typing import Any
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class RerankerConfig(BaseModel):
    """Reranker 模型配置"""
    model_name: str = Field(..., description="Reranker 模型名称")
    provider: str = Field(..., description="Reranker 模型提供者")
    max_tokens: int | None = Field(512, description="最大输入序列长度")

class BaseReranker(ABC):
    """Reranker 模型抽象基类"""
    
    @abstractmethod
    async def startup(self) -> None:
        """初始化 reranker 模型"""
        pass
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        根据与查询的相关性对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 需要重排序的文档列表
            top_k: 返回的顶部文档数量（None 表示返回所有）
            
        Returns:
            包含重排序结果的字典列表
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """清理资源"""
        pass