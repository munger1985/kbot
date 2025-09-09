import cohere

from typing import Any
from pydantic import Field
from loguru import logger

from .base import RerankerConfig, BaseReranker

class CohereRerankerConfig(RerankerConfig):
    """Cohere Reranker 模型配置类"""
    api_endpoint: str = Field("https://api.cohere.ai", description="Cohere API 端点")
    api_key: str = Field(..., description="Cohere API 密钥")
    timeout: int = Field(10, description="API 请求超时时间（秒）")
    
class CohereReranker(BaseReranker):
    """Cohere Reranker 重排器类"""


    def __init__(self, config: CohereRerankerConfig):
        """
        初始化 Cohere 重排器
        
        Args:
            config: 模型配置
        """
        
        self.config = config
        self.client = None

        # 运行时状态
        self._is_initialized = False
            
        logger.info(f"正在初始化 {self.__class__.__name__}，模型: {self.config.model_name}")
    
    
    async def startup(self) -> None:
        """初始化 reranker 模型"""
        if self._is_initialized:
            return

        timeout = self.config.timeout if hasattr(self.config, 'timeout') else 10
        self.client = cohere.Client(api_key=self.config.api_key, timeout=timeout)
        

        self._is_initialized = True
        logger.info(f"Reranker 模型 {self.config.model_name} 初始化成功")
    
    
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        根据与查询的相关性对文档进行重排序
        
        Args:
            query: 搜索查询
            documents: 需要重排序的文档列表
            top_k: 返回的顶部文档数量（None 表示返回所有）
            
        Returns:
            包含 'index' 和 'score' 键的字典列表
        """
        if not self.client:
            raise RuntimeError("模型未初始化，请先调用 startup() 方法")
        
        if not documents:
            return []
        
        # 如果未指定 top_k，则设置为文档数量
        if top_k is None:
            top_k = len(documents)
        else:
            top_k = min(top_k, len(documents))
        
        try:
            response = self.client.rerank(
                    model=self.config.model_name,
                    query=query,
                    documents=documents,
                    top_n=top_k,
            )

            return [{"index": result["index"], "score": float(result["relevance_score"])} for result in response["results"]] # type: ignore
        
        except Exception as e:
            logger.error(f"重排序过程中发生错误: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """清理资源"""
        if self.client:
            self.client = None
            self._is_initialized = False
            logger.info(f"{self.__class__.__name__} 模型资源已释放")