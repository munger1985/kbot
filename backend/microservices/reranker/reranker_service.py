import os
import sys
from typing import List, Dict, Any, Optional
from loguru import logger

# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
    
from microservices.reranker.model_pool import ModelPool
from models.reranker.base import BaseReranker


class RerankerService:
    """
    统一的reranker服务，用于管理和使用不同的reranker模型
    """
    
    def __init__(self):
        """
        初始化reranker服务
        """
        self._model_pool = ModelPool()
        self._initialized = False
        
    async def initialize(self):
        """
        初始化reranker服务和模型池
        """
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True
            logger.info("Reranker service initialized")
        
    async def shutdown(self):
        """
        关闭reranker服务和所有模型
        """
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False
            logger.info("Reranker service has been shutdown")
    
    async def get_reranker_model(self, model_id: int) -> BaseReranker:
        """
        Get a reranker model by ID // 获取指定ID的reranker模型

        Args:
            model_id: The ID of the model to get // 要获取的模型ID

        Returns:
            Reranker model instance // Reranker模型实例

        Raises:
            ValueError: If model_id is not found in database // 如果模型ID在数据库中不存在
            RuntimeError: If model creation fails // 如果模型创建失败
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.load_model(model_id)
    
    async def rerank(
        self,
        model_id: int,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return (None for all)
            return_scores: Whether to return scores with indices
            
        Returns:
            List of dicts with 'index' and 'score' keys
        """
        if not documents:
            return []
        
        try:
            model = await self.get_reranker_model(model_id)
            return await model.rerank(query, documents, top_k)
                
        except Exception as e:
            logger.error(f"Failed to rerank documents: {e}")
            raise RuntimeError("Failed to rerank documents") from e
    
    
    async def unload_model(self, model_id: int):
        """
        Unload a model from the pool // 从模型池中卸载模型

        Args:
            model_id: The ID of the model to unload // 要卸载的模型ID
        """
        if self._initialized:
            await self._model_pool.unload_model(model_id)
            logger.info(f"Model {model_id} has been unloaded.")
    
    async def reload_model(self, model_id: int) -> BaseReranker:
        """
        Reload a model from the pool // 重新加载模型

        Args:
            model_id: 要重新加载的模型ID

        Returns:
            The reloaded reranker model instance // 重新加载的reranker模型实例
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.reload_model(model_id)