from typing import Any
from loguru import logger
from .model_pool import RerankerModelPool
from .model import BaseReranker


class RerankerService:
    """
    统一的 reranker 服务，用于管理和使用不同的 reranker 模型
    """
    
    def __init__(self):
        """
        初始化 reranker 服务
        """
        self._model_pool = RerankerModelPool()
        self._initialized = False
        
    async def initialize(self):
        """初始化 reranker 服务和模型池。"""
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True
            logger.info("Reranker 服务初始化完成")
        
    async def shutdown(self):
        """关闭 reranker 服务和所有模型。"""
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False
            logger.info("Reranker 服务已关闭")
    
    async def get_reranker_model(self, model_id: int) -> BaseReranker:
        """通过唯一名称获取 reranker 模型。"""
        if not self._initialized:
            await self.initialize()

        return await self._model_pool.load_model(model_id)
    
    async def rerank(
        self,
        model_id: int,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        根据与查询的相关性对文档进行重排序
        
        Args:
            model_id: 模型唯一标识符
            query: 搜索查询
            documents: 需要重排序的文档列表
            top_k: 返回的顶部文档数量（None 表示返回所有）
            
        Returns:
            包含 'index' 和 'score' 键的字典列表
        """
        if not documents:
            return []
        
        try:
            model = await self.get_reranker_model(model_id)
            return await model.rerank(query, documents, top_k)
                
        except Exception as e:
            logger.error(f"文档重排序失败: {e}")
            raise RuntimeError("文档重排序失败") from e
        
    async def warmup(self):
        """
        预热模型池中的所有模型
        """
        if not self._initialized:
            await self.initialize()
        
        await self._model_pool.warmup()

    async def load_model(self, model_id: int) -> bool:
        """通过模型唯一标识符加载模型到内存中
        
        Args:
            model_id: 模型唯一标识符
            
        Returns:
            bool: 加载是否成功
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.reload_model(model_id)

        
    async def unload_model(self, model_id: int) -> bool:
        """通过模型唯一标识符卸载模型到内存中。
        
        Args:
            model_id: 模型唯一标识符
            
        Returns:
            bool: 卸载是否成功
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.unload_model(model_id)