import numpy as np
from loguru import logger
from model_pool import ModelPool
from model.base import BaseEmbedding, EmbeddingResponse


class EmbeddingService:
    """统一的嵌入服务，用于管理和使用不同的嵌入模型。"""
    
    def __init__(self):
        """初始化嵌入服务实例。"""
        self._model_pool = ModelPool()
        self._initialized = False
        
    async def initialize(self):
        """初始化嵌入服务和模型池。"""
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True
            logger.info("嵌入服务初始化完成")
        
    async def shutdown(self):
        """关闭嵌入服务和所有模型资源。"""
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False
            logger.info("嵌入服务已关闭")
    
    async def get_embedding_model(self, model_id: int) -> BaseEmbedding:
        """获取指定唯一名称的嵌入模型实例。"""
        if not self._initialized:
            await self.initialize()

        return await self._model_pool.load_model(model_id)

    
    async def embed_texts(
        self, 
        model_id: int, 
        texts: list[str], 
        batch_size: int = 0,
        is_query: bool = True
    ) -> EmbeddingResponse:
        """使用指定模型对文本列表进行嵌入处理。
        
        Args:
            model_id: 嵌入模型唯一名称
            texts: 待嵌入的文本列表
            batch_size: 批处理大小，为0时由模型自动决定
            is_query: 是否为查询文本，默认为True
            
        Returns:
            EmbeddingResponse: 标准OpenAI格式的嵌入响应，包含向量数据和使用情况
            
        Raises:
            RuntimeError: 当嵌入处理过程中发生错误时抛出
        """
        if not texts:
            return EmbeddingResponse(
                data=[],
                model=self._model_pool._model_names.get(model_id, str(model_id)),
                object="list",
                usage={"prompt_tokens": 0, "total_tokens": 0}
            )
        
        try:
            model = await self.get_embedding_model(model_id)
            response = await model.embed(texts=texts, batch_size=batch_size, is_query=is_query)
            
            # 验证返回的响应数据有效性
            if not response.data or len(response.data) == 0:
                return EmbeddingResponse(
                    data=[],
                    model=self._model_pool._model_names.get(model_id, str(model_id)),
                    object="list",
                    usage={"prompt_tokens": 0, "total_tokens": 0}
                )
            return response
                
        except Exception as e:
            logger.exception(f"文本嵌入处理失败，模型: {self._model_pool._model_names.get(model_id, str(model_id))}, 错误: {e}")
            # 处理底层模型返回的0维张量错误
            if "0-d tensor" in str(e):
                return EmbeddingResponse(
                    data=[],
                    model=self._model_pool._model_names.get(model_id, str(model_id)),
                    object="list",
                    usage={"prompt_tokens": 0, "total_tokens": 0}
                )
            raise RuntimeError(f"文本嵌入处理失败: {e}")
    
    async def compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray, 
        method: str = "cosine"
    ) -> float:
        """计算两个嵌入向量之间的相似度分数。
        
        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量
            method: 相似度计算方法，支持"cosine"(余弦相似度)和"dot"(点积)
            
        Returns:
            float: 相似度分数，范围通常在[-1, 1]之间，值越大表示越相似
            
        Raises:
            ValueError: 当向量维度不匹配或方法不支持时抛出
        """
        if embedding1.shape != embedding2.shape:
            raise ValueError(f"嵌入向量维度不匹配: {embedding1.shape} 与 {embedding2.shape}")
        
        # 确保向量是一维的
        vec1 = embedding1.flatten()
        vec2 = embedding2.flatten()
        
        if method.lower() == "cosine":
            # 计算余弦相似度
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        elif method.lower() == "dot":
            # 计算点积
            return float(np.dot(vec1, vec2))
        else:
            raise ValueError(f"不支持的相似度计算方法: {method}")
    
    async def warmup(self):
        """预加载所有模型到内存中进行预热。
        
        Raises:
            Exception: 预热过程中发生错误时抛出
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