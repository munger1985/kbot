import numpy as np
from loguru import logger
from model_pool import ModelPool
from model.base import BaseEmbedding, EmbeddingResponse


class EmbeddingService:
    """
    统一的嵌入服务，用于管理和使用不同的嵌入模型
    """
    
    def __init__(self):
        """
        Initialize embedding service // 初始化嵌入服务
        """
        self._model_pool = ModelPool()
        self._initialized = False
        
    async def initialize(self):
        """
        Initialize embedding service and model pool // 初始化嵌入服务和模型池
        """
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True
            logger.info("Embedding service initialized")
        
    async def shutdown(self):
        """
        Shutdown embedding service and all models // 关闭嵌入服务和所有模型
        """
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False
            logger.info("Embedding service has been shutdown")
    
    async def get_embedding_model(self, model_unique_name: str) -> BaseEmbedding:
        """
        Get a embedding model by ID // 获取指定ID的embedding模型

        Args:
            model_unique_name: The unique name of the model to get // 要获取的模型ID

        Returns:
            embedding model instance // embedding模型实例

        Raises:
            ValueError: If model_unique_name is not found in database // 如果模型ID在数据库中不存在
            RuntimeError: If model creation fails // 如果模型创建失败
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.load_model(model_unique_name)
    
    async def embed_texts(
        self, 
        model_unique_name: str, 
        texts: list[str], 
        batch_size: int = 0
    ) -> EmbeddingResponse:
        """
        Embed texts with the specified model // 使用指定模型对文本列表进行嵌入

        Args:
            model_unique_name: 要使用的嵌入模型ID
            texts: 要嵌入的文本列表
            batch_size: 批处理大小，如果为0则由模型决定

        Returns:
            EmbeddingResponse: 标准OpenAI格式的嵌入响应，包含嵌入向量和token使用情况
        """
        if not texts:
            return EmbeddingResponse(
                data=[],
                model=model_unique_name,
                object="list",
                usage={"prompt_tokens": 0, "total_tokens": 0}
            )
        
        try:
            model = await self.get_embedding_model(model_unique_name)
            response = await model.embed(texts=texts, batch_size=batch_size)
            
            # 确保返回的响应数据有效
            if not response.data or len(response.data) == 0:
                return EmbeddingResponse(
                    data=[],
                    model=model_unique_name,
                    object="list",
                    usage={"prompt_tokens": 0, "total_tokens": 0}
                )
            return response
                
        except Exception as e:
            logger.exception(f"Failed to embed texts, model_unique_name: {model_unique_name}, error: {e}")
            # 如果底层模型返回0-d tensor错误，返回空响应
            if "0-d tensor" in str(e):
                return EmbeddingResponse(
                    data=[],
                    model=model_unique_name,
                    object="list",
                    usage={"prompt_tokens": 0, "total_tokens": 0}
                )
            raise RuntimeError(f"Failed to embed texts: {e}")
    
    async def compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray, 
        method: str = "cosine"
    ) -> float:
        """
        计算两个嵌入向量之间的相似度

        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量
            method: 相似度计算方法，支持 "cosine"（余弦相似度）和 "dot"（点积）

        Returns:
            相似度分数，范围通常在[-1, 1]之间，值越大表示越相似

        Raises:
            ValueError: 如果向量维度不匹配或方法不支持
        """
        if embedding1.shape != embedding2.shape:
            raise ValueError(f"Embedding vectors have different shapes: {embedding1.shape} vs {embedding2.shape}")
        
        # 确保向量是一维的
        vec1 = embedding1.flatten()
        vec2 = embedding2.flatten()
        
        if method.lower() == "cosine":
            # 余弦相似度
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        elif method.lower() == "dot":
            # 点积
            return float(np.dot(vec1, vec2))
        else:
            raise ValueError(f"Unsupported similarity method: {method}")
    
    async def unload_model(self, model_unique_name: str):
        """
        从模型池中卸载模型

        Args:
            model_unique_name: 要卸载的模型ID
        """
        if self._initialized:
            await self._model_pool.unload_model(model_unique_name)
            logger.info(f"Model {model_unique_name} has been unloaded.")
    
    async def reload_model(self, model_unique_name: str) -> BaseEmbedding:
        """
        Reload a model from the pool // 重新加载模型

        Args:
            model_unique_name: The unique name of the model to reload // 要重新加载的模型ID

        Returns:
            The reloaded embedding model instance // 重新加载的嵌入模型实例
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.reload_model(model_unique_name)