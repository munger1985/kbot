import os
import sys
from typing import List, Dict, Any, Optional, Union
import numpy as np
from loguru import logger

# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
    
from microservices.embedding.model_pool import ModelPool
from models.embedding.base import BaseEmbedding


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
    
    async def get_embedding_model(self, model_id: int) -> BaseEmbedding:
        """
        Get a embedding model by ID // 获取指定ID的embedding模型

        Args:
            model_id: The ID of the model to get // 要获取的模型ID

        Returns:
            embedding model instance // embedding模型实例

        Raises:
            ValueError: If model_id is not found in database // 如果模型ID在数据库中不存在
            RuntimeError: If model creation fails // 如果模型创建失败
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.load_model(model_id)
    
    async def embed_texts(
        self, 
        model_id: int, 
        texts: List[str], 
        batch_size: int = 0
    ) -> np.ndarray:
        """
        Embed texts with the specified model // 使用指定模型对文本列表进行嵌入

        Args:
            model_id: 要使用的嵌入模型ID
            texts: 要嵌入的文本列表
            batch_size: 批处理大小，如果为0则由模型决定

        Returns:
            Embeddings array, each row corresponds to an embedding vector for the input text.
            嵌入向量数组，每行对应一个输入文本的嵌入向量。

        """
        if not texts:
            return np.array([])
        
        try:
            model = await self.get_embedding_model(model_id)
            
            # 如果指定了批处理大小，则分批处理
            if batch_size > 0 and len(texts) > batch_size:
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = await model.embed(batch_texts)
                    if isinstance(batch_embeddings, list):
                        batch_embeddings = np.array(batch_embeddings)
                    all_embeddings.append(batch_embeddings)
                return np.vstack(all_embeddings)
            else:
                # 直接处理所有文本
                embeddings = await model.embed(texts)
                return embeddings
                
        except Exception as e:
            logger.error(f"Failed to embed texts, model_id: {model_id}, error: {e}")
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
    
    async def unload_model(self, model_id: int):
        """
        从模型池中卸载模型

        Args:
            model_id: 要卸载的模型ID
        """
        if self._initialized:
            await self._model_pool.unload_model(model_id)
            logger.info(f"Model {model_id} has been unloaded.")
    
    async def reload_model(self, model_id: int) -> BaseEmbedding:
        """
        Reload a model from the pool // 重新加载模型

        Args:
            model_id: The ID of the model to reload // 要重新加载的模型ID

        Returns:
            The reloaded embedding model instance // 重新加载的嵌入模型实例
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.reload_model(model_id)