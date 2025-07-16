from typing import List, Dict, Any, Optional, Union
import numpy as np
from loguru import logger

from services.embedding.model_pool import ModelPool
from models.embedding.base import BaseEmbedding


class EmbeddingService:
    """
    统一的嵌入服务，用于管理和使用不同的嵌入模型
    """
    
    def __init__(self):
        """
        初始化嵌入服务
        """
        self._model_pool = ModelPool()
        self._initialized = False
        
    async def initialize(self):
        """
        初始化嵌入服务和模型池
        """
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True
            logger.info("嵌入服务已初始化")
        
    async def shutdown(self):
        """
        关闭嵌入服务和所有模型
        """
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False
            logger.info("嵌入服务已关闭")
    
    async def get_embedding_model(self, model_id: int) -> BaseEmbedding:
        """
        获取指定ID的嵌入模型

        Args:
            model_id: 模型ID

        Returns:
            嵌入模型实例

        Raises:
            ValueError: 如果模型ID在数据库中不存在
            RuntimeError: 如果模型创建失败
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
        使用指定模型对文本列表进行嵌入

        Args:
            model_id: 要使用的嵌入模型ID
            texts: 要嵌入的文本列表
            batch_size: 批处理大小，如果为0则由模型决定

        Returns:
            嵌入向量数组，每行对应一个输入文本的嵌入向量

        Raises:
            ValueError: 如果模型ID在数据库中不存在
            RuntimeError: 如果嵌入过程失败
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
            logger.error(f"嵌入文本失败，模型ID: {model_id}, 错误: {e}")
            raise RuntimeError(f"嵌入文本失败: {e}")
    
    async def embed_query(self, model_id: int, query: str) -> np.ndarray:
        """
        嵌入单个查询文本

        Args:
            model_id: 要使用的嵌入模型ID
            query: 要嵌入的查询文本

        Returns:
            查询文本的嵌入向量

        Raises:
            ValueError: 如果模型ID在数据库中不存在
            RuntimeError: 如果嵌入过程失败
        """
        embeddings = await self.embed_texts(model_id, [query])
        if embeddings.size == 0:
            raise RuntimeError("嵌入查询失败: 返回了空的嵌入向量")
        return embeddings[0]
    
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
            raise ValueError(f"嵌入向量维度不匹配: {embedding1.shape} vs {embedding2.shape}")
        
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
            raise ValueError(f"不支持的相似度计算方法: {method}")
    
    async def unload_model(self, model_id: int):
        """
        从模型池中卸载模型

        Args:
            model_id: 要卸载的模型ID
        """
        if self._initialized:
            await self._model_pool.unload_model(model_id)
            logger.info(f"已卸载模型 {model_id}")
    
    async def reload_model(self, model_id: int) -> BaseEmbedding:
        """
        重新加载模型

        Args:
            model_id: 要重新加载的模型ID

        Returns:
            重新加载的嵌入模型实例
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.reload_model(model_id)


# 创建全局嵌入服务实例
embedding_service = EmbeddingService()