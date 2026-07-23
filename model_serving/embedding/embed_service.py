import numpy as np
from typing import Callable
from loguru import logger
from .model_pool import EmbeddingModelPool
from .model import BaseEmbedding, EmbeddingResponse


class EmbeddingService:
    """Unified embedding service for managing and utilizing different embedding models."""
    
    def __init__(self):
        """Initialize EmbeddingService instance."""
        self._model_pool = EmbeddingModelPool()
        self._initialized = False

    def bind_session_factory(self, session_factory: Callable) -> None:
        self._model_pool.set_session_factory(session_factory)
        
    async def initialize(self):
        """Initialize embedding service and model pool."""
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True
            logger.info("Embedding 服务初始化成功")
        
    async def shutdown(self):
        """Shutdown embedding service and release all model resources."""
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False
            logger.info("Embedding 服务已停止")
    
    async def get_embedding_model(self, model_name: str) -> BaseEmbedding:
        """Get embedding model instance by its unique name."""
        if not self._initialized:
            await self.initialize()

        return await self._model_pool.load_model(model_name)

    async def embed_texts(
        self, 
        model_name: str, 
        texts: list[str], 
        batch_size: int | None = None,
        is_query: bool = True
    ) -> EmbeddingResponse:
        """Generate embeddings for a list of texts using the specified model.
        
        Args:
            model_name: Unique name of the embedding model
            texts: List of texts to generate embeddings for
            batch_size: Batch size for processing (model-determined if None)
            is_query: Whether the texts are query inputs (default: True)
            
        Returns:
            EmbeddingResponse: Standard OpenAI-formatted embedding response containing
                               vector data and token usage statistics
            
        Raises:
            RuntimeError: Raised when errors occur during embedding processing
        """
        if not texts:
            return EmbeddingResponse(
                data=[],
                model=model_name,
                object="list",
                usage={"prompt_tokens": 0, "total_tokens": 0}
            )
        
        try:
            model = await self.get_embedding_model(model_name)
            response = await model.embed(texts=texts, batch_size=batch_size, is_query=is_query)
            
            # Validate response data validity
            if not response.data or len(response.data) == 0:
                return EmbeddingResponse(
                    data=[],
                    model=model_name,
                    object="list",
                    usage={"prompt_tokens": 0, "total_tokens": 0}
                )
            return response
                
        except Exception as e:
            logger.exception(f"Text embedding processing failed, model: {model_name}, error: {e}")
            # Handle 0-dimensional tensor errors returned by underlying models
            if "0-d tensor" in str(e):
                return EmbeddingResponse(
                    data=[],
                    model=model_name,
                    object="list",
                    usage={"prompt_tokens": 0, "total_tokens": 0}
                )
            raise RuntimeError(f"Text embedding processing failed: {e}")
    
    async def compute_similarity(
        self,
        model_name: str,
        text1: str, 
        text2: str, 
        method: str = "cosine"
    ) -> float:
        """Compute similarity score between two embedding vectors.
        
        Args:
            model_name: Unique name of the embedding model
            text1: First text string
            text2: Second text string
            method: Similarity calculation method - supports "cosine" (cosine similarity)
                    and "dot" (dot product)
            
        Returns:
            float: Similarity score (typically in range [-1, 1]), higher values indicate
                   greater similarity
            
        Raises:
            ValueError: Raised when vector dimensions mismatch or unsupported method is used
        """
        # Get embeddings for both texts
        embed_texts = [text1, text2]
        response = await self.embed_texts(model_name, embed_texts, batch_size=2)
        
        # Check if response data is empty
        if not response.data or len(response.data) < 2:
            logger.error(f"Embedding 响应无效：预期 2 个向量，实际为 {len(response.data) if response.data else 0}")
            raise ValueError(f"Failed to get text embedding vectors, possibly due to model error or CUDA issues")
        
        # Extract embedding vectors
        embedding1 = np.array(response.data[0].embedding)
        embedding2 = np.array(response.data[1].embedding)
        
        # Validate vector dimension matching
        if embedding1.shape != embedding2.shape:
            raise ValueError(f"Embedding vector dimensions mismatch: {embedding1.shape} vs {embedding2.shape}")
        
        # Ensure vectors are 1-dimensional
        vec1 = embedding1.flatten()
        vec2 = embedding2.flatten()
        
        if method.lower() == "cosine":
            # Calculate cosine similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        elif method.lower() == "dot":
            # Calculate dot product
            return float(np.dot(vec1, vec2))
        else:
            raise ValueError(f"Unsupported similarity calculation method: {method}")
    
    async def warmup(self):
        """Preload all models into memory for warmup.
        
        Raises:
            Exception: Raised when errors occur during warmup process
        """
        if not self._initialized:
            await self.initialize()
        
        await self._model_pool.warmup()

    async def load_model(self, model_name: str) -> bool:
        """Load model into memory by its unique identifier
        
        Args:
            model_name: Unique identifier of the model
            
        Returns:
            bool: Whether the model was loaded successfully
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.reload_model(model_name)

        
    async def unload_model(self, model_name: str) -> bool:
        """Unload model from memory by its unique identifier.
        
        Args:
            model_name: Unique identifier of the model
            
        Returns:
            bool: Whether the model was unloaded successfully
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.unload_model(model_name)
