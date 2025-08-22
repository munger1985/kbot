import os
import cohere

from typing import Any
from pydantic import Field
from loguru import logger

from .base import RerankerConfig, BaseReranker

class CohereRerankerConfig(RerankerConfig):
    """Reranker 模型配置类"""
    model_name: str = Field(..., description="Name of the reranker model")
    api_endpoint: str = Field("https://api.cohere.ai", description="Cohere API endpoint")
    api_key: str | None = Field(None, description="Cohere API key")
    timeout: int = Field(10, description="Timeout for API requests in seconds")
    
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

        # Runtime state
        self._is_initialized = False
            
        logger.info(f"Initializing {self.__class__.__name__} with model: {self.model_name}")
    
    
    async def startup(self) -> None:
        """Initialize the reranker model."""
        if self._is_initialized:
            return

        timeout = self.config.timeout if hasattr(self.config, 'timeout') else 10
        self.client = cohere.ClientV2(timeout=timeout)
        

        self._is_initialized = True
        logger.info(f"Reranker model {self.config.model_name} initialized successfully.")
    
    
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return (None for all)
            
        Returns:
            List of dicts with 'index' and 'score' keys
        """
        if not self.client:
            raise RuntimeError("Model not initialized. Call startup() first.")
        
        if not documents:
            return []
        
        # Set top_k to number of documents if not specified
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

            # # Create list of (index, score) tuples
            # scored_results = [(i, score) for i, score in enumerate(scores)]
            
            # # Sort by score in descending order
            # scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # # Limit to top_k results
            # scored_results = scored_results[:top_k]
            
            # # Return results in requested format
            # return [{"index": idx, "score": float(score)} for idx, score in scored_results]
            return response
        
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Clean up resources."""
        if self.client:

            self.client = None
            self._is_initialized = False
            logger.info(f"{self.__class__.__name__} model resources released")

    def health_check(self) -> dict[str, Any]:
        """Check model health and resource status."""
        status = {
            "initialized": self._is_initialized,
            "model_loaded": self.client is not None,
            "model_name": self.config.model_name
        }
        
        return status