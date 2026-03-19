from typing import Any
from loguru import logger
from .model_pool import RerankerModelPool
from .model import BaseReranker


class RerankerService:
    """
    Unified reranker service for managing and utilizing different reranker models.
    
    This service provides a consistent interface for:
    - Model pool initialization and lifecycle management
    - Document reranking with different models
    - Model warmup, loading, and unloading operations
    - Centralized error handling and logging
    
    All operations are asynchronous to prevent blocking the event loop and
    ensure compatibility with async application frameworks.
    """
    
    def __init__(self):
        """
        Initialize reranker service with empty model pool.
        
        Creates a model pool instance but does not initialize it immediately -
        initialization is deferred to the initialize() method for better
        application startup control.
        """
        self._model_pool = RerankerModelPool()
        self._initialized = False
        
    async def initialize(self):
        """Initialize reranker service and underlying model pool.
        
        Performs one-time initialization of the model pool, loading configuration
        and preparing for model management operations. Idempotent - safe to call
        multiple times (only initializes once).
        """
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True
            logger.info("Reranker service initialized successfully")
        
    async def shutdown(self):
        """Shut down reranker service and all managed models.
        
        Cleans up resources by shutting down all loaded models and the model pool.
        Resets initialization state to allow re-initialization if needed.
        """
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False
            logger.info("Reranker service shut down successfully")
    
    async def get_reranker_model(self, model_name: str) -> BaseReranker:
        """Retrieve a reranker model instance by its unique technical name.
        
        Ensures service is initialized before attempting to load the model.
        Loads the model into memory if it's not already loaded.
        
        Args:
            model_name: Unique technical name of the reranker model
            
        Returns:
            BaseReranker: Initialized reranker model instance
        """
        if not self._initialized:
            await self.initialize()

        return await self._model_pool.load_model(model_name)
    
    async def rerank(
        self,
        model_name: str,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Rerank documents by relevance to the given query using specified model.
        
        Provides a high-level interface for document reranking with centralized
        error handling and logging. Returns empty list for empty document input
        to prevent unnecessary model calls.
        
        Args:
            model_name: Technical name of the reranker model to use
            query: Search query text to measure relevance against
            documents: List of document texts to be reranked
            top_k: Number of top relevant documents to return (None returns all)
            
        Returns:
            list[dict[str, Any]]: List of reranked results sorted by relevance score (descending),
                each containing at minimum:
                - "index": Original position in the input documents list
                - "score": Relevance score (higher = more relevant)
                
        Raises:
            RuntimeError: If reranking operation fails (wraps original exception)
        """
        # Return empty list immediately for empty input to avoid model calls
        if not documents:
            return []
        
        try:
            # Get or load the requested model
            model = await self.get_reranker_model(model_name)
            # Perform reranking with the model
            return await model.rerank(query, documents, top_k)
                
        except Exception as e:
            # Log detailed error and raise user-friendly exception
            logger.error(f"Document reranking failed with model {model_name}: {e}")
            # Preserve original exception context with 'from e'
            raise RuntimeError("Failed to rerank documents") from e
        
    async def warmup(self):
        """
        Warm up all models in the model pool.
        
        Initializes all configured models and performs health checks to ensure
        they're ready for immediate use, eliminating cold start latency for
        first requests.
        """
        if not self._initialized:
            await self.initialize()
        
        await self._model_pool.warmup()

    async def load_model(self, model_name: str) -> bool:
        """Load a specific model into memory by its technical name.
        
        Reloads the model even if it's already loaded (useful for configuration
        changes). Ensures service is initialized before operation.
        
        Args:
            model_name: Technical name of the model to load
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.reload_model(model_name)

        
    async def unload_model(self, model_name: str) -> bool:
        """Unload a specific model from memory by its technical name.
        
        Releases resources used by the model (GPU/CPU memory, connections, etc.).
        Ensures service is initialized before operation.
        
        Args:
            model_name: Technical name of the model to unload
            
        Returns:
            bool: True if model unloaded successfully, False otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.unload_model(model_name)