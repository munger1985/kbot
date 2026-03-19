import asyncio
from typing import Any
from pydantic import Field
from loguru import logger
from cohere import AsyncClient

from .base import RerankerConfig, BaseReranker

class CohereRerankerConfig(RerankerConfig):
    """Cohere Reranker model configuration class.
    
    Extends the base reranker configuration with Cohere API-specific parameters
    including authentication, request limits, and timeout settings.
    """
    api_key: str = Field(..., description="Cohere API key for authentication")
    timeout: int = Field(30, description="API request timeout in seconds")
    # Cohere API enforces maximum 1000 documents per request
    batch_size: int = Field(1000, description="Maximum number of documents per API request (Cohere limit)")

class CohereReranker(BaseReranker[CohereRerankerConfig]):
    """
    Optimized reranker implementation for Cohere API integration.
    
    Key optimizations:
    - Fully asynchronous Cohere client for non-blocking API calls
    - Automatic batching for large document sets exceeding API limits
    - Robust response parsing and error handling
    - Global result reordering for batched requests
    """

    def __init__(self, config: CohereRerankerConfig):
        """Initialize Cohere reranker with configuration.
        
        Args:
            config: Cohere-specific reranker configuration object
        """
        super().__init__(config)
        self._client: AsyncClient | None = None
        self._is_initialized = False

    async def startup(self) -> None:
        """Initialize fully asynchronous Cohere client with connectivity test.
        
        Creates the async Cohere client instance and performs a minimal API call
        to verify credentials and connectivity. Idempotent - safe to call multiple times.
        
        Raises:
            Exception: If client initialization or connectivity test fails
        """
        if self._is_initialized:
            return

        try:
            # Create async Cohere client with configured timeout
            self._client = AsyncClient(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )

            if not self._client:
                raise Exception("Cohere client initialization failed")
            
            # Perform minimal connectivity test to validate credentials
            await self._client.rerank(
                model=self.model_name,
                query="ping",
                documents=["pong"],
                top_n=1
            )
            
            self._is_initialized = True
            logger.info(f"✅ Cohere Reranker ({self.model_name}) initialized successfully")
        except Exception as e:
            logger.error(f"❌ Cohere Reranker initialization failed: {e}")
            raise

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Perform reranking with support for large document sets via automatic batching.
        
        Handles both small requests (under API limits) and large requests requiring
        batching with global result reordering. Uses fully asynchronous API calls
        to prevent event loop blocking.
        
        Args:
            query: Input query text for relevance scoring
            documents: List of document texts to rerank
            top_k: Number of top relevant documents to return (None returns all)
            
        Returns:
            list[dict[str, Any]]: Reranked results sorted by relevance score (descending),
                each containing "index" (original position) and "score" (relevance score)
                
        Raises:
            Exception: If API request fails or response parsing encounters errors
        """
        # Ensure client is initialized before processing
        if not self._is_initialized:
            await self.startup()

        # Return empty list for empty document input
        if not documents:
            return []

        # Handle large document sets exceeding Cohere API limits with batching
        if len(documents) > self.config.batch_size:
            return await self._rerank_batched(query, documents, top_k)

        # Set top_n to requested top_k or all documents for single batch requests
        top_n = top_k if top_k is not None else len(documents)
        
        try:
            # Make non-blocking async API call to Cohere rerank endpoint
            response = await self._client.rerank(  # type: ignore
                model=self.model_name,
                query=query,
                documents=documents,
                top_n=top_n,
            )

            # Parse and normalize response format
            results = []
            for result in response.results:
                results.append({
                    "index": result.index,
                    "score": float(result.relevance_score)
                })
            return results

        except Exception as e:
            logger.error(f"❌ Cohere Reranker API request failed: {e}")
            raise

    async def _rerank_batched(
        self, 
        query: str, 
        documents: list[str], 
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Process large reranking tasks exceeding Cohere API limits.
        
        Splits documents into batches compliant with Cohere's API limits, processes
        batches in parallel, combines results with corrected global indices, and
        performs global reordering by relevance score.
        
        Args:
            query: Input query text for relevance scoring
            documents: List of document texts to rerank (exceeds API limits)
            top_k: Number of top relevant documents to return (None returns all)
            
        Returns:
            list[dict[str, Any]]: Globally reranked results with corrected indices
        """
        # Create async tasks for parallel batch processing
        tasks = []
        for batch_start in range(0, len(documents), self.config.batch_size):
            # Extract current batch documents
            batch_docs = documents[batch_start : batch_start + self.config.batch_size]
            
            # Create async task for batch processing
            tasks.append(self._client.rerank(  # type: ignore
                model=self.model_name,
                query=query,
                documents=batch_docs,
                top_n=len(batch_docs)  # Return all results for batch to enable global ranking
            ))
        
        # Execute all batch requests in parallel
        responses = await asyncio.gather(*tasks)
        
        # Combine results and correct global indices (batch-local to global)
        combined_results = []
        for batch_idx, response in enumerate(responses):
            # Calculate index offset for current batch
            batch_offset = batch_idx * self.config.batch_size
            
            # Adjust indices and collect results
            for result in response.results:
                combined_results.append({
                    "index": result.index + batch_offset,  # Correct to global index
                    "score": float(result.relevance_score)
                })
        
        # Perform global reordering by relevance score (descending)
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply top-k filtering if specified
        return combined_results[:top_k] if top_k else combined_results

    async def shutdown(self) -> None:
        """Clean up Cohere client resources.
        
        Resets client instance and initialization state to ensure proper
        resource cleanup during application shutdown.
        """
        if self._client:
            self._client = None
        self._is_initialized = False
        logger.info("♻️ Cohere Reranker client closed successfully")

    @property
    def is_initialized(self) -> bool:
        """Check if reranker is properly initialized and ready for requests.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return self._is_initialized