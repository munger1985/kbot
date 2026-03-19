import httpx
from typing import Any
from pydantic import Field
from loguru import logger

from .base import BaseReranker, RerankerConfig

class OpenAIRerankerConfig(RerankerConfig):
    """OpenAI-compatible Reranker configuration class.
    
    Extends the base reranker configuration with parameters for OpenAI-style
    API integration, including authentication, endpoint configuration, and
    timeout settings for private deployments (vLLM, TEI, SGLang, etc.).
    """
    api_key: str = Field(..., description="API key for authentication (Bearer token)")
    api_endpoint: str = Field(..., description="API endpoint URL (e.g., http://localhost:8000/v1/rerank)")
    timeout: int = Field(30, description="Request timeout in seconds")
    # Some backends (e.g., vLLM) require specific model names for routing
    model_name: str = Field("bge-reranker-v2-m3", description="Model name for API routing")

class OpenAIReranker(BaseReranker[OpenAIRerankerConfig]):
    """
    Reranker client for OpenAI-compatible API endpoints.
    
    Designed for private reranker deployments including:
    - vLLM (High-performance LLM serving)
    - TEI (Text Embeddings Inference)
    - SGLang (Efficient LLM serving)
    
    Key features:
    - Asynchronous HTTP client with proper connection management
    - Standardized request/response handling across compatible backends
    - Robust error handling with status code logging
    - Bandwidth optimization (no document return by default)
    """

    def __init__(self, config: OpenAIRerankerConfig):
        """Initialize OpenAI-compatible reranker with configuration.
        
        Args:
            config: OpenAI-compatible reranker configuration object
        """
        super().__init__(config)
        self._client: httpx.AsyncClient | None = None
        self._is_initialized = False

    async def startup(self) -> None:
        """Initialize asynchronous HTTP client with connectivity test.
        
        Creates an httpx AsyncClient with proper authentication headers and
        performs a minimal API call to verify endpoint connectivity and
        authentication. Idempotent - safe to call multiple times.
        
        Raises:
            Exception: If client initialization or connectivity test fails
        """
        if self._is_initialized:
            return

        # Create async HTTP client with authentication headers
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            },
            timeout=self.config.timeout
        )
        
        # Perform minimal connectivity test to validate endpoint
        try:
            test_payload = {
                "model": self.model_name,
                "query": "hi",
                "documents": ["hi"],
                "top_n": 1
            }
            response = await self._client.post(self.config.api_endpoint, json=test_payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            self._is_initialized = True
            logger.info(f"✅ OpenAI-Compatible Reranker ({self.model_name}) initialized successfully")
        except Exception as e:
            logger.error(f"❌ Reranker connectivity test failed: {e}")
            raise

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Perform reranking via OpenAI-compatible API endpoint.
        
        Sends standardized rerank request to configured endpoint with
        bandwidth optimization (no document return) and flexible response
        parsing compatible with major private deployment backends.
        
        Args:
            query: Input query text for relevance scoring
            documents: List of document texts to rerank
            top_k: Number of top relevant documents to return (None returns all)
            
        Returns:
            list[dict[str, Any]]: Reranked results with "index" (original position)
                and "score" (relevance score, 0-1 range)
                
        Raises:
            RuntimeError: If client is not initialized
            httpx.HTTPStatusError: For HTTP error status codes
            Exception: For other request/parsing errors
        """
        # Ensure client is initialized before processing
        if not self._is_initialized:
            await self.startup()

        # Validate client initialization state
        if not self._client:
            raise RuntimeError("OpenAI Reranker client not initialized")
        
        # Return empty list for empty document input
        if not documents:
            return []

        # Prepare request payload with bandwidth optimization
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_k or len(documents),
            "return_documents": False  # Avoid returning full documents to save bandwidth
        }

        try:
            # Send async POST request to rerank endpoint
            response = await self._client.post(self.config.api_endpoint, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors (4xx/5xx)
            response_data = response.json()

            # Parse results with compatibility for different backend formats
            # Standard format: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
            # Fallback format: {"results": [{"index": 0, "score": 0.9}, ...]}
            results = []
            for item in response_data.get("results", []):
                # Use relevance_score with score as fallback for compatibility
                relevance_score = item.get("relevance_score") or item.get("score", 0.0)
                results.append({
                    "index": item.get("index"),
                    "score": float(relevance_score)
                })
            
            return results

        except httpx.HTTPStatusError as e:
            # Detailed logging for HTTP errors with status code and response text
            logger.error(f"❌ Rerank API status error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            # General error logging with full context
            logger.error(f"💥 Rerank request exception: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """Clean up HTTP client resources gracefully.
        
        Closes the async HTTP client connection and resets initialization state
        to prevent resource leaks during application shutdown.
        """
        if self._client:
            await self._client.aclose()  # Properly close async client
            self._client = None
        self._is_initialized = False
        logger.info("♻️ OpenAI Reranker client closed successfully")

    @property
    def is_initialized(self) -> bool:
        """Check if reranker is properly initialized and ready for requests.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return self._is_initialized