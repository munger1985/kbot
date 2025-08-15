import os
import asyncio
import numpy as np
from typing import Any, cast
from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError
from prometheus_client import Histogram, Counter, Gauge
from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse, EmbeddingDataItem
from loguru import logger

class OpenAIEmbeddingConfig(EmbeddingConfig):
    api_endpoint: str
    timeout: int = 30
    max_retries: int = 3
    api_key: str = ""
    organization: str | None = None # Optional


class OpenAIEmbedding(BaseEmbedding):
    """
    Production-grade OpenAI embedding client with enhanced monitoring and resilience.
    
    Features:
    - Automatic request batching
    - Comprehensive error handling
    - Detailed performance metrics
    - Adaptive retry mechanism
    - Resource cleanup safeguards
    """

    # Prometheus metrics
    LATENCY_HIST = Histogram(
        'openai_embedding_latency_seconds',
        'Latency for embedding requests',
        ['model_name', 'status']
    )
    
    ERROR_COUNTER = Counter(
        'openai_embedding_errors_total',
        'Count of embedding errors',
        ['model_name', 'error_type']
    )
    
    REQUEST_COUNTER = Counter(
        'openai_embedding_requests_total',
        'Count of embedding requests',
        ['model_name']
    )
    
    BATCH_SIZE_GAUGE = Gauge(
        'openai_embedding_batch_size',
        'Effective batch size used',
        ['model_name']
    )
    
    TOKEN_USAGE = Gauge(
        'openai_embedding_tokens_used',
        'Tokens consumed per request',
        ['model_name']
    )

    def __init__(self, config: OpenAIEmbeddingConfig):
        """
        Initialize with enhanced configuration options.
        
        Args:
            config: OpenAIEmbeddingConfig containing:
                - model_name: OpenAI model identifier
                - api_key: API key (recommend using environment variables)
                - timeout: Request timeout in seconds
                - max_retries: Maximum retry attempts
                - organization: Optional organization ID
                - max_batch_size: Maximum texts per API call
                - min_batch_size: Minimum texts per API call
                - retry_delay: Base delay between retries in seconds
        """

        # 初始化参数
        self._client: AsyncOpenAI | None = None
        self.model_name = config.model_name
        self.api_key = config.api_key
        self.timeout = config.timeout or 30
        self.max_retries = config.max_retries or 0
        self.organization = config.organization
        self.max_batch_size = getattr(config, 'max_batch_size', 100)
        self.min_batch_size = getattr(config, 'min_batch_size', 10)
        self.retry_delay = getattr(config, 'retry_delay', 1.0)
        self._is_initialized = False

    async def startup(self) -> None:
        """Initialize client with connection pooling and validation."""
        if self._is_initialized:
            logger.warning("Client already initialized")
            return
            
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided")
            
        try:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
                organization=self.organization
            )
            
            # Validate connection
            await self._validate_connection()
            self._is_initialized = True
            logger.info(f"OpenAI client initialized for model: {self.model_name}")
            
        except Exception as e:
            self._is_initialized = False
            logger.error(f"Initialization failed: {str(e)}")
            raise RuntimeError("OpenAI client initialization failed") from e

    async def _validate_connection(self) -> None:
        """Perform a lightweight validation request."""
        try:
            test_response = await self._client.embeddings.create( # type: ignore
                model=self.model_name,
                input=["connection test"],
                encoding_format="float"
            )
            if not test_response.data:
                raise ValueError("Empty test response")
        except Exception as e:
            await self._client.close() # type: ignore
            raise RuntimeError(f"Connection test failed: {str(e)}") from e

    async def shutdown(self) -> None:
        """Graceful shutdown with resource cleanup."""
        if not self._is_initialized:
            return
            
        try:
            if self._client:
                await self._client.close()
            self._client = None
            self._is_initialized = False
            logger.info("OpenAI client shutdown completed")
        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")
            raise

    async def embed(
        self,
        texts: list[str],
        batch_size: int = 0,
        normalize: bool = False,
        raise_on_error: bool = True
    ) -> EmbeddingResponse:
        """
        Enhanced embedding generation with intelligent batching and resilience.
        
        Args:
            texts: Input texts to embed
            batch_size: Override auto-detected batch size (0 for auto)
            normalize: L2-normalize output embeddings
            raise_on_error: Whether to raise exceptions or return empty response
            
        Returns:
            EmbeddingResponse: Standardized response format
            
        Raises:
            RuntimeError: For initialization or critical failures
        """
        if not self._is_initialized:
            raise RuntimeError("Client not initialized. Call startup() first.")

        if not texts:
            logger.warning("Empty input text list")
            return self._empty_response()

        # Calculate effective batch size
        effective_batch = self._calculate_batch_size(len(texts), batch_size)
        self.BATCH_SIZE_GAUGE.labels(model_name=self.model_name).set(effective_batch)
        self.REQUEST_COUNTER.labels(model_name=self.model_name).inc()

        try:
            with self.LATENCY_HIST.labels(model_name=self.model_name, status="success").time():
                return await self._process_batches(texts, effective_batch, normalize)
                
        except Exception as e:
            self._handle_error(e)
            if raise_on_error:
                raise
            return self._empty_response()

    async def _process_batches(
        self,
        texts: list[str],
        batch_size: int,
        normalize: bool
    ) -> EmbeddingResponse:
        """Process texts in batches with retry logic."""
        all_embeddings = []
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{len(texts)//batch_size + 1}")
            
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self._client.embeddings.create( # type: ignore
                        model=self.model_name,
                        input=batch,
                        encoding_format="float"
                    )
                    
                    batch_embeddings = [item.embedding for item in response.data]
                    if normalize:
                        batch_embeddings = [self._normalize_embedding(e) for e in batch_embeddings]
                        
                    all_embeddings.extend(batch_embeddings)
                    total_tokens += response.usage.total_tokens
                    break
                    
                except RateLimitError:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                except APIConnectionError as e:
                    if attempt == self.max_retries:
                        raise
                    await asyncio.sleep(self.retry_delay)
                except APIError as e:
                    self.ERROR_COUNTER.labels(
                        model_name=self.model_name,
                        error_type="api_error"
                    ).inc()
                    raise

        self.TOKEN_USAGE.labels(model_name=self.model_name).set(total_tokens)
        return self._build_response(all_embeddings, total_tokens)

    def _calculate_batch_size(self, num_texts: int, user_batch_size: int) -> int:
        """Determine optimal batch size considering API limits."""
        if user_batch_size > 0:
            return min(user_batch_size, self.max_batch_size)
            
        # Auto-calculation based on text count
        if num_texts <= self.min_batch_size:
            return num_texts
        return min(
            max(self.min_batch_size, num_texts // 4),
            self.max_batch_size
        )

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        """
        Safely normalize embedding vector with proper type handling.
        
        Args:
            embedding: Input vector to normalize
            
        Returns:
            list[float]: Normalized vector guaranteed to be list[float]
        """
        try:
            norm = np.linalg.norm(embedding)
            if norm <= 0:
                return embedding
                
            # 方案1：显式转换
            # return [float(x / norm) for x in embedding]
            
            # 或者方案2：使用NumPy
            arr = np.array(embedding, dtype=np.float32)
            normalized = (arr / norm).tolist()
            return cast(list[float], normalized)
            
        except Exception as e:
            logger.warning(f"Normalization failed: {str(e)}")
            return embedding

    def _build_response(self, embeddings: list[list[float]], total_tokens: int) -> EmbeddingResponse:
        """Construct standardized response object."""
        data = [
            EmbeddingDataItem(
                embedding=emb,
                index=i,
                object="embedding"
            ) for i, emb in enumerate(embeddings)
        ]
        
        return EmbeddingResponse(
            data=data,
            model=self.model_name,
            object="list",
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        )

    def _handle_error(self, error: Exception) -> None:
        """Centralized error handling and logging."""
        error_type = type(error).__name__
        self.ERROR_COUNTER.labels(
            model_name=self.model_name,
            error_type=error_type
        ).inc()
        
        self.LATENCY_HIST.labels(
            model_name=self.model_name,
            status="error"
        ).observe(0)  # Record failed request
        
        logger.error(f"Embedding failed - Model: {self.model_name}, Error: {str(error)}")

    def _empty_response(self) -> EmbeddingResponse:
        """Generate empty response for error cases."""
        return EmbeddingResponse(
            data=[],
            model=self.model_name,
            object="list",
            usage={"prompt_tokens": 0, "total_tokens": 0}
        )

    @property
    def embedding_dim(self) -> int:
        """Get output dimension for the configured model."""
        dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dim_map.get(self.model_name, 1536)  # Default to 1536 if unknown

    async def health_check(self) -> dict[str, Any]:
        """Check service health status."""
        return {
            "initialized": self._is_initialized,
            "model": self.model_name,
            "last_error": None,
            "throughput": "N/A"  # Could track actual metrics
        }