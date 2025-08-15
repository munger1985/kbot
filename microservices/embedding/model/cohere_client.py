from typing import Any
from cohere import AsyncClient
from prometheus_client import Histogram, Counter, Gauge
from loguru import logger
import asyncio
from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse, EmbeddingDataItem

class CohereEmbeddingConfig(EmbeddingConfig):
    api_endpoint: str
    timeout: int = 30
    max_retries: int = 3
    api_key: str = ""

class CohereEmbedding(BaseEmbedding):
    """
    Production-grade Cohere embedding client with enhanced features:
    - Intelligent batching
    - Adaptive retry mechanism
    - Comprehensive monitoring
    - Resource optimization
    """

    # Prometheus metrics with Cohere-specific dimensions
    LATENCY_HIST = Histogram(
        'cohere_embedding_latency_seconds',
        'Latency for embedding requests',
        ['model_name', 'input_type', 'status']
    )
    
    ERROR_COUNTER = Counter(
        'cohere_embedding_errors_total',
        'Count of embedding errors',
        ['model_name', 'error_type']
    )
    
    REQUEST_COUNTER = Counter(
        'cohere_embedding_requests_total',
        'Count of embedding requests',
        ['model_name', 'input_type']
    )
    
    BATCH_SIZE_GAUGE = Gauge(
        'cohere_embedding_batch_size',
        'Effective batch size used',
        ['model_name']
    )
    
    TOKEN_USAGE = Gauge(
        'cohere_embedding_tokens_estimated',
        'Estimated tokens consumed',
        ['model_name']
    )

    def __init__(self, config: CohereEmbeddingConfig):
        """
        Initialize with Cohere-specific configuration.
        
        Args:
            config: CohereEmbeddingConfig containing:
                - api_key: Cohere API key
                - model_name: Model identifier (e.g. "embed-english-v3.0")
                - timeout: Request timeout in seconds
                - max_retries: (Removed as not supported by Cohere client)
                - default_input_type: Default input type ("search_document"/"search_query")
                - max_batch_size: Maximum texts per API call (default 96)
                - retry_delay: Base delay between retries in seconds
                - truncate_strategy: Default truncation ("END"/"START"/"NONE")
        """
        self._client: AsyncClient | None = None
        self.api_key = config.api_key
        self.model_name = config.model_name
        self.timeout = config.timeout or 30
        self.default_input_type = getattr(config, 'default_input_type', 'search_document')
        self.max_batch_size = getattr(config, 'max_batch_size', 96)  # Cohere recommended
        self.retry_delay = getattr(config, 'retry_delay', 1.0)
        self.truncate_strategy = getattr(config, 'truncate_strategy', 'END')
        self._is_initialized = False

    async def startup(self) -> None:
        """Initialize client with custom retry handling."""
        if self._is_initialized:
            return
            
        if not self.api_key:
            raise ValueError("Cohere API key must be provided")

        try:
            # Initialize without unsupported max_retries parameter
            self._client = AsyncClient(
                api_key=self.api_key,
                timeout=self.timeout  # Cohere client only accepts timeout
            )
            
            await self._validate_connection()
            self._is_initialized = True
            logger.info(f"Cohere client initialized for model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise RuntimeError("Cohere client initialization failed") from e

    async def _validate_connection(self) -> None:
        """Perform a lightweight connection test."""
        try:
            test_response = await self._client.embed(  # type: ignore
                texts=["connection test"],
                model=self.model_name,
                input_type=self.default_input_type,
                truncate=self.truncate_strategy
            )
            if not test_response.embeddings:
                raise ValueError("Empty test response")
        except Exception as e:
            raise RuntimeError(f"Connection test failed: {str(e)}") from e

    async def shutdown(self) -> None:
        """Graceful shutdown (Cohere client doesn't require explicit close)."""
        self._client = None
        self._is_initialized = False
        logger.info("Cohere client shutdown completed")

    async def embed(
        self,
        texts: list[str],
        batch_size: int = 0,
        input_type: str | None = None,
        truncate: str | None = None,
        raise_on_error: bool = True,
        **kwargs: Any
    ) -> EmbeddingResponse:
        """
        Enhanced Cohere embedding with intelligent batching and resilience.
        
        Args:
            texts: Input texts to embed
            batch_size: Override auto-detected batch size (0 for auto)
            input_type: Override default input type
            truncate: Override default truncation strategy
            raise_on_error: Whether to raise exceptions
            kwargs: Additional Cohere API parameters
            
        Returns:
            EmbeddingResponse: Standardized response format
            
        Raises:
            RuntimeError: For initialization failures
        """
        if not self._is_initialized:
            raise RuntimeError("Client not initialized. Call startup() first.")

        if not texts:
            logger.warning("Empty input text list")
            return self._empty_response()

        # Determine effective parameters
        effective_input_type = input_type or self.default_input_type
        effective_truncate = truncate or self.truncate_strategy
        effective_batch = self._calculate_batch_size(len(texts), batch_size)
        
        self.REQUEST_COUNTER.labels(
            model_name=self.model_name,
            input_type=effective_input_type
        ).inc()
        
        self.BATCH_SIZE_GAUGE.labels(model_name=self.model_name).set(effective_batch)

        try:
            with self.LATENCY_HIST.labels(
                model_name=self.model_name,
                input_type=effective_input_type,
                status="success"
            ).time():
                return await self._process_batches(
                    texts=texts,
                    batch_size=effective_batch,
                    input_type=effective_input_type,
                    truncate=effective_truncate,
                    **kwargs
                )
                
        except Exception as e:
            self._handle_error(e, effective_input_type)
            if raise_on_error:
                raise
            return self._empty_response()

    async def _process_batches(
        self,
        texts: list[str],
        batch_size: int,
        input_type: str,
        truncate: str,
        **kwargs: Any
    ) -> EmbeddingResponse:
        """Process batches with manual retry logic."""
        all_embeddings = []
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for attempt in range(3 + 1):  # Manual retry (max_retries = 3)
                try:
                    response = await self._client.embed(  # type: ignore
                        texts=batch,
                        model=self.model_name,
                        input_type=input_type,
                        truncate=truncate,
                        **kwargs
                    )
                    
                    all_embeddings.extend(response.embeddings)
                    total_tokens += sum(len(text.split()) for text in batch) * 2
                    break
                    
                except Exception as e:
                    if attempt == 3:  # Final attempt
                        raise
                        
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)

        return self._build_response(all_embeddings, total_tokens)

    def _calculate_batch_size(self, num_texts: int, user_batch_size: int) -> int:
        """Determine optimal batch size considering Cohere's limits."""
        if user_batch_size > 0:
            return min(user_batch_size, self.max_batch_size)
            
        # Auto-calculation based on text count
        if num_texts <= 32:  # Small batches for better latency
            return num_texts
        return min(
            max(32, num_texts // 4),  # Balance latency and throughput
            self.max_batch_size
        )

    def _build_response(self, embeddings: list[list[float]], total_tokens: int) -> EmbeddingResponse:
        """Construct standardized response object."""
        data = [
            EmbeddingDataItem(
                embedding=embedding,
                index=i,
                object="embedding"
            ) for i, embedding in enumerate(embeddings)
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

    def _handle_error(self, error: Exception, input_type: str) -> None:
        """Centralized error handling and logging."""
        error_type = type(error).__name__
        self.ERROR_COUNTER.labels(
            model_name=self.model_name,
            error_type=error_type
        ).inc()
        
        self.LATENCY_HIST.labels(
            model_name=self.model_name,
            input_type=input_type,
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
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384
        }
        return dim_map.get(self.model_name, 1024)  # Default to 1024

    async def health_check(self) -> dict[str, Any]:
        """Check service health status."""
        return {
            "initialized": self._is_initialized,
            "model": self.model_name,
            "last_error": None,
            "throughput": "N/A"
        }