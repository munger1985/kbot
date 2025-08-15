from typing import Any
from openai import AsyncAzureOpenAI, APIConnectionError, RateLimitError, APIStatusError
from prometheus_client import Histogram, Counter, Gauge
from loguru import logger
import asyncio
from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse, EmbeddingDataItem

class AzureEmbeddingConfig(EmbeddingConfig):
    api_endpoint: str
    timeout: int = 30
    max_retries: int = 3
    api_key: str = ""
    deployment_name: str = ""
    api_version: str = "2023-05-15"
    
class AzureEmbedding(BaseEmbedding):
    """
    Production-grade Azure OpenAI embedding service with:
    - Intelligent batching
    - Adaptive retry policies
    - Comprehensive monitoring
    - Azure-specific optimizations
    """

    # Enhanced metrics with Azure-specific dimensions
    LATENCY_HIST = Histogram(
        'azure_embedding_latency_seconds',
        'Embedding request latency distribution',
        ['deployment', 'api_version', 'status']
    )
    
    ERROR_COUNTER = Counter(
        'azure_embedding_errors_total',
        'Embedding error counts by type',
        ['deployment', 'error_code']
    )
    
    REQUEST_COUNTER = Counter(
        'azure_embedding_requests_total',
        'Total embedding requests processed',
        ['deployment', 'api_version']
    )
    
    BATCH_SIZE_GAUGE = Gauge(
        'azure_embedding_batch_size',
        'Effective batch size per request',
        ['deployment']
    )
    
    TOKEN_USAGE = Gauge(
        'azure_embedding_tokens_used',
        'Tokens consumed per request',
        ['deployment']
    )

    def __init__(self, config: AzureEmbeddingConfig):
        """
        Initialize with Azure-specific configuration.
        
        Args:
            config: RemoteEmbeddingConfig containing:
                - api_key: Azure API key
                - deployment_name: Deployment name
                - endpoint: Azure endpoint URL
                - api_version: API version (default "2023-05-15")
                - timeout: Request timeout (default 30s)
                - max_retries: Maximum retries (default 3)
                - max_batch_size: Maximum texts per request (default 16)
                - min_batch_size: Minimum texts per request (default 1)
                - retry_delay: Base retry delay (default 1.0s)
                - headers: Custom HTTP headers
                - azure_params: Additional Azure parameters
        """
        self._client: AsyncAzureOpenAI | None = None
        self.api_key = config.api_key
        self.deployment_name = config.deployment_name
        self.endpoint = config.api_endpoint
        self.api_version = config.api_version or "2023-05-15"
        self.timeout = config.timeout #or settings['embed']['timeout']
        self.max_retries = getattr(config, 'max_retries', 3)
        self.max_batch_size = getattr(config, 'max_batch_size', 16)  # Azure recommendation
        self.min_batch_size = getattr(config, 'min_batch_size', 1)
        self.retry_delay = getattr(config, 'retry_delay', 1.0)
        self.custom_headers = getattr(config, 'headers', {})
        self._azure_params = getattr(config, 'azure_params', {})
        self._is_initialized = False

    async def startup(self) -> None:
        """Initialize client with connection validation."""
        if self._is_initialized:
            return
            
        if not all([self.api_key, self.endpoint, self.deployment_name]):
            raise ValueError("Missing required Azure configuration")

        try:
            headers = {
                "User-Agent": "AzureEmbedding/1.0",
                "X-Deployment-Name": self.deployment_name,
                **self.custom_headers
            }

            self._client = AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                timeout=self.timeout,
                max_retries=self.max_retries,
                default_headers=headers,
                **self._azure_params
            )
            
            await self._validate_connection()
            self._is_initialized = True
            logger.success(f"Azure client ready for {self.deployment_name}")
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise RuntimeError("Azure client initialization failed") from e

    async def _validate_connection(self) -> None:
        """Perform lightweight connection test."""
        try:
            test_response = await self._client.embeddings.create(  # type: ignore
                model=self.deployment_name,
                input=["connection test"],
                encoding_format="float"
            )
            if not test_response.data:
                raise ValueError("Empty test response")
        except Exception as e:
            await self._client.close()  # type: ignore
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
            logger.info("Azure client shutdown completed")
        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")
            raise

    async def embed(
        self,
        texts: list[str],
        batch_size: int = 0,
        raise_on_error: bool = True,
        **kwargs: Any
    ) -> EmbeddingResponse:
        """
        Generate embeddings with Azure-specific optimizations.
        
        Args:
            texts: Input texts to embed
            batch_size: Override auto batch size (0 for auto)
            raise_on_error: Whether to raise exceptions
            kwargs: Additional Azure API parameters
            
        Returns:
            EmbeddingResponse: Standardized response
            
        Raises:
            RuntimeError: If client not initialized
        """
        if not self._is_initialized:
            raise RuntimeError("Client not initialized. Call startup() first.")

        if not texts:
            logger.warning("Received empty input texts")
            return self._empty_response()

        # Calculate effective batch size
        effective_batch = self._calculate_batch_size(len(texts), batch_size)
        self.REQUEST_COUNTER.labels(
            deployment=self.deployment_name,
            api_version=self.api_version
        ).inc()
        
        self.BATCH_SIZE_GAUGE.labels(deployment=self.deployment_name).set(effective_batch)

        try:
            with self.LATENCY_HIST.labels(
                deployment=self.deployment_name,
                api_version=self.api_version,
                status="success"
            ).time():
                return await self._process_batches(texts, effective_batch, **kwargs)
                
        except Exception as e:
            self._handle_error(e)
            if raise_on_error:
                raise
            return self._empty_response()

    async def _process_batches(
        self,
        texts: list[str],
        batch_size: int,
        **kwargs: Any
    ) -> EmbeddingResponse:
        """Process batches with Azure-specific retry logic."""
        all_embeddings = []
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self._client.embeddings.create(  # type: ignore
                        model=self.deployment_name,
                        input=batch,
                        encoding_format="float",
                        **kwargs
                    )
                    
                    all_embeddings.extend([item.embedding for item in response.data])
                    total_tokens += response.usage.total_tokens
                    break
                    
                except RateLimitError:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                except APIConnectionError:
                    if attempt == self.max_retries:
                        raise
                    await asyncio.sleep(self.retry_delay)
                except APIStatusError as e:
                    if e.status_code >= 500:  # Retry server errors
                        if attempt == self.max_retries:
                            raise
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        raise

        self.TOKEN_USAGE.labels(deployment=self.deployment_name).set(total_tokens)
        return self._build_response(all_embeddings, total_tokens)

    def _calculate_batch_size(self, num_texts: int, user_batch_size: int) -> int:
        """Calculate optimal batch size considering Azure limits."""
        if user_batch_size > 0:
            return min(user_batch_size, self.max_batch_size)
            
        # Auto-calculate based on text length
        avg_length = sum(len(t) for t in texts) / max(1, len(texts)) # type: ignore
        if avg_length > 1000:  # Reduce batch size for long documents
            return min(8, self.max_batch_size)
        return min(
            max(self.min_batch_size, num_texts // 4),
            self.max_batch_size
        )

    def _build_response(self, embeddings: list[list[float]], total_tokens: int) -> EmbeddingResponse:
        """Construct standardized response."""
        data = [
            EmbeddingDataItem(
                embedding=embedding,
                index=i,
                object="embedding"
            ) for i, embedding in enumerate(embeddings)
        ]
        
        return EmbeddingResponse(
            data=data,
            model=self.deployment_name,
            object="list",
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        )

    def _handle_error(self, error: Exception) -> None:
        """Centralized error handling."""
        error_code = "unknown"
        if isinstance(error, RateLimitError):
            error_code = "rate_limit"
        elif isinstance(error, APIConnectionError):
            error_code = "connection"
        elif isinstance(error, APIStatusError):
            error_code = f"http_{error.status_code}"
            
        self.ERROR_COUNTER.labels(
            deployment=self.deployment_name,
            error_code=error_code
        ).inc()
        
        self.LATENCY_HIST.labels(
            deployment=self.deployment_name,
            api_version=self.api_version,
            status="error"
        ).observe(0)
        
        logger.error(f"Embedding failed - {self.deployment_name}: {str(error)}")

    def _empty_response(self) -> EmbeddingResponse:
        """Generate empty response for error cases."""
        return EmbeddingResponse(
            data=[],
            model=self.deployment_name,
            object="list",
            usage={"prompt_tokens": 0, "total_tokens": 0}
        )

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension for the deployment."""
        dim_map = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        return dim_map.get(self.deployment_name.split('-')[0], 1536)  # Default fallback

    async def health_check(self) -> dict[str, Any]:
        """Get service health status."""
        return {
            "initialized": self._is_initialized,
            "deployment": self.deployment_name,
            "api_version": self.api_version,
            "last_error": None,
            "throughput": "N/A"
        }