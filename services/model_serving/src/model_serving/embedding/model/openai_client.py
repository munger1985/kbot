import asyncio
from pydantic import Field
from loguru import logger
from openai import AsyncOpenAI, APIError, RateLimitError

from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse

class OpenAIEmbeddingConfig(EmbeddingConfig):
    """OpenAI Embedding service configuration"""
    api_key: str = Field(..., description="OpenAI API Key")
    api_base: str | None = Field(None, description="API proxy endpoint")
    dimensions: int | None = Field(None, description="Output dimensions (only supported by v3 series models)")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(2, description="SDK internal retry count")
    max_concurrent_requests: int = Field(5, description="Maximum concurrent requests")

class OpenAIEmbedding(BaseEmbedding[OpenAIEmbeddingConfig]):
    """
    Refactored OpenAI Embedding implementation.
    Optimizations: Concurrent batch processing, granular exception handling, decoupled initialization logic.
    """

    def __init__(self, config: OpenAIEmbeddingConfig):
        super().__init__(config)
        self._client: AsyncOpenAI | None = None
        self._is_initialized = False
        # Use semaphore to control concurrency and prevent API rate limit violations
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def startup(self) -> None:
        """Initialize: Create asynchronous client connection"""
        if self._is_initialized:
            return

        if not self.config.api_key:
            raise ValueError("OpenAI API Key not configured")

        try:
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
            # Simplified connectivity test (log-level confirmation only)
            logger.info(f"🚀 OpenAI client ready: {self.model_name}")
            self._is_initialized = True
        except Exception as e:
            logger.error(f"❌ OpenAI client initialization failed: {e}")
            raise

    async def _embed_batch(self, batch: list[str], **kwargs) -> tuple[list[list[float]], int]:
        """Atomic batch processing with concurrency control and error handling"""
        if self._client is None:
            raise ValueError("OpenAI client not initialized")
        
        async with self._semaphore:
            try:
                response = await self._client.embeddings.create(
                    input=batch,
                    **kwargs
                )
                embeddings = [item.embedding for item in response.data]
                tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
                return embeddings, tokens
            except RateLimitError:
                logger.warning("⚠️ OpenAI rate limit triggered - consider reducing concurrency or batch size")
                raise
            except APIError as e:
                logger.error(f"❌ OpenAI API exception: {e}")
                raise

    async def embed(
        self, 
        texts: list[str], 
        batch_size: int | None = None, 
        is_query: bool = True
    ) -> EmbeddingResponse:
        """
        High-performance embedding implementation:
        1. Auto-initialization
        2. Batch splitting with concurrent coroutine execution
        3. Result aggregation
        """
        if not self._is_initialized:
            await self.startup()

        if not texts:
            return self._build_empty_response(self.model_name)

        # 1. Prepare request parameters
        eff_batch_size = batch_size if batch_size is not None and 0 < batch_size <= 96 else self.batch_size
        embed_kwargs = {
            "model": self.model_name,
            "encoding_format": "float"
        }
        
        # Handle Matryoshka dimension truncation for v3 models
        if self.config.dimensions and "text-embedding-3" in self.model_name:
            embed_kwargs["dimensions"] = self.config.dimensions # type: ignore

        # 2. Create concurrent task queue
        tasks = []
        for i in range(0, len(texts), eff_batch_size):
            batch = texts[i : i + eff_batch_size]
            tasks.append(self._embed_batch(batch, **embed_kwargs))

        # 3. Execute concurrently and collect results
        # Use gather for concurrent requests to significantly speed up long list processing
        try:
            results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"💥 Embedding task group execution failed: {e}")
            raise

        # 4. Merge results
        all_embeddings = []
        total_tokens = 0
        for embeddings, tokens in results:
            all_embeddings.extend(embeddings)
            total_tokens += tokens

        return self._build_standard_response(
            embeddings=all_embeddings,
            model_name=self.model_name,
            tokens=total_tokens
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
        self._is_initialized = False
        logger.info("♻️ OpenAI client closed")

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension (with model-specific defaults)"""
        if self.config.dimensions:
            return self.config.dimensions
            
        dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dim_map.get(self.model_name, 1536)

    @property
    def is_initialized(self) -> bool:
        """Check if client is initialized"""
        return self._is_initialized