import asyncio
import math
from pydantic import Field
from loguru import logger
from openai import AsyncOpenAI, APIError, RateLimitError

from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse

class OpenAIEmbeddingConfig(EmbeddingConfig):
    """OpenAI Embedding service configuration"""
    api_key: str = Field(..., description="OpenAI API Key")
    api_base: str | None = Field(None, description="API proxy endpoint")
    dimensions: int | None = Field(None, description="输出向量维度")
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

    def _split_input(self, text: str) -> list[str]:
        """按配置预算切分远程输入，并尽量保留自然文本边界。"""
        byte_limit = int(self.config.max_tokens)
        if byte_limit <= 0:
            raise ValueError("Embedding 模型 max_tokens 必须为正整数")
        remaining = str(text)
        if not remaining.strip():
            raise ValueError("Embedding 输入不能为空")
        chunks: list[str] = []
        while len(remaining.encode("utf-8")) > byte_limit:
            candidate = remaining.encode("utf-8")[:byte_limit].decode(
                "utf-8", errors="ignore"
            )
            if not candidate:
                raise ValueError("Embedding 模型 max_tokens 小于单个字符的 UTF-8 长度")
            cut = len(candidate)
            minimum_boundary = max(1, len(candidate) // 2)
            for marker in ("\n\n", "\n", "。", "！", "？", ". ", "; ", " "):
                position = candidate.rfind(marker)
                boundary = position + len(marker)
                if position >= 0 and boundary >= minimum_boundary:
                    cut = boundary
                    break
            chunk = remaining[:cut].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[cut:]
        tail = remaining.strip()
        if tail:
            chunks.append(tail)
        return chunks

    @staticmethod
    def _merge_chunk_embeddings(
        vectors: list[list[float]], weights: list[int]
    ) -> list[float]:
        """按块内容长度加权合并并归一化，维持一条输入对应一个向量。"""
        if not vectors or len(vectors) != len(weights):
            raise RuntimeError("Embedding 分块结果数量不一致")
        dimension = len(vectors[0])
        if dimension <= 0 or any(len(vector) != dimension for vector in vectors):
            raise RuntimeError("Embedding 分块向量维度不一致")
        if len(vectors) == 1:
            return vectors[0]
        total_weight = sum(weights)
        if total_weight <= 0:
            raise RuntimeError("Embedding 分块权重无效")
        merged = [
            sum(vector[index] * weight for vector, weight in zip(vectors, weights))
            / total_weight
            for index in range(dimension)
        ]
        norm = math.sqrt(sum(value * value for value in merged))
        if norm <= 0:
            raise RuntimeError("Embedding 分块合并结果无法归一化")
        return [value / norm for value in merged]

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

        chunks_by_input = [self._split_input(text) for text in texts]
        remote_inputs = [
            chunk for chunks in chunks_by_input for chunk in chunks
        ]
        split_count = sum(len(chunks) > 1 for chunks in chunks_by_input)
        if split_count:
            logger.info(
                "Embedding 超长输入已完整分块：模型={} 输入条数={} 分块数={} max_tokens={}",
                self.model_name,
                split_count,
                len(remote_inputs),
                self.config.max_tokens,
            )

        # 1. Prepare request parameters
        eff_batch_size = batch_size if batch_size is not None and 0 < batch_size <= 96 else self.batch_size
        if self.model_name in {"text-embedding-v3", "text-embedding-v4"}:
            eff_batch_size = min(eff_batch_size, 10)
        embed_kwargs = {
            "model": self.model_name,
            "encoding_format": "float"
        }
        
        # 百炼 text-embedding-v3/v4 的默认输出为 1024，必须显式传递目录声明的维度。
        if self.config.dimensions and self.model_name in {
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-v3",
            "text-embedding-v4",
        }:
            embed_kwargs["dimensions"] = self.config.dimensions # type: ignore

        # 2. Create concurrent task queue
        tasks = []
        for i in range(0, len(remote_inputs), eff_batch_size):
            batch = remote_inputs[i : i + eff_batch_size]
            tasks.append(self._embed_batch(batch, **embed_kwargs))

        # 3. Execute concurrently and collect results
        # Use gather for concurrent requests to significantly speed up long list processing
        try:
            results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"💥 Embedding task group execution failed: {e}")
            raise

        # 4. Merge results
        chunk_embeddings = []
        total_tokens = 0
        for embeddings, tokens in results:
            chunk_embeddings.extend(embeddings)
            total_tokens += tokens

        if len(chunk_embeddings) != len(remote_inputs):
            raise RuntimeError("Embedding Provider 返回的向量数量与分块数量不一致")
        all_embeddings: list[list[float]] = []
        offset = 0
        for chunks in chunks_by_input:
            end = offset + len(chunks)
            all_embeddings.append(self._merge_chunk_embeddings(
                chunk_embeddings[offset:end],
                [len(chunk.encode("utf-8")) for chunk in chunks],
            ))
            offset = end

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
