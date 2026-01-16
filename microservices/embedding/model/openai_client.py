import asyncio
from pydantic import Field
from loguru import logger
from openai import AsyncOpenAI, APIError, RateLimitError

from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse

class OpenAIEmbeddingConfig(EmbeddingConfig):
    """OpenAI 嵌入服务配置"""
    api_key: str = Field(..., description="OpenAI API Key")
    api_base: str | None = Field(None, description="API 代理地址")
    dimensions: int | None = Field(None, description="输出维度(仅支持v3系列模型)")
    timeout: int = Field(30, description="超时时间")
    max_retries: int = Field(2, description="SDK 内部重试次数")
    max_concurrent_requests: int = Field(5, description="最大并发请求数")

class OpenAIEmbedding(BaseEmbedding[OpenAIEmbeddingConfig]):
    """
    重构后的 OpenAI Embedding 实现
    优化点：并发批处理、异常细分处理、初始化逻辑解耦
    """

    def __init__(self, config: OpenAIEmbeddingConfig):
        super().__init__(config)
        self._client: AsyncOpenAI | None = None
        self._is_initialized = False
        # 使用信号量控制并发，防止触发 API 速率限制 (Rate Limit)
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def startup(self) -> None:
        """初始化：建立异步客户端"""
        if self._is_initialized:
            return

        if not self.config.api_key:
            raise ValueError("未配置 OpenAI API Key")

        try:
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
            # 简化连通性测试，仅在 log 层面确认
            logger.info(f"🚀 OpenAI 客户端就绪: {self.model_name}")
            self._is_initialized = True
        except Exception as e:
            logger.error(f"❌ OpenAI 客户端初始化失败: {e}")
            raise

    async def _embed_batch(self, batch: list[str], **kwargs) -> tuple[list[list[float]], int]:
        """原子级批处理逻辑，增加并发控制和错误截获"""
        if self._client is None:
            raise ValueError("OpenAI 客户端未初始化")
        
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
                logger.warning("⚠️ OpenAI 触发速率限制，请考虑调低并发数或批大小")
                raise
            except APIError as e:
                logger.error(f"❌ OpenAI API 异常: {e}")
                raise

    async def embed(
        self, 
        texts: list[str], 
        batch_size: int | None = None, 
        is_query: bool = True
    ) -> EmbeddingResponse:
        """
        高性能向量化实现：
        1. 自动初始化
        2. 切分 Batch 并通过协程并发执行
        3. 聚合结果
        """
        if not self._is_initialized:
            await self.startup()

        if not texts:
            return self._build_empty_response(self.model_name)

        # 1. 准备请求参数
        eff_batch_size = batch_size if batch_size is not None and 0 < batch_size <= 96 else self.batch_size
        embed_kwargs = {
            "model": self.model_name,
            "encoding_format": "float"
        }
        
        # 处理 v3 模型的 Matryoshka 维度裁剪
        if self.config.dimensions and "text-embedding-3" in self.model_name:
            embed_kwargs["dimensions"] = self.config.dimensions # type: ignore

        # 2. 构造并发任务队列
        tasks = []
        for i in range(0, len(texts), eff_batch_size):
            batch = texts[i : i + eff_batch_size]
            tasks.append(self._embed_batch(batch, **embed_kwargs))

        # 3. 并发执行并收集结果
        # 使用 gather 并发请求，大幅提升处理长列表的速度
        try:
            results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"💥 Embedding 任务组执行失败: {e}")
            raise

        # 4. 合并数据
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
        logger.info("♻️ OpenAI 客户端已关闭")

    @property
    def embedding_dim(self) -> int:
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
        return self._is_initialized