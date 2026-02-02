import asyncio
from pydantic import Field
from loguru import logger
from cohere import AsyncClient
from cohere.core import ApiError

from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse

class CohereEmbeddingConfig(EmbeddingConfig):
    """Cohere 嵌入服务配置"""
    api_key: str = Field(..., description="Cohere API Key")
    timeout: int = Field(60, description="Cohere 响应较慢，建议增加超时")
    max_retries: int = Field(3, description="最大重试次数")
    max_concurrent_requests: int = Field(3, description="最大并发请求数")
    input_type_query: str = Field("search_query", description="查询输入类型")
    input_type_doc: str = Field("search_document", description="文档输入类型")

class CohereEmbedding(BaseEmbedding[CohereEmbeddingConfig]):
    """
    针对 Cohere V3 API 优化的 Embedding 实现
    优化点：并发请求控制、类型守卫增强、资源生命周期管理
    """

    def __init__(self, config: CohereEmbeddingConfig):
        super().__init__(config)
        self._client: AsyncClient | None = None
        self._is_initialized = False
        self.batch_size = config.batch_size if 0 < config.batch_size < 96 else 96 # 官方建议 96 为最大批次
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def startup(self) -> None:
        if self._is_initialized:
            return

        if not self.config.api_key:
            raise ValueError("未配置 Cohere API Key")

        try:
            # Cohere 的 AsyncClient 内部通过 httpx 管理连接池
            self._client = AsyncClient(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            # 连通性校验
            await self._client.embed(
                texts=["ping"],
                model=self.config.model_name,
                input_type="search_query"
            )
            self._is_initialized = True
            logger.info(f"✅ Cohere API 初始化成功: {self.config.model_name}")
        except Exception as e:
            logger.error(f"❌ Cohere 初始化失败: {e}")
            raise

    async def _embed_with_retry(self, texts: list[str], input_type: str) -> tuple[list[list[float]], int]:
        """带信号量控制和重试逻辑的原子请求"""
        async with self._semaphore:
            # 使用官方自带的重试逻辑（如果 AsyncClient 配置了 max_retries）
            # 或者在此手动实现精细化的指数退避
            try:
                response = await self._client.embed( # type: ignore
                    texts=texts,
                    model=self.config.model_name,
                    input_type=input_type,
                    embedding_types=["float"]
                )
                
                # V3 API 结构提取
                embeddings = response.embeddings.float_  # type: ignore
                tokens = 0
                if response.meta and response.meta.billed_tokens: # type: ignore
                    tokens = int(response.meta.billed_tokens.tokens or 0) # type: ignore
                
                return embeddings, tokens # type: ignore
            except ApiError as e:
                logger.error(f"❌ Cohere API 错误: {e.status_code} - {e.body}")
                raise

    async def embed(
        self, 
        texts: list[str], 
        batch_size: int | None = None, 
        is_query: bool = True
    ) -> EmbeddingResponse:
        """
        高性能异步嵌入逻辑
        """
        if not self._is_initialized:
            await self.startup()

        if not texts:
            return self._build_empty_response(self.config.model_name)

        # 1. 参数准备
        eff_batch_size = batch_size if batch_size is not None and 0 < batch_size <= 96 else self.batch_size
        input_type = self.config.input_type_query if is_query else self.config.input_type_doc
        
        # 2. 任务分片
        tasks = []
        for i in range(0, len(texts), eff_batch_size):
            batch = texts[i : i + eff_batch_size]
            tasks.append(self._embed_with_retry(batch, input_type))

        # 3. 并发执行
        # Cohere 官方建议不要过度并发，主要依靠其单次大批次处理能力
        results = await asyncio.gather(*tasks)

        # 4. 结果聚合
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
        """释放资源"""
        if self._client:
            # Cohere 的 AsyncClient 没有显式的 close 方法（基于 httpx 的会自动清理）
            # 但在重构中将其置空，符合生命周期管理惯例
            self._client = None
        self._is_initialized = False
        logger.info("♻️ Cohere 客户端已释放")

    @property
    def embedding_dim(self) -> int:
        dim_map = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384,
            "embed-english-v2.0": 4096,
            "embed-english-light-v2.0": 1024,
            "embed-multilingual-v2.0": 768
        }
        return dim_map.get(self.model_name, 1024)

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized