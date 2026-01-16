import asyncio
from typing import Any
from pydantic import Field
from loguru import logger
from cohere import AsyncClient

from .base import RerankerConfig, BaseReranker

class CohereRerankerConfig(RerankerConfig):
    """Cohere Reranker 模型配置类"""
    api_key: str = Field(..., description="Cohere API 密钥")
    timeout: int = Field(30, description="API 请求超时时间")
    # Cohere API 限制单次请求最多 1000 个文档
    batch_size: int = Field(1000, description="单次请求最大文档数限制")

class CohereReranker(BaseReranker[CohereRerankerConfig]):
    """
    针对 Cohere API 优化的重排器实现
    优化点：全异步 Client、自动分批、健壮的响应解析
    """

    def __init__(self, config: CohereRerankerConfig):
        super().__init__(config)
        self._client: AsyncClient | None = None
        self._is_initialized = False

    async def startup(self) -> None:
        """初始化全异步 Cohere 客户端"""
        if self._is_initialized:
            return

        try:
            self._client = AsyncClient(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            # 连通性简单测试
            await self._client.rerank(
                model=self.model_name,
                query="ping",
                documents=["pong"],
                top_n=1
            )
            self._is_initialized = True
            logger.info(f"✅ Cohere Reranker ({self.model_name}) 初始化成功")
        except Exception as e:
            logger.error(f"❌ Cohere Reranker 初始化失败: {e}")
            raise

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        执行重排序：支持超大规模文档分批处理
        """
        if not self._is_initialized:
            await self.startup()

        if not documents:
            return []

        # Cohere 限制单次请求 docs 数量 (通常为 1000)
        # 如果超过限制，需要进行分批并重新合并排序
        if len(documents) > self.config.batch_size:
            return await self._rerank_batched(query, documents, top_k)

        top_n = top_k if top_k is not None else len(documents)
        
        try:
            # 使用异步非阻塞调用
            response = await self._client.rerank( # type: ignore
                model=self.model_name,
                query=query,
                documents=documents,
                top_n=top_n,
            )

            results = []
            for result in response.results:
                results.append({
                    "index": result.index,
                    "score": float(result.relevance_score)
                })
            return results

        except Exception as e:
            logger.error(f"❌ Cohere Reranker 请求失败: {e}")
            raise

    async def _rerank_batched(
        self, 
        query: str, 
        documents: list[str], 
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """处理超过 API 限制的大规模重排任务"""
        tasks = []
        for i in range(0, len(documents), self.config.batch_size):
            batch = documents[i : i + self.config.batch_size]
            tasks.append(self._client.rerank(  # type: ignore
                model=self.model_name,
                query=query,
                documents=batch,
                top_n=len(batch)
            ))
        
        responses = await asyncio.gather(*tasks)
        
        # 结果合并并修正全局索引
        combined_results = []
        for batch_idx, resp in enumerate(responses):
            offset = batch_idx * self.config.batch_size
            for res in resp.results:
                combined_results.append({
                    "index": res.index + offset,
                    "score": float(res.relevance_score)
                })
        
        # 按分数重新全局排序
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results[:top_k] if top_k else combined_results

    async def shutdown(self) -> None:
        """释放资源"""
        if self._client:
            self._client = None
        self._is_initialized = False
        logger.info("♻️ Cohere Reranker 客户端已关闭")

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized