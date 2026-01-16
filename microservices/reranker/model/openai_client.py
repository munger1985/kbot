import httpx
from typing import Any
from pydantic import Field
from loguru import logger

from .base import BaseReranker, RerankerConfig

class OpenAIRerankerConfig(RerankerConfig):
    """OpenAI 兼容接口 Reranker 配置"""
    api_key: str = Field(..., description="API 密钥")
    api_endpoint: str = Field(..., description="API 端点 (例如 http://localhost:8000/v1/rerank)")
    timeout: int = Field(30, description="请求超时时间")
    # 某些后端（如 vLLM）可能需要特定的模型名称
    model_name: str = Field("bge-reranker-v2-m3", description="模型名称")

class OpenAIReranker(BaseReranker[OpenAIRerankerConfig]):
    """
    OpenAI 兼容协议的 Reranker 客户端
    适用于 vLLM, TEI, SGLang 等私有化部署后端
    """

    def __init__(self, config: OpenAIRerankerConfig):
        super().__init__(config)
        self._client: httpx.AsyncClient | None = None
        self._is_initialized = False

    async def startup(self) -> None:
        """初始化 HTTP 异步客户端"""
        if self._is_initialized:
            return

        # 使用 httpx 构建标准异步请求客户端
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            },
            timeout=self.config.timeout
        )
        
        # 简单联通性测试
        try:
            test_data = {
                "model": self.model_name,
                "query": "hi",
                "documents": ["hi"],
                "top_n": 1
            }
            response = await self._client.post(self.config.api_endpoint, json=test_data)
            response.raise_for_status()
            self._is_initialized = True
            logger.info(f"✅ OpenAI-Compatible Reranker ({self.model_name}) 初始化成功")
        except Exception as e:
            logger.error(f"❌ Reranker 连通性测试失败: {e}")
            raise

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        执行重排序请求
        """
        if not self._is_initialized:
            await self.startup()

        if not self._client:
            raise RuntimeError("OpenAI Reranker 客户端未初始化")
        
        if not documents:
            return []

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_k or len(documents),
            "return_documents": False # 通常不需要返回原文，节省带宽
        }

        try:
            response = await self._client.post(self.config.api_endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

            # 解析结果：适配主流后端返回的格式
            # 主流格式：{"results": [{"index": 0, "relevance_score": 0.9}, ...]}
            results = []
            for item in data.get("results", []):
                results.append({
                    "index": item.get("index"),
                    "score": float(item.get("relevance_score") or item.get("score", 0.0))
                })
            
            return results

        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Rerank API 状态错误: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"💥 Rerank 请求异常: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._is_initialized = False
        logger.info("♻️ OpenAI Reranker 客户端已正常关闭")

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized