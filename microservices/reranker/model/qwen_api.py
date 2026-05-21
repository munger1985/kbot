import httpx
from typing import Any
from pydantic import Field
from loguru import logger

# 假设你的基类在同级目录的 base.py 中
from .base import BaseReranker, RerankerConfig


class QwenRerankerConfig(RerankerConfig):
    """阿里云百炼 Qwen Reranker 模型配置类。
    
    专门针对 DashScope 平台的 API 规范进行扩展，配置默认端点、
    模型名称以及鉴权所需的 API Key。
    """
    api_key: str = Field(..., description="阿里云 DashScope API Key")
    # 默认使用百炼平台的通用 Rerank 统一入口
    api_endpoint: str = Field(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/rerank", 
        description="DashScope Rerank API 端点"
    )
    timeout: int = Field(30, description="请求超时时间（秒）")
    # Qwen 官方推荐的最新主流重排模型（例如：gte-rerank）
    model_name: str = Field("gte-rerank", description="Qwen/DashScope 适用的重排模型名称")


class QwenReranker(BaseReranker[QwenRerankerConfig]):
    """阿里云百炼 Qwen/GTE Reranker 适配客户端。
    
    对接 DashScope 文本重排服务，支持异步调用、异常处理及标准的接口出参对齐。
    """

    def __init__(self, config: QwenRerankerConfig):
        super().__init__(config)
        self._client: httpx.AsyncClient | None = None
        self._is_initialized = False

    async def startup(self) -> None:
        """初始化异步 HTTP 客户端并执行联通性测试。
        
        使用 DashScope 特定的请求头（X-DashScope-ApiKey）构建客户端。
        """
        if self._is_initialized:
            return

        # DashScope 鉴权规定使用 X-DashScope-ApiKey 头部
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            },
            timeout=self.config.timeout
        )

        # 极简冒烟测试，验证 API Key 与网络连通性
        try:
            test_payload = {
                "model": self.model_name,
                "query": "hi",
                "documents": ["hi"],
                "top_n": 1
            }
            response = await self._client.post(self.config.api_endpoint, json=test_payload)
            response.raise_for_status()
            self._is_initialized = True
            logger.info(f"✅ Qwen Reranker ({self.model_name}) 初始化并连接成功")
        except Exception as e:
            logger.error(f"❌ Qwen Reranker 连通性测试失败: {e}")
            raise

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """执行重排，并将 DashScope 的返回结构对齐为基类要求的标准格式。
        
        基类规范返回值：
            list[dict]: 包含 "index" 和 "score"（由大到小排序）
        """
        if not self._is_initialized:
            await self.startup()

        if not self._client:
            raise RuntimeError("Qwen Reranker client 未初始化")
        
        if not documents:
            return []

        # 构造百炼标准的入参结构 (把 query 和 docs 压入 input 字段中)
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_k or len(documents),
            "return_documents": False
        }

        try:
            response = await self._client.post(self.config.api_endpoint, json=payload)
            response.raise_for_status()
            response_data = response.json()

            # --- 核心适配：解析 DashScope 的返回结构 ---
            raw_results = response_data.get("results", [])
            results = []
            for item in raw_results:
                score = item.get("relevance_score") if item.get("relevance_score") is not None else item.get("score", 0.0)
                results.append({
                    "index": int(item.get("index")),
                    "score": float(score)
                })
            
            results.sort(key=lambda x: x["score"], reverse=True)
            return results

        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Qwen Rerank API 状态错误: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"💥 Qwen Rerank 请求发生未知异常: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """优雅关闭 HTTP 客户端，释放连接池资源。"""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._is_initialized = False
        logger.info("♻️ Qwen Reranker 客户端已成功关闭")

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized