from typing import Any
from openai import AsyncAzureOpenAI, APIConnectionError, RateLimitError, APIStatusError
from prometheus_client import Histogram, Counter, Gauge
from loguru import logger
import asyncio
from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse, EmbeddingDataItem

class AzureEmbeddingConfig(EmbeddingConfig):
    """Azure OpenAI 嵌入服务配置类"""
    api_endpoint: str  # API 端点地址
    timeout: int = 30  # 请求超时时间（秒）
    max_retries: int = 3  # 最大重试次数
    api_key: str = ""  # API 密钥
    deployment_name: str = ""  # 部署名称
    api_version: str = "2023-05-15"  # API 版本
    
class AzureEmbedding(BaseEmbedding):
    """
    生产级 Azure OpenAI 嵌入服务，具备以下特性：
    - 智能批处理
    - 自适应重试策略
    - 全面的监控指标
    - Azure 特定优化
    """

    # 增强的监控指标（包含 Azure 特定维度）
    LATENCY_HIST = Histogram(
        'azure_embedding_latency_seconds',
        '嵌入请求延迟分布',
        ['deployment', 'api_version', 'status']
    )
    
    ERROR_COUNTER = Counter(
        'azure_embedding_errors_total',
        '按类型统计的嵌入错误次数',
        ['deployment', 'error_code']
    )
    
    REQUEST_COUNTER = Counter(
        'azure_embedding_requests_total',
        '处理的嵌入请求总数',
        ['deployment', 'api_version']
    )
    
    BATCH_SIZE_GAUGE = Gauge(
        'azure_embedding_batch_size',
        '每个请求的有效批处理大小',
        ['deployment']
    )
    
    TOKEN_USAGE = Gauge(
        'azure_embedding_tokens_used',
        '每个请求消耗的令牌数',
        ['deployment']
    )

    def __init__(self, config: AzureEmbeddingConfig):
        """
        使用 Azure 特定配置进行初始化
        
        Args:
            config: 包含以下参数的配置对象：
                - api_key: Azure API 密钥
                - deployment_name: 部署名称
                - endpoint: Azure 端点 URL
                - api_version: API 版本（默认 "2023-05-15"）
                - timeout: 请求超时时间（默认 30 秒）
                - max_retries: 最大重试次数（默认 3 次）
                - max_batch_size: 每个请求的最大文本数（默认 16）
                - min_batch_size: 每个请求的最小文本数（默认 1）
                - retry_delay: 基础重试延迟（默认 1.0 秒）
                - headers: 自定义 HTTP 头部
                - azure_params: 额外的 Azure 参数
        """
        self._client: AsyncAzureOpenAI | None = None
        self.api_key = config.api_key
        self.deployment_name = config.deployment_name
        self.endpoint = config.api_endpoint
        self.api_version = config.api_version or "2023-05-15"
        self.timeout = config.timeout
        self.max_retries = getattr(config, 'max_retries', 3)
        self.max_batch_size = getattr(config, 'max_batch_size', 16)  # Azure 推荐值
        self.min_batch_size = getattr(config, 'min_batch_size', 1)
        self.retry_delay = getattr(config, 'retry_delay', 1.0)
        self.custom_headers = getattr(config, 'headers', {})
        self._azure_params = getattr(config, 'azure_params', {})
        self._is_initialized = False

    async def startup(self) -> None:
        """初始化客户端并进行连接验证"""
        if self._is_initialized:
            return
            
        if not all([self.api_key, self.endpoint, self.deployment_name]):
            raise ValueError("缺少必要的 Azure 配置参数")

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
            logger.success(f"Azure 客户端已就绪，部署名称: {self.deployment_name}")
            
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise RuntimeError("Azure 客户端初始化失败") from e

    async def _validate_connection(self) -> None:
        """执行轻量级连接测试"""
        try:
            test_response = await self._client.embeddings.create(  # type: ignore
                model=self.deployment_name,
                input=["连接测试"],
                encoding_format="float"
            )
            if not test_response.data:
                raise ValueError("测试响应为空")
        except Exception as e:
            await self._client.close()  # type: ignore
            raise RuntimeError(f"连接测试失败: {str(e)}") from e

    async def shutdown(self) -> None:
        """优雅关闭，清理资源"""
        if not self._is_initialized:
            return
            
        try:
            if self._client:
                await self._client.close()
            self._client = None
            self._is_initialized = False
            logger.info("Azure 客户端关闭完成")
        except Exception as e:
            logger.error(f"关闭过程中发生错误: {str(e)}")
            raise

    async def embed(
        self,
        texts: list[str],
        batch_size: int = 0,
        raise_on_error: bool = True,
        **kwargs: Any
    ) -> EmbeddingResponse:
        """
        使用 Azure 特定优化生成嵌入向量
        
        Args:
            texts: 需要嵌入的输入文本列表
            batch_size: 覆盖自动批处理大小（0 表示自动）
            raise_on_error: 是否在出错时抛出异常
            kwargs: 额外的 Azure API 参数
            
        Returns:
            EmbeddingResponse: 标准化的响应对象
            
        Raises:
            RuntimeError: 如果客户端未初始化
        """
        if not self._is_initialized:
            raise RuntimeError("客户端未初始化，请先调用 startup() 方法")

        if not texts:
            logger.warning("接收到空的输入文本")
            return self._empty_response()

        # 计算有效的批处理大小
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
        """使用 Azure 特定的重试逻辑处理批处理"""
        all_embeddings = []
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"正在处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
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
                    logger.warning(f"受到速率限制，将在 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                except APIConnectionError:
                    if attempt == self.max_retries:
                        raise
                    await asyncio.sleep(self.retry_delay)
                except APIStatusError as e:
                    if e.status_code >= 500:  # 重试服务器错误
                        if attempt == self.max_retries:
                            raise
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        raise

        self.TOKEN_USAGE.labels(deployment=self.deployment_name).set(total_tokens)
        return self._build_response(all_embeddings, total_tokens)

    def _calculate_batch_size(self, num_texts: int, user_batch_size: int) -> int:
        """考虑 Azure 限制计算最优批处理大小"""
        if user_batch_size > 0:
            return min(user_batch_size, self.max_batch_size)
            
        # 基于文本长度自动计算
        avg_length = sum(len(t) for t in texts) / max(1, len(texts)) # type: ignore
        if avg_length > 1000:  # 对于长文档减少批处理大小
            return min(8, self.max_batch_size)
        return min(
            max(self.min_batch_size, num_texts // 4),
            self.max_batch_size
        )

    def _build_response(self, embeddings: list[list[float]], total_tokens: int) -> EmbeddingResponse:
        """构建标准化响应"""
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
        """集中错误处理"""
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
        
        logger.error(f"嵌入失败 - 部署: {self.deployment_name}, 错误: {str(error)}")

    def _empty_response(self) -> EmbeddingResponse:
        """为错误情况生成空响应"""
        return EmbeddingResponse(
            data=[],
            model=self.deployment_name,
            object="list",
            usage={"prompt_tokens": 0, "total_tokens": 0}
        )

    @property
    def embedding_dim(self) -> int:
        """获取部署的嵌入维度"""
        dim_map = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        return dim_map.get(self.deployment_name.split('-')[0], 1536)  # 默认回退值
