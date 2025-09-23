from typing import Any
from cohere import AsyncClient
from prometheus_client import Histogram, Counter, Gauge
from loguru import logger
import asyncio
from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse, EmbeddingDataItem

class CohereEmbeddingConfig(EmbeddingConfig):
    """Cohere 嵌入服务配置"""
    api_endpoint: str
    timeout: int
    max_retries: int
    api_key: str

class CohereEmbedding(BaseEmbedding):
    """
    生产级 Cohere 嵌入客户端，具备增强特性：
    - 智能批处理
    - 自适应重试机制
    - 全面的监控指标
    - 资源优化
    """

    # Prometheus 监控指标（Cohere 特定维度）
    LATENCY_HIST = Histogram(
        'cohere_embedding_latency_seconds',
        '嵌入请求延迟时间（秒）',
        ['model_name', 'input_type', 'status']
    )
    
    ERROR_COUNTER = Counter(
        'cohere_embedding_errors_total',
        '嵌入错误次数统计',
        ['model_name', 'error_type']
    )
    
    REQUEST_COUNTER = Counter(
        'cohere_embedding_requests_total',
        '嵌入请求次数统计',
        ['model_name', 'input_type']
    )
    
    BATCH_SIZE_GAUGE = Gauge(
        'cohere_embedding_batch_size',
        '使用的有效批处理大小',
        ['model_name']
    )
    
    TOKEN_USAGE = Gauge(
        'cohere_embedding_tokens_estimated',
        '预估消耗的令牌数',
        ['model_name']
    )

    def __init__(self, config: CohereEmbeddingConfig):
        """
        使用 Cohere 特定配置进行初始化
        
        Args:
            config: 包含以下参数的配置对象：
                - api_key: Cohere API 密钥
                - model_name: 模型标识符（例如 "embed-english-v3.0"）
                - timeout: 请求超时时间（秒）
                - default_input_type: 默认输入类型（"search_document"/"search_query"）
                - max_batch_size: 每次 API 调用的最大文本数（默认 96）
                - retry_delay: 重试之间的基础延迟时间（秒）
                - truncate_strategy: 默认截断策略（"END"/"START"/"NONE"）
        """
        self._client: AsyncClient | None = None
        self.api_key = config.api_key
        self.model_name = config.model_name
        self.timeout = config.timeout or 30
        self.default_input_type = getattr(config, 'default_input_type', 'search_document')
        self.max_batch_size = getattr(config, 'max_batch_size', 96)  # Cohere 推荐值
        self.retry_delay = getattr(config, 'retry_delay', 1.0)
        self.truncate_strategy = getattr(config, 'truncate_strategy', 'END')
        self._is_initialized = False

    async def startup(self) -> None:
        """初始化客户端并配置自定义重试处理"""
        if self._is_initialized:
            return
            
        if not self.api_key:
            raise ValueError("必须提供 Cohere API 密钥")

        try:
            # 初始化时不支持 max_retries 参数
            self._client = AsyncClient(
                api_key=self.api_key,
                timeout=self.timeout  # Cohere 客户端只接受 timeout 参数
            )
            
            await self._validate_connection()
            self._is_initialized = True
            logger.info(f"Cohere 客户端初始化完成，模型: {self.model_name}")
            
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise RuntimeError("Cohere 客户端初始化失败") from e

    async def _validate_connection(self) -> None:
        """执行轻量级连接测试"""
        try:
            test_response = await self._client.embed(  # type: ignore
                texts=["连接测试"],
                model=self.model_name,
                input_type=self.default_input_type,
                truncate=self.truncate_strategy
            )
            if not test_response.embeddings:
                raise ValueError("测试响应为空")
        except Exception as e:
            raise RuntimeError(f"连接测试失败: {str(e)}") from e

    async def shutdown(self) -> None:
        """优雅关闭（Cohere 客户端不需要显式关闭）"""
        self._client = None
        self._is_initialized = False
        logger.info("Cohere 客户端关闭完成")

    async def embed(
        self,
        texts: list[str],
        batch_size: int = 0,
        input_type: str | None = None,
        truncate: str | None = None,
        raise_on_error: bool = True,
        **kwargs
    ) -> EmbeddingResponse:
        """
        增强的 Cohere 嵌入功能，具备智能批处理和容错能力
        
        Args:
            texts: 需要嵌入的输入文本列表
            batch_size: 覆盖自动检测的批处理大小（0 表示自动）
            input_type: 覆盖默认输入类型
            truncate: 覆盖默认截断策略
            raise_on_error: 是否在出错时抛出异常
            kwargs: 额外的 Cohere API 参数
            
        Returns:
            EmbeddingResponse: 标准化的响应格式
            
        Raises:
            RuntimeError: 初始化失败时抛出
        """
        if not self._is_initialized:
            raise RuntimeError("客户端未初始化，请先调用 startup() 方法")

        if not texts:
            logger.warning("输入文本列表为空")
            return self._empty_response()

        # 确定有效参数
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
        """使用手动重试逻辑处理批处理"""
        all_embeddings = []
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"正在处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            for attempt in range(3 + 1):  # 手动重试（最大重试次数 = 3）
                try:
                    response = await self._client.embed(  # type: ignore
                        texts=batch,
                        model=self.model_name,
                        input_type=input_type,
                        truncate=truncate,
                        **kwargs
                    )
                    
                    all_embeddings.extend(response.embeddings)
                    # 估算令牌使用量（假设每个单词约2个令牌）
                    total_tokens += sum(len(text.split()) for text in batch) * 2
                    break
                    
                except Exception as e:
                    if attempt == 3:  # 最后一次尝试
                        logger.error(f"批次处理失败，已达到最大重试次数: {str(e)}")
                        raise
                        
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"第 {attempt + 1} 次尝试失败，{wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)

        self.TOKEN_USAGE.labels(model_name=self.model_name).set(total_tokens)
        return self._build_response(all_embeddings, total_tokens)

    def _calculate_batch_size(self, num_texts: int, user_batch_size: int) -> int:
        """考虑 Cohere 限制确定最优批处理大小"""
        if user_batch_size > 0:
            return min(user_batch_size, self.max_batch_size)
            
        # 基于文本数量自动计算
        if num_texts <= 32:  # 小批次以获得更好的延迟
            return num_texts
        return min(
            max(32, num_texts // 4),  # 平衡延迟和吞吐量
            self.max_batch_size
        )

    def _build_response(self, embeddings: list[list[float]], total_tokens: int) -> EmbeddingResponse:
        """构建标准化响应对象"""
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
        """集中错误处理和日志记录"""
        error_type = type(error).__name__
        self.ERROR_COUNTER.labels(
            model_name=self.model_name,
            error_type=error_type
        ).inc()
        
        self.LATENCY_HIST.labels(
            model_name=self.model_name,
            input_type=input_type,
            status="error"
        ).observe(0)  # 记录失败的请求
        
        logger.error(f"嵌入失败 - 模型: {self.model_name}, 错误: {str(error)}")

    def _empty_response(self) -> EmbeddingResponse:
        """为错误情况生成空响应"""
        return EmbeddingResponse(
            data=[],
            model=self.model_name,
            object="list",
            usage={"prompt_tokens": 0, "total_tokens": 0}
        )

    @property
    def embedding_dim(self) -> int:
        """获取配置模型的输出维度"""
        dim_map = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384
        }
        return dim_map.get(self.model_name, 1024)  # 默认返回 1024