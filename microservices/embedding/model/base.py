from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram


class EmbeddingConfig(BaseModel):
    """嵌入模型配置"""
    model_name: str = Field(..., description="模型名称")
    provider: str = Field(..., description="服务提供商")
    max_tokens: int = Field(..., description="最大令牌数")
    batch_size: int = Field(..., description="批处理大小")


class EmbeddingDataItem(BaseModel):
    """嵌入数据项"""
    embedding: list[float] = Field(..., description="嵌入向量")
    index: int = Field(..., description="在批次中的索引位置")
    object: str = Field("embedding", description="对象类型，始终为 'embedding'")


class EmbeddingResponse(BaseModel):
    """嵌入响应"""
    data: list[EmbeddingDataItem] = Field(..., description="嵌入数据项列表")
    model: str = Field(..., description="使用的嵌入模型名称")
    object: str = Field("list", description="对象类型，始终为 'list'")
    usage: dict[str, int] = Field(..., description="令牌使用信息")


class BaseEmbedding(ABC):
    """
    嵌入模型抽象基类
    定义所有嵌入模型实现的标准接口
    """
    
    # 监控指标
    LATENCY_HIST = Histogram(
        'embedding_latency_seconds', 
        '嵌入延迟时间（秒）',
        ['model_type']
    )
    ERROR_COUNTER = Counter(
        'embedding_errors_total', 
        '嵌入错误总数', 
        ['provider']
    )

    @abstractmethod
    async def startup(self) -> None:
        """
        初始化嵌入模型并创建客户端
        
        Raises:
            RuntimeError: 初始化失败时抛出
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        关闭嵌入模型和客户端连接
        
        Raises:
            RuntimeError: 关闭过程中发生错误时抛出
        """
        pass
    
    @abstractmethod
    async def embed(self, texts: list[str], batch_size: int | None = None, is_query: bool = True) -> EmbeddingResponse:
        """
        为文本列表生成嵌入向量，遵循 OpenAI 标准格式
        
        Args:
            texts: 需要嵌入的文本列表
            batch_size: 批处理大小，如为 None 则使用默认值
            is_query: 是否为查询文本，默认为 True
            
        Returns:
            EmbeddingResponse: 符合 OpenAI 标准的响应对象，包含：
                - data: 嵌入数据项列表（包含向量、索引和对象类型）
                - model: 使用的模型名称
                - object: 始终为 "list"
                - usage: 令牌使用信息
                
        Raises:
            ValueError: 输入文本为空或无效时抛出
            RuntimeError: 模型未初始化或处理过程中发生错误时抛出
            RateLimitError: 达到速率限制时抛出
            APIError: API 调用失败时抛出
        """
        pass