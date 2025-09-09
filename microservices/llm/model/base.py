"""
LLM基础配置和接口定义。
包含：
1. 基础配置类 LLMConfig
2. 基础接口类 BaseLLM
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, AsyncGenerator
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from prometheus_client import Counter, Histogram


class LLMConfig(BaseModel):
    """LLM模型的基础配置类"""
    model_name: str
    provider: str


class BaseLLM(ABC):
    """LLM实现的基类"""
    
    # Prometheus监控指标
    ERROR_COUNTER = Counter('llm_errors', '按提供商统计的错误次数', ['provider'])
    LATENCY_HIST = Histogram('llm_latency', '生成延迟', ['model_type'])
    
    def __init__(self, config: LLMConfig) -> None:
        """使用配置初始化LLM
        
        Args:
            config: LLM配置对象
        """
        self.config = config
        self.provider = config.provider
    
    @abstractmethod
    async def startup(self) -> None:
        """异步初始化资源"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """异步释放资源"""
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]] | str,
        stream: bool = False,
        **kwargs: Any
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None] | None:
        """异步生成聊天响应
        
        Args:
            messages: 消息列表或单条消息字符串
            stream: 是否使用流式输出
            **kwargs: 其他生成参数
            
        Returns:
            聊天完成对象或异步生成器
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError