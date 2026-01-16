"""
LLM基础配置和接口定义。
包含：
1. 基础配置类 LLMConfig
2. 基础接口类 BaseLLM
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, AsyncGenerator, TypeVar, Generic
from openai.types.chat import ChatCompletion, ChatCompletionChunk


class LLMConfig(BaseModel):
    """LLM模型的基础配置类"""
    model_name: str = Field(..., description="模型名称")
    provider: str = Field(..., description="模型提供商")
    max_tokens: int = Field(8192, description="最大令牌数")

T = TypeVar("T", bound=LLMConfig)

class BaseLLM(ABC, Generic[T]):
    """LLM实现的基类"""
    
    def __init__(self, config: T) -> None:
        """使用配置初始化LLM
        
        Args:
            config: LLM配置对象
        """
        self.config: T = config
        self.provider = config.provider
        self.model_name = config.model_name
        self.max_tokens = config.max_tokens
    
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