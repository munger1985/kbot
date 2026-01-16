"""
VLM (视觉语言模型) 基础配置和接口定义。
包含：
1. 基础配置类 VLMConfig
2. 基础接口类 BaseVLM
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, AsyncGenerator, TypeVar, Generic
from openai.types.chat import ChatCompletion, ChatCompletionChunk


class VLMConfig(BaseModel):
    """VLM 模型的基础配置"""   
    model_name: str = Field(..., description="模型名称")
    provider: str = Field(..., description="模型提供商")
    max_tokens: int = Field(8192, description="最大令牌数")

T = TypeVar("T", bound=VLMConfig)

class BaseVLM(ABC, Generic[T]):
    """VLM 实现的基类"""
    def __init__(self, config: T) -> None:
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
    async def inference(self, messages: list[dict[str, Any]], 
                        stream: bool = False, 
                        **kwargs) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None] | None:
        """
        执行推理任务

        参数:
            messages: 消息字典列表，每个字典包含：
                {
                    "role": str,   # "user"、"system" 或 "assistant" 之一
                    "content": [
                        {
                            "type": "text",
                            "text": str
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": str
                            }
                        }
                    ]
                }
            stream: 如果为 True，结果将以流式方式返回
            **kwargs: 推理模型的额外参数
        
        返回:
            生成的文本块或流式输出
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """对远程或本地模型进行健康检查
        
        返回:
            包含健康状态信息的字典
        """
        pass