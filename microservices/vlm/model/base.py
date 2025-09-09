"""
VLM (视觉语言模型) 基础配置和接口定义。
包含：
1. 基础配置类 VLMConfig
2. 基础接口类 BaseVLM
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel
from prometheus_client import Counter, Histogram
from typing import Any, AsyncGenerator
from openai.types.chat import ChatCompletion, ChatCompletionChunk


class VLMConfig(BaseModel):
    """VLM 模型的基础配置"""   
    model_name: str
    provider: str
    max_tokens: int = 512

class BaseVLM(ABC):
    """VLM 实现的基类"""
    
    LATENCY_HIST = Histogram(
        'vlm_latency_seconds', 
        'VLM 延迟时间（秒）',
        ['model_type']
    )
    ERROR_COUNTER = Counter(
        'vlm_errors_total', 
        'VLM 错误总数', 
        ['provider']
    )
    
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