from openai import AsyncOpenAI, APIError
from loguru import logger
from typing import AsyncGenerator

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam
)
from .base import LLMConfig, BaseLLM


class OpenaiLLMConfig(LLMConfig):
    """OpenAI LLM客户端配置"""
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    timeout: int | None = None
    api_endpoint: str
    api_key: str


class OpenaiClient(BaseLLM):
    """OpenAI LLM客户端实现"""
    
    def __init__(self, config: OpenaiLLMConfig):
        """初始化OpenAI LLM客户端
        
        Args:
            config: OpenAI LLM配置对象
        """
        super().__init__(config)
        self.client = None
        self._is_running = False
    
    async def startup(self) -> None:
        """初始化OpenAI客户端"""
        try:
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,  # type: ignore
                base_url=self.config.api_endpoint,  # type: ignore
                timeout=self.config.timeout  # type: ignore
            )
            self._is_running = True
            logger.info("OpenAI客户端初始化成功")
        except Exception as e:
            logger.error(f"初始化OpenAI客户端时出错: {str(e)}")
            raise
        
    async def shutdown(self) -> None:
        """关闭OpenAI客户端"""
        if self.client:
            await self.client.close()
        self._is_running = False
        logger.info("OpenAI客户端已关闭")

    
    async def chat(
        self,
        messages: list[dict[str, str]] | str,
        stream: bool = False,
        **kwargs
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None] | None:
        """生成聊天响应（保持一致的返回类型）
        
        Args:
            messages: 消息列表或单个提示字符串
            stream: 是否使用流式输出
            **kwargs: 额外的生成参数
            
        Returns:
            - 如果不使用流式: ChatCompletion对象
            - 如果使用流式: 生成ChatCompletionChunk对象的异步生成器
            
        Raises:
            APIError: OpenAI API调用错误
            Exception: 生成响应时出错
        """
        if not self._is_running:
            await self.startup()
        
        # 如果需要，将字符串转换为消息列表
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # 将消息转换为正确的格式
        converted_messages = [self._convert_message(msg) for msg in messages]

        # 合并配置和任何覆盖参数
        params = {
            "model": self.config.model_name,
            "messages": converted_messages,
            "temperature": kwargs.get('temperature', self.config.temperature),  # type: ignore
            "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),  # type: ignore
            "top_p": kwargs.get('top_p', self.config.top_p),  # type: ignore
            "frequency_penalty": kwargs.get('frequency_penalty', self.config.frequency_penalty),  # type: ignore
            "presence_penalty": kwargs.get('presence_penalty', self.config.presence_penalty),  # type: ignore
            "stream": stream,
        }
        
        try:
            response = await self.client.chat.completions.create(**params)  # type: ignore
            return response
            
        except APIError as e:
            self.ERROR_COUNTER.labels(provider="openai").inc()
            logger.error(f"API错误: {e.code} - {e.message}")
            raise
        except Exception as e:
            self.ERROR_COUNTER.labels(provider="openai").inc()
            logger.error(f"使用OpenAI生成聊天响应时出错: {str(e)}")
            raise Exception(f"使用OpenAI生成聊天响应时出错: {str(e)}")

    def _convert_message(self, msg: dict):
        """将消息字典转换为OpenAI消息参数对象
        
        Args:
            msg: 消息字典，包含role和content
            
        Returns:
            OpenAI消息参数对象
            
        Raises:
            ValueError: 未知的角色类型
        """
        if msg['role'] == 'user':
            return ChatCompletionUserMessageParam(role='user', content=msg['content'])
        elif msg['role'] == 'system':
            return ChatCompletionSystemMessageParam(role='system', content=msg['content'])
        elif msg['role'] == 'assistant':
            return ChatCompletionAssistantMessageParam(role='assistant', content=msg['content'])
        else:
            raise ValueError(f"未知的角色类型: {msg['role']}")