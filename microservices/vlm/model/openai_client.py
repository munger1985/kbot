from openai import AsyncOpenAI
from loguru import logger
from typing import Any, AsyncGenerator
from pydantic import field_validator
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from .base import BaseVLM, VLMConfig


class OpenAIVLMConfig(VLMConfig):
    """云 API 配置"""
    api_key: str    # 云端服务必需
    api_endpoint: str | None = None
    api_version: str = "2023-08-01"
    max_retries: int = 3
    temperature: float = 0.7    # 覆盖基础温度值
    timeout: int = 30   # 秒

    @field_validator('temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('温度值必须在 0 到 2 之间')
        return v

class OpenAIVLM(BaseVLM):
    """
    使用 OpenAI SDK 实现的 Open AI 云 API
    支持与阿里云 DashScope 兼容的 OpenAI 兼容模式
    """

    def __init__(self, config: OpenAIVLMConfig):
        if not isinstance(config, OpenAIVLMConfig):
            raise TypeError("配置必须是 RemoteVLMConfig 类型")

        self.config = config
        self._is_initialized = False
        self._openai_client: AsyncOpenAI | None = None

    async def startup(self) -> None:
        """初始化 OpenAI 客户端"""
        try:
            self._openai_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_endpoint,
                timeout=self.config.timeout if hasattr(self.config, 'timeout') else 30
            )
            self._is_initialized = True
            logger.info("OpenAI AsyncOpenAI 客户端已初始化")
        except Exception as e:
            logger.error(f"初始化 OpenAI 客户端时出错: {str(e)}")
            raise

    async def inference(
        self,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None] | None:
        """使用 OpenAI SDK 执行云端推理
        
        参数:
            messages: 严格遵循 OpenAI 格式的消息列表:
                - 文本: {"type": "text", "text": "..."}
                - 图片 URL: {"type": "image_url", "image_url": {"url": "..."}}
            stream: 是否流式返回响应
            **kwargs: 额外的生成参数
            
        异常:
            ValueError: 如果消息格式不符合 OpenAI 要求
        """
        if not self._validate_messages(messages):
            raise ValueError("无效的 OpenAI 消息格式")

        try:
            response = await self._openai_client.chat.completions.create( # type: ignore
                model=self.config.model_name,
                messages=messages, # type: ignore
                stream=stream,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", None),
                presence_penalty=kwargs.get("presence_penalty", None),
                frequency_penalty=kwargs.get("frequency_penalty", None)
            )

            return response # type: ignore
        except Exception as e:
            logger.error(f"OpenAI SDK 错误: {str(e)}")
            raise


    def _validate_messages(self, messages: list[dict[str, Any]]) -> bool:
        """验证消息是否符合 VLM 模型的 OpenAI 格式"""
        required_keys = {"role", "content"}
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if not required_keys.issubset(msg.keys()):
                return False
            if msg["role"] not in {"user", "system", "assistant"}:
                return False
            
            # 验证 VLM 模型的内容结构
            if not isinstance(msg["content"], list):
                return False
            
            for content_item in msg["content"]:
                if not isinstance(content_item, dict):
                    return False
                if "type" not in content_item:
                    return False
                
                if content_item["type"] == "text":
                    if "text" not in content_item:
                        return False
                elif content_item["type"] == "image_url":
                    if "image_url" not in content_item or not isinstance(content_item["image_url"], dict):
                        return False
                    if "url" not in content_item["image_url"]:
                        return False
                else:
                    return False
        return True

    async def shutdown(self) -> None:
        """清理 OpenAI 客户端资源"""
        if self._openai_client:
            await self._openai_client.close()
            self._openai_client = None
        self._is_initialized = False
        logger.debug("OpenAI 客户端已关闭")

    async def health_check(self) -> dict[str, Any]:
        """检查服务健康状态"""
        return {
            "initialized": self._is_initialized,
            "model": self.config.model_name,
            "last_error": None,
            "throughput": "N/A"  # 可以跟踪实际指标
        }