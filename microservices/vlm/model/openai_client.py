from typing import Any, AsyncGenerator
from openai import AsyncOpenAI, APIError
from loguru import logger
from pydantic import Field
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .base import BaseVLM, VLMConfig


class OpenAIVLMConfig(VLMConfig):
    """OpenAI VLM 客户端配置"""
    api_key: str = Field(..., description="API 密钥")
    api_endpoint: str | None = Field(None, description="API 基础地址")
    api_version: str = "2023-08-01"
    max_retries: int = 3
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="采样温度")
    timeout: int = Field(30, ge=1, description="请求超时时间")


class OpenAIVLM(BaseVLM[OpenAIVLMConfig]):
    """
    使用 OpenAI SDK 实现的多模态视觉模型客户端
    兼容 OpenAI GPT-4o, 阿里 DashScope Qwen-VL 等兼容接口
    """

    def __init__(self, config: OpenAIVLMConfig):
        super().__init__(config)
        self._openai_client: AsyncOpenAI | None = None
        self._is_initialized: bool = False

    async def startup(self) -> None:
        """初始化 OpenAI 异步客户端"""
        if self._is_initialized:
            return

        try:
            self._openai_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_endpoint,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
            self._is_initialized = True
            logger.info(f"✅ OpenAIVLM 客户端就绪: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ 初始化 OpenAI 客户端失败: {str(e)}")
            raise

    async def inference(
        self,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None]:
        """
        执行多模态推理
        """
        if not self._is_initialized:
            await self.startup()

        if not self._validate_messages(messages):
            raise ValueError("无效的 OpenAI VLM 消息格式，请确保 content 为包含 type: text/image_url 的列表")

        # 1. 动态参数聚合
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p"),
            "presence_penalty": kwargs.get("presence_penalty"),
            "frequency_penalty": kwargs.get("frequency_penalty"),
            "response_format": kwargs.get("response_format"), # 支持 JSON Mode
        }

        # 2. 清理 None 值以防 SDK 报错
        api_params = {k: v for k, v in payload.items() if v is not None}

        try:
            logger.debug(f"🚀 VLM 推理开始: {self.model_name} (Stream={stream})")
            response = await self._openai_client.chat.completions.create(**api_params) # type: ignore
            return response
        except APIError as e:
            logger.error(f"❌ OpenAI API 错误: {e.code} - {e.message}")
            raise
        except Exception as e:
            logger.error(f"💥 VLM 推理过程中发生意外错误: {str(e)}")
            raise

    def _validate_messages(self, messages: list[dict[str, Any]]) -> bool:
        """验证消息是否符合 VLM 视觉模型规范"""
        try:
            for msg in messages:
                # 基础结构校验
                if not isinstance(msg, dict) or not {"role", "content"}.issubset(msg.keys()):
                    return False
                if msg["role"] not in {"user", "system", "assistant"}:
                    return False
                
                # VLM content 必须是列表格式
                content = msg["content"]
                if not isinstance(content, list):
                    return False
                
                for item in content:
                    if not isinstance(item, dict) or "type" not in item:
                        return False
                    
                    item_type = item["type"]
                    if item_type == "text":
                        if "text" not in item: return False
                    elif item_type == "image_url":
                        if "image_url" not in item or "url" not in item["image_url"]:
                            return False
                    else:
                        # 不支持的 content type
                        return False
            return True
        except Exception:
            return False

    async def shutdown(self) -> None:
        """资源释放"""
        if self._openai_client:
            await self._openai_client.close()
            self._openai_client = None
        self._is_initialized = False
        logger.info("♻️ OpenAIVLM 客户端资源已回收")

    async def health_check(self) -> dict[str, Any]:
        """增强的健康检查"""
        return {
            "status": "healthy" if self._is_initialized else "uninitialized",
            "model": self.model_name,
            "provider": "OpenAI-Compatible VLM",
            "initialized": self._is_initialized
        }