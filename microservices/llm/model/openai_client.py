from typing import AsyncGenerator
from openai import AsyncOpenAI, APIError
from loguru import logger
from pydantic import Field

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam
)
from .base import LLMConfig, BaseLLM


class OpenaiLLMConfig(LLMConfig):
    """OpenAI LLM 客户端配置"""
    temperature: float | None = Field(None, ge=0, le=2, description="温度参数，控制输出随机性")
    top_p: float | None = Field(None, ge=0, le=1, description="Top-p 采样参数")
    frequency_penalty: float | None = Field(None, ge=-2, le=2, description="频率惩罚参数")
    presence_penalty: float | None = Field(None, ge=-2, le=2, description="存在惩罚参数")
    timeout: int | None = Field(None, gt=0, description="请求超时时间")
    api_endpoint: str = Field(..., description="API 基础地址")
    api_key: str = Field(..., description="API 密钥")


class OpenaiClient(BaseLLM[OpenaiLLMConfig]):
    """
    OpenAI LLM 客户端
    优化点：动态参数解构、更优雅的消息处理、完善的日志追踪
    """
    
    def __init__(self, config: OpenaiLLMConfig):
        super().__init__(config)
        self._client: AsyncOpenAI | None = None
        self._is_initialized = False
    
    async def startup(self) -> None:
        """异步初始化客户端"""
        if self._is_initialized:
            return

        try:
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_endpoint,
                timeout=self.config.timeout
            )
            self._is_initialized = True
            logger.info(f"✅ OpenAI 客户端就绪: {self.config.api_endpoint}")
        except Exception as e:
            logger.error(f"❌ OpenAI 初始化失败: {e}")
            raise
        
    async def shutdown(self) -> None:
        """关闭客户端资源"""
        if self._client:
            await self._client.close()
            self._client = None
        self._is_initialized = False
        logger.info("♻️ OpenAI 客户端已安全关闭")

    async def chat(
        self,
        messages: list[dict[str, str]] | str,
        stream: bool = False,
        **kwargs
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None]:
        """
        统一聊天接口
        """
        if not self._is_initialized:
            await self.startup()

        if self._client is None:
            raise ValueError("OpenAI 客户端未初始化")

        # 1. 消息格式标准化
        prepared_messages = self._prepare_messages(messages)

        # 2. 参数聚合（优先级：kwargs > config）
        # 使用字典解构简化代码，只保留非 None 的有效参数
        base_params = {
            "model": self.config.model_name,
            "messages": prepared_messages,
            "stream": stream,
            "temperature": kwargs.get('temperature', self.config.temperature),
            "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
            "top_p": kwargs.get('top_p', self.config.top_p),
            "frequency_penalty": kwargs.get('frequency_penalty', self.config.frequency_penalty),
            "presence_penalty": kwargs.get('presence_penalty', self.config.presence_penalty),
            "tools": kwargs.get('tools'),
            "tool_choice": kwargs.get('tool_choice'),
            "response_format": kwargs.get('response_format'), # 增加对 JSON Mode 的支持
        }
        
        # 过滤 None 值
        api_params = {k: v for k, v in base_params.items() if v is not None}

        try:
            logger.debug(f"🚀 发送请求 [{self.config.model_name}] - Stream: {stream}")
            response = await self._client.chat.completions.create(**api_params)

            # 3. 针对非流式结果的日志增强
            if not stream:
                self._log_completion_info(response) # type: ignore
            
            return response
            
        except APIError as e:
            logger.error(f"❌ OpenAI API 异常: {e.code} | {e.message}")
            raise
        except Exception as e:
            logger.error(f"💥 未知生成错误: {str(e)}")
            raise

    def _prepare_messages(self, messages: list[dict[str, str]] | str) -> list[ChatCompletionMessageParam]:
        """将输入转换为 OpenAI 规范的 Message 对象列表"""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        
        # 验证并转换格式
        processed = []
        for m in messages:
            role = m.get("role", "user")
            if role not in ["system", "user", "assistant", "tool"]:
                logger.warning(f"检测到非标准角色: {role}，已自动更正为 user")
                role = "user"
            processed.append({"role": role, "content": m.get("content", "")})
        return processed # type: ignore

    def _log_completion_info(self, response: ChatCompletion) -> None:
        """记录响应的元数据（Token 消耗、工具调用等）"""
        usage = response.usage
        if usage:
            logger.info(f"📊 Tokens: {usage.prompt_tokens}(P) + {usage.completion_tokens}(C) = {usage.total_tokens}")
        
        message = response.choices[0].message
        if message.tool_calls:
            logger.info(f"🛠️ 触发工具调用: {[tc.function.name for tc in message.tool_calls]}")

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized