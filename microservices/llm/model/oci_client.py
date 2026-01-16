import oci
import json
import asyncio
from typing import Any
from pydantic import Field
from loguru import logger

from .base import LLMConfig, BaseLLM

class OCILLMConfig(LLMConfig):
    """OCI LLM客户端配置"""
    temperature: float | None = Field(None, ge=0, le=2, description="温度参数，控制输出随机性")
    top_p: float | None = Field(None, ge=0, le=1, description="Top-p 采样参数")
    top_k: int | None = Field(None, ge=0, description="Top-k 采样参数")
    frequency_penalty: float | None = Field(None, ge=-2, le=2, description="频率惩罚参数")
    presence_penalty: float | None = Field(None, ge=-2, le=2, description="存在惩罚参数")
    api_endpoint: str = Field(..., description="OCI Generative AI Endpoint")
    compartment_id: str = Field(..., description="OCI Compartment OCID")
    config_file: dict | str = Field(..., description="OCI Auth Config (dict or JSON string)")

class OCIClient(BaseLLM[OCILLMConfig]):
    """
    针对 OCI Generative AI 优化的 LLM 实现
    支持 Cohere, Llama 3, Grok 等多种模型格式适配
    """
    
    def __init__(self, config: OCILLMConfig):
        super().__init__(config)
        self.client: oci.generative_ai_inference.GenerativeAiInferenceClient | None = None
        self._is_initialized = False
        self.config = config

    async def startup(self) -> None:
        """异步初始化 OCI 客户端"""
        if self._is_initialized:
            return

        try:
            # 1. 自动解析配置
            oci_config = self.config.config_file
            if isinstance(oci_config, str):
                oci_config = json.loads(oci_config)
            
            # 2. 初始化推理客户端
            # 注意：OCI SDK 是同步的，连接过程通常很快，但在高并发场景下需注意
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=oci_config,
                service_endpoint=self.config.api_endpoint,
                retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY, # 使用默认重试策略
                timeout=(10, 240)
            )
            self._is_initialized = True
            logger.info(f"✅ OCI LLM 客户端初始化成功 (Model: {self.config.model_name})")
        except Exception as e:
            logger.error(f"❌ OCI 初始化失败: {e}")
            raise

    def _convert_to_oci_messages(self, messages: list[dict[str, str]] | str) -> list[Any]:
        """将标准消息格式转换为 OCI Message 对象"""
        if isinstance(messages, str):
            messages = [{"role": "USER", "content": messages}]
        
        oci_msgs = []
        for msg in messages:
            content = oci.generative_ai_inference.models.TextContent()
            content.text = msg.get("content", "")
            
            oci_msg = oci.generative_ai_inference.models.Message()
            # OCI 角色通常为 USER 或 ASSISTANT (大写)
            oci_msg.role = msg.get("role", "USER").upper()
            oci_msg.content = [content]
            oci_msgs.append(oci_msg)
        return oci_msgs

    def _build_chat_request(self, messages: list[dict[str, str]] | str, **kwargs) -> Any:
        """根据模型类型构建特定的请求对象"""
        model_name = self.config.model_name.lower()
        
        # 基础参数提取
        params = {
            "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
            "temperature": kwargs.get('temperature', self.config.temperature),
            "top_p": kwargs.get('top_p', self.config.top_p),
            "top_k": kwargs.get('top_k', self.config.top_k),
        }

        # 1. Cohere 模型适配
        if "cohere" in model_name:
            request = oci.generative_ai_inference.models.CohereChatRequest()
            # Cohere 在 OCI 上通常接受单一 message 字符串
            request.message = messages[-1]['content'] if isinstance(messages, list) else messages
            request.frequency_penalty = kwargs.get('frequency_penalty', self.config.frequency_penalty)
            params["max_tokens"] = min(params["max_tokens"], 4000)
            
        # 2. Llama / Grok / Generic 模型适配
        else:
            request = oci.generative_ai_inference.models.GenericChatRequest()
            request.api_format = oci.generative_ai_inference.models.GenericChatRequest.API_FORMAT_GENERIC
            request.messages = self._convert_to_oci_messages(messages)
            
            if "llama" in model_name:
                params["max_tokens"] = min(params["max_tokens"], 4096) # Llama 3 限制较宽
            elif "grok" in model_name:
                params["max_tokens"] = min(params["max_tokens"], 20000)

        # 注入通用参数
        for key, value in params.items():
            if value is not None and hasattr(request, key):
                setattr(request, key, value)
        
        return request

    async def chat(
        self,
        messages: list[dict[str, str]] | str,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """异步生成聊天响应"""
        if not self._is_initialized:
            await self.startup()

        if self.client is None:
            raise ValueError("OCI 客户端未初始化")

        # 1. 构建请求细节
        chat_request = self._build_chat_request(messages, **kwargs)
        chat_request.is_stream = stream

        chat_detail = oci.generative_ai_inference.models.ChatDetails()
        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=self.config.model_name
        )
        chat_detail.chat_request = chat_request
        chat_detail.compartment_id = self.config.compartment_id

        try:
            # 2. 异步执行同步 SDK 调用
            # OCI SDK 本身不支持 await，使用 run_in_executor 防止阻塞事件循环
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.chat(chat_detail) # type: ignore
            )
            return response
        except Exception as e:
            logger.error(f"❌ OCI 生成响应失败: {e}")
            raise

    async def shutdown(self) -> None:
        self.client = None
        self._is_initialized = False
        logger.info("♻️ OCI LLM 客户端已关闭")