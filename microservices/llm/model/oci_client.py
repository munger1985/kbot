import oci
import json
from loguru import logger
from .base import LLMConfig, BaseLLM


class OCILLMConfig(LLMConfig):
    """OCI LLM客户端配置"""
    temperature: float = 1.0
    max_tokens: int = 600
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float = 0
    presence_penalty: float = 0
    api_endpoint: str
    compartment_id: str
    config_file: dict


class OCIClient(BaseLLM):
    """OCI LLM客户端实现"""
    
    def __init__(self, config: OCILLMConfig):
        """初始化OCI LLM客户端
        
        Args:
            config: OCI LLM配置对象
        """
        super().__init__(config)
        self.client = None
        self._is_running = False
    
    async def startup(self) -> None:
        """初始化OCI客户端"""
        try:
            # 如果需要，将config_file从JSON字符串解析为字典
            if isinstance(self.config.config_file, str):  # type: ignore
                oci_config = json.loads(self.config.config_file)  # type: ignore
            else:
                oci_config = self.config.config_file  # type: ignore
            
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=oci_config,
                service_endpoint=self.config.api_endpoint,  # type: ignore
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240))
            self._is_running = True
            logger.info("OCI客户端初始化成功")
        except Exception as e:
            logger.error(f"初始化OCI客户端时出错: {str(e)}")
            raise RuntimeError(f"初始化OCI客户端时出错: {str(e)}")
        
    async def shutdown(self) -> None:
        """关闭OCI客户端"""
        if self.client:
            self.client = None
        self._is_running = False
        logger.info("OCI客户端已关闭")

    
    async def chat(
        self,
        messages: list[dict[str, str]] | str,
        stream: bool = False,
        **kwargs
    ):
        """生成聊天响应（保持一致的返回类型）
        
        Args:
            messages: 消息列表或单个提示字符串
            stream: 是否使用流式输出
            **kwargs: 额外的生成参数
        
        Returns:
            聊天响应对象
        
        Notes:
            此方法需要OCI聊天服务支持
        
        Raises:
            ValueError: 未提供消息或提供商不支持
            Exception: 生成响应时出错
        """
        
        if not self._is_running:
            await self.startup()
        
        converted_messages = ""
        # 如果需要，将字符串转换为消息列表
        if isinstance(messages, list):
            converted_messages = messages[0]['content'] if len(messages) > 0 else ''
        
        if isinstance(messages, str):
            converted_messages = messages
        
        if converted_messages == '':
            raise ValueError("未提供消息用于生成响应")
        
        # OCI max_tokens的最大限制为4000
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)  # type: ignore

        if "cohere" in self.config.model_name.lower():
            chat_request = oci.generative_ai_inference.models.CohereChatRequest()
            chat_request.message = converted_messages
            chat_request.frequency_penalty = kwargs.get('frequency_penalty', self.config.frequency_penalty)  # type: ignore
            chat_request.top_p = kwargs.get('top_p', self.config.top_p)  # type: ignore
            chat_request.top_k = kwargs.get('top_k', self.config.top_k)  # type: ignore

            if max_tokens and max_tokens > 4000:
                max_tokens = 4000
                logger.warning("OCI cohere模型max_tokens超过限制，已设置为4000")

        elif "grok" in self.config.model_name.lower():
            chat_request = oci.generative_ai_inference.models.GenericChatRequest()
            chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
            content = oci.generative_ai_inference.models.TextContent()
            content.text = converted_messages
            message = oci.generative_ai_inference.models.Message()
            message.role = "USER"
            message.content = [content]
            chat_request.messages = [message]
            
            chat_request.top_p = kwargs.get('top_p', self.config.top_p)  # type: ignore
            chat_request.top_k = kwargs.get('top_k', self.config.top_k)  # type: ignore
            if max_tokens and max_tokens > 20000:
                max_tokens = 20000
                logger.warning("OCI grok模型max_tokens超过限制，已设置为20000")

        elif "llama" in self.config.model_name.lower():
            chat_request = oci.generative_ai_inference.models.GenericChatRequest()
            chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
            content = oci.generative_ai_inference.models.TextContent()
            content.text = converted_messages
            message = oci.generative_ai_inference.models.Message()
            message.role = "USER"
            message.content = [content]
            chat_request.messages = [message]
            
            chat_request.frequency_penalty = kwargs.get('frequency_penalty', self.config.frequency_penalty)  # type: ignore
            chat_request.presence_penalty = kwargs.get('presence_penalty', self.config.presence_penalty)  # type: ignore
            chat_request.top_p = kwargs.get('top_p', self.config.top_p)  # type: ignore
            if max_tokens and max_tokens > 600:
                max_tokens = 600
                logger.warning("OCI llama模型max_tokens超过限制，已设置为600")
        else:
            raise ValueError(f"不支持的OCI提供商: {self.provider}")

        chat_request.max_tokens = max_tokens
        chat_request.temperature = kwargs.get('temperature', self.config.temperature)  # type: ignore
        chat_request.is_stream = stream

        chat_detail = oci.generative_ai_inference.models.ChatDetails()
        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=self.config.model_name)
        chat_detail.chat_request = chat_request
        chat_detail.compartment_id = self.config.compartment_id  # type: ignore

        try:
            response = self.client.chat(chat_detail)  # type: ignore
            
            return response

        except Exception as e:
            self.ERROR_COUNTER.labels(provider="OCI").inc()
            logger.error(f"使用OCI生成聊天响应时出错: {str(e)}")
            raise Exception(f"使用OCI生成聊天响应时出错: {str(e)}")