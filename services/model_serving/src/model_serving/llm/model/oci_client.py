import oci
import asyncio
from dataclasses import dataclass
from typing import Any
from pydantic import Field
from loguru import logger

from model_serving.common.oci_auth import validated_oci_config

from .base import LLMConfig, BaseLLM


@dataclass(frozen=True, slots=True)
class OCIChatProfile:
    """描述 OCI 模型家族在原生 Chat API 中的请求差异。"""

    request_format: str
    output_token_field: str
    supported_sampling_params: frozenset[str]
    output_token_cap: int | None = None


_GENERIC_SAMPLING_PARAMS = frozenset({"temperature", "top_p", "top_k"})
_COHERE_PROFILE = OCIChatProfile(
    request_format="cohere",
    output_token_field="max_tokens",
    supported_sampling_params=_GENERIC_SAMPLING_PARAMS,
    output_token_cap=4000,
)
_GENERIC_PROFILE = OCIChatProfile(
    request_format="generic",
    output_token_field="max_tokens",
    supported_sampling_params=_GENERIC_SAMPLING_PARAMS,
)
_LLAMA_PROFILE = OCIChatProfile(
    request_format="generic",
    output_token_field="max_tokens",
    supported_sampling_params=_GENERIC_SAMPLING_PARAMS,
    output_token_cap=4096,
)
_GROK_PROFILE = OCIChatProfile(
    request_format="generic",
    output_token_field="max_tokens",
    supported_sampling_params=_GENERIC_SAMPLING_PARAMS,
    output_token_cap=20000,
)
_OPENAI_GPT5_PROFILE = OCIChatProfile(
    request_format="generic",
    output_token_field="max_completion_tokens",
    # OCI 上的 GPT-5 推理模型不应继承通用采样默认值。
    supported_sampling_params=frozenset(),
)


def resolve_oci_chat_profile(model_name: str) -> OCIChatProfile:
    """按 OCI Provider 模型标识解析唯一的请求能力配置。"""

    normalized = model_name.strip().lower()
    if "cohere" in normalized:
        return _COHERE_PROFILE
    if normalized.startswith("openai.gpt-5"):
        return _OPENAI_GPT5_PROFILE
    if "llama" in normalized:
        return _LLAMA_PROFILE
    if "grok" in normalized:
        return _GROK_PROFILE
    return _GENERIC_PROFILE


def extract_oci_text_content(content_items: Any) -> str:
    """从 OCI SDK 对象或流式字典的内容块中提取全部正文。"""

    if not isinstance(content_items, list):
        return ""
    text_parts: list[str] = []
    for item in content_items:
        if isinstance(item, dict):
            content_type = item.get("type")
            text = item.get("text")
        else:
            content_type = getattr(item, "type", None)
            text = getattr(item, "text", None)
        if content_type and str(content_type).upper() != "TEXT":
            continue
        if isinstance(text, str):
            text_parts.append(text)
    return "".join(text_parts)

class OCILLMConfig(LLMConfig):
    """Configuration class for OCI Generative AI LLM client.
    
    Extends the base LLM configuration with OCI-specific parameters and 
    model generation settings compatible with OCI's Generative AI service.
    """
    temperature: float | None = Field(
        None, 
        ge=0, 
        le=2, 
        description="Temperature parameter controlling output randomness (0-2)"
    )
    top_p: float | None = Field(
        None, 
        ge=0, 
        le=1, 
        description="Top-p sampling parameter for nucleus sampling (0-1)"
    )
    top_k: int | None = Field(
        None, 
        ge=0, 
        description="Top-k sampling parameter for token selection (0+)"
    )
    frequency_penalty: float | None = Field(
        None, 
        ge=-2, 
        le=2, 
        description="Frequency penalty to reduce repetitive text (-2 to 2)"
    )
    presence_penalty: float | None = Field(
        None, 
        ge=-2, 
        le=2, 
        description="Presence penalty to encourage new topics (-2 to 2)"
    )
    api_endpoint: str = Field(
        ..., 
        description="OCI Generative AI service endpoint URL"
    )
    compartment_id: str = Field(
        ..., 
        description="OCI Compartment OCID for resource authorization"
    )
    config_file: dict | str = Field(
        ..., 
        description="OCI authentication configuration (dict or JSON string)"
    )

class OCIClient(BaseLLM[OCILLMConfig]):
    """
    OCI Generative AI optimized LLM implementation.
    
    Provides asynchronous interface to OCI's Generative AI service with 
    model-specific adaptations for Cohere, Llama 3, Grok, and other models 
    available on OCI. Handles authentication, request formatting, and 
    response processing while maintaining compatibility with the BaseLLM interface.
    """
    
    def __init__(self, config: OCILLMConfig):
        """Initialize OCI LLM client with configuration.
        
        Args:
            config: OCI-specific LLM configuration object
        """
        super().__init__(config)
        self.client: oci.generative_ai_inference.GenerativeAiInferenceClient | None = None
        self._is_initialized = False
        self.config = config

    async def startup(self) -> None:
        """Asynchronously initialize OCI Generative AI client.
        
        Handles configuration parsing and client initialization with retry 
        strategy and timeout settings. Idempotent - safe to call multiple times.
        
        Raises:
            Exception: If client initialization fails (authentication, endpoint, etc.)
        """
        if self._is_initialized:
            return

        try:
            oci_config = validated_oci_config(self.config.config_file)
            
            # Initialize inference client with retry and timeout settings
            # Note: OCI SDK is synchronous but connection setup is typically fast
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=oci_config,
                service_endpoint=self.config.api_endpoint,
                retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY,
                timeout=(10, 240)  # Connect timeout: 10s, read timeout: 240s
            )
            self._is_initialized = True
            logger.info(f"✅ OCI LLM client initialized successfully (Model: {self.config.model_name})")
        except Exception as e:
            logger.error(
                "OCI LLM 客户端初始化失败：model={} error_type={} error={}",
                self.config.model_name,
                type(e).__name__,
                str(e),
            )
            raise

    def _convert_to_oci_messages(self, messages: list[dict[str, str]] | str) -> list[Any]:
        """Convert standard message format to OCI Message objects.
        
        Transforms either a list of message dictionaries (role/content pairs)
        or a single string prompt into OCI's required message format with
        proper role normalization (USER/ASSISTANT in uppercase).
        
        Args:
            messages: Input messages in standard format (list of dicts or string)
            
        Returns:
            List of OCI Message objects compatible with the API
        """
        # Convert single string to standard user message
        if isinstance(messages, str):
            messages = [{"role": "USER", "content": messages}]
        
        oci_messages = []
        for msg in messages:
            # Create text content object
            content = oci.generative_ai_inference.models.TextContent()
            content.text = msg.get("content", "")
            
            # Create OCI message object with normalized role
            oci_msg = oci.generative_ai_inference.models.Message()
            oci_msg.role = msg.get("role", "USER").upper()  # OCI requires uppercase roles
            oci_msg.content = [content]
            oci_messages.append(oci_msg)
            
        return oci_messages

    def _build_chat_request(self, messages: list[dict[str, str]] | str, **kwargs) -> Any:
        """根据 OCI 模型家族构造原生 Chat 请求。"""

        profile = resolve_oci_chat_profile(self.config.model_name)
        output_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        if profile.output_token_cap is not None:
            output_tokens = min(output_tokens, profile.output_token_cap)

        sampling_params = {
            "temperature": kwargs.get('temperature', self.config.temperature),
            "top_p": kwargs.get('top_p', self.config.top_p),
            "top_k": kwargs.get('top_k', self.config.top_k),
        }

        if profile.request_format == "cohere":
            request = oci.generative_ai_inference.models.CohereChatRequest()
            request.message = messages[-1]['content'] if isinstance(messages, list) else messages
            request.frequency_penalty = kwargs.get('frequency_penalty', self.config.frequency_penalty)
        else:
            request = oci.generative_ai_inference.models.GenericChatRequest()
            request.api_format = oci.generative_ai_inference.models.GenericChatRequest.API_FORMAT_GENERIC
            request.messages = self._convert_to_oci_messages(messages)

        request_params = {profile.output_token_field: output_tokens}
        request_params.update({
            name: value
            for name, value in sampling_params.items()
            if name in profile.supported_sampling_params
        })
        for param_name, param_value in request_params.items():
            if param_value is not None and hasattr(request, param_name):
                setattr(request, param_name, param_value)

        return request

    async def chat(
        self,
        messages: list[dict[str, str]] | str,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """Generate chat completion response asynchronously via OCI API.
        
        Handles request construction, asynchronous execution of the synchronous
        OCI SDK call (via thread executor), and response return. Supports both
        streaming and non-streaming responses.
        
        Args:
            messages: Input messages (list of role/content dicts or single string)
            stream: Whether to enable streaming responses (True/False)
            **kwargs: Additional generation parameters to override config values
            
        Returns:
            OCI chat response object (complete or streaming)
            
        Raises:
            ValueError: If client is not initialized
            Exception: If API call fails (network, authentication, validation)
        """
        # Ensure client is initialized
        if not self._is_initialized:
            await self.startup()

        if self.client is None:
            raise ValueError("OCI Generative AI client not initialized")

        # Build complete chat request
        chat_request = self._build_chat_request(messages, **kwargs)
        chat_request.is_stream = stream

        # Create chat details with serving mode and compartment info
        chat_details = oci.generative_ai_inference.models.ChatDetails()
        chat_details.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=self.config.model_name
        )
        chat_details.chat_request = chat_request
        chat_details.compartment_id = self.config.compartment_id

        try:
            # Execute synchronous OCI SDK call asynchronously to avoid event loop blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.chat(chat_details)  # type: ignore
            )
            return response
        except Exception as e:
            logger.error(f"❌ OCI chat completion failed: {e}")
            raise

    async def shutdown(self) -> None:
        """Clean up OCI client resources.
        
        Resets client instance and initialization flag, ensuring proper
        resource cleanup for graceful shutdown.
        """
        self.client = None
        self._is_initialized = False
        logger.info("♻️ OCI LLM client shutdown completed")
