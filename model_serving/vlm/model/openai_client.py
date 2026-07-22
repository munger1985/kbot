from typing import Any, AsyncGenerator
from openai import AsyncOpenAI, APIError
from loguru import logger
from pydantic import Field
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .base import BaseVLM, VLMConfig


class OpenAIVLMConfig(VLMConfig):
    """OpenAI VLM client configuration
    
    Attributes:
        api_key: API key for authentication
        api_endpoint: Base URL for the API (optional, uses OpenAI default if None)
        api_version: API version to use
        max_retries: Maximum number of retries for failed requests
        temperature: Sampling temperature (0.0-2.0)
        timeout: Request timeout in seconds
    """
    api_key: str = Field(..., description="API key")
    api_endpoint: str | None = Field(None, description="API base URL")
    api_version: str = "2023-08-01"
    max_retries: int = 3
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    timeout: int = Field(30, ge=1, description="Request timeout in seconds")


class OpenAIVLM(BaseVLM[OpenAIVLMConfig]):
    """
    Multimodal vision-language model client implemented with OpenAI SDK
    
    Compatible with OpenAI GPT-4o, Alibaba DashScope Qwen-VL, and other
    OpenAI-compatible VLM APIs.
    """

    def __init__(self, config: OpenAIVLMConfig):
        super().__init__(config)
        self._openai_client: AsyncOpenAI | None = None
        self._is_initialized: bool = False

    async def startup(self) -> None:
        """Initialize OpenAI async client
        
        Establishes the connection to the OpenAI-compatible API and sets up
        the client with configured authentication and timeout settings.
        """
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
            logger.info(f"✅ OpenAIVLM client ready: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {str(e)}")
            raise

    async def inference(
        self,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None]:
        """
        Execute multimodal inference
        
        Performs vision-language inference using the configured OpenAI-compatible
        model, supporting both text and image inputs.
        
        Args:
            messages: List of message dictionaries following OpenAI VLM format
            stream: If True, returns streaming response chunks
            **kwargs: Additional inference parameters (overrides config values)
        
        Returns:
            ChatCompletion object for non-streaming requests, or async generator
            of ChatCompletionChunk for streaming requests
        
        Raises:
            ValueError: If message format is invalid
            APIError: If OpenAI API returns an error
            Exception: For other unexpected errors during inference
        """
        if not self._is_initialized:
            await self.startup()

        if not self._validate_messages(messages):
            raise ValueError("Invalid OpenAI VLM message format - ensure content is a list containing type: text/image_url items")

        # 1. Dynamic parameter aggregation
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p"),
            "presence_penalty": kwargs.get("presence_penalty"),
            "frequency_penalty": kwargs.get("frequency_penalty"),
            "response_format": kwargs.get("response_format"),  # Supports JSON Mode
        }

        # 2. Clean up None values to prevent SDK errors
        api_params = {k: v for k, v in payload.items() if v is not None}

        # Remove penalty parameters at neutral (0.0) values — some models
        # reject them entirely, and 0.0 matches API default behavior.
        for _key in ('frequency_penalty', 'presence_penalty'):
            if _key in api_params and api_params[_key] == 0.0:
                del api_params[_key]

        try:
            logger.debug(f"🚀 VLM inference started: {self.model_name} (Stream={stream})")
            response = await self._openai_client.chat.completions.create(**api_params)  # type: ignore
            return response
        except APIError as e:
            logger.error(f"❌ OpenAI API error: {e.code} - {e.message}")
            raise
        except Exception as e:
            logger.error(f"💥 Unexpected error during VLM inference: {str(e)}")
            raise

    def _validate_messages(self, messages: list[dict[str, Any]]) -> bool:
        """Validate message format compliance with VLM vision model specifications
        
        Checks that messages follow the OpenAI multimodal format with proper
        structure for text and image inputs.
        
        Args:
            messages: List of message dictionaries to validate
        
        Returns:
            True if messages are valid, False otherwise
        """
        try:
            for msg in messages:
                # Basic structure validation
                if not isinstance(msg, dict) or not {"role", "content"}.issubset(msg.keys()):
                    return False
                if msg["role"] not in {"user", "system", "assistant"}:
                    return False
                
                # VLM content must be in list format
                content = msg["content"]
                if not isinstance(content, list):
                    return False
                
                for item in content:
                    if not isinstance(item, dict) or "type" not in item:
                        return False
                    
                    item_type = item["type"]
                    if item_type == "text":
                        if "text" not in item:
                            return False
                    elif item_type == "image_url":
                        if "image_url" not in item or "url" not in item["image_url"]:
                            return False
                    else:
                        # Unsupported content type
                        return False
            return True
        except Exception:
            return False

    async def shutdown(self) -> None:
        """Release client resources
        
        Closes the OpenAI client connection and resets initialization state.
        """
        if self._openai_client:
            await self._openai_client.close()
            self._openai_client = None
        self._is_initialized = False
        logger.info("♻️ OpenAI VLM client resources recycled")

    async def health_check(self) -> dict[str, Any]:
        """Enhanced health check
        
        Returns current status of the VLM client including initialization state.
        
        Returns:
            Dictionary containing health status information
        """
        return {
            "status": "healthy" if self._is_initialized else "uninitialized",
            "model": self.model_name,
            "provider": "OpenAI-Compatible VLM",
            "initialized": self._is_initialized
        }