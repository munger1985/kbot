from openai import AsyncOpenAI
from loguru import logger
from typing import Any, AsyncGenerator
from pydantic import field_validator
from prometheus_client import Histogram, Counter
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from .base import BaseVLM, VLMConfig


class OpenAIVLMConfig(VLMConfig):
    """Cloud API configuration"""
    api_key: str    # Required for cloud
    api_endpoint: str | None = None
    api_version: str = "2023-08-01"
    max_retries: int = 3
    temperature: float = 0.7    # Override base temperature
    timeout: int = 30   # 秒

    @field_validator('temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('temperature must be between 0 and 2')
        return v

class OpenAIVLM(BaseVLM):
    """
    Open AI cloud API implementation using OpenAI SDK
    Supports OpenAI compatible mode with AliCloud DashScope
    """

    # Metrics
    API_LATENCY = Histogram(
        'qwenvl_cloud_latency_seconds', 
        'API call latency',
        ['api_endpoint']
    )
    API_ERRORS = Counter(
        'qwenvl_cloud_errors_total',
        'API error counts',
        ['error_code']
    )

    def __init__(self, config: OpenAIVLMConfig):
        if not isinstance(config, OpenAIVLMConfig):
            raise TypeError("config must be RemoteVLMConfig")

        self.config = config
        self._is_initialized = False
        self._openai_client: AsyncOpenAI | None = None

    async def startup(self) -> None:
        """Initialize OpenAI client"""
        try:
            self._openai_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_endpoint,
                timeout=self.config.timeout if hasattr(self.config, 'timeout') else 30
            )
            self._is_initialized = True
            logger.info("OpenAI AsyncOpenAI client initialized")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            raise

    async def inference(
        self,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None] | None:
        """Execute cloud inference using OpenAI SDK.
        
        Args:
            messages: List of messages in strict OpenAI format:
                - Text: {"type": "text", "text": "..."}
                - Image URL: {"type": "image_url", "image_url": {"url": "..."}}
            stream: Whether to stream the response.
            **kwargs: Additional generation parameters.
            
        Raises:
            ValueError: If messages are not in valid OpenAI format.
        """
        if not self._validate_messages(messages):
            raise ValueError("Invalid OpenAI messages format")

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
            logger.error(f"OpenAI SDK error: {str(e)}")
            raise


    def _validate_messages(self, messages: list[dict[str, Any]]) -> bool:
        """Validate if messages are in OpenAI format for VLM models."""
        required_keys = {"role", "content"}
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if not required_keys.issubset(msg.keys()):
                return False
            if msg["role"] not in {"user", "system", "assistant"}:
                return False
            
            # Validate content structure for VLM models
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
        """Cleanup OpenAI client resources"""
        if self._openai_client:
            await self._openai_client.close()
            self._openai_client = None
        self._is_initialized = False
        logger.debug("OpenAI client shutdown")

    async def health_check(self) -> dict[str, Any]:
        """Check service health status."""
        return {
            "initialized": self._is_initialized,
            "model": self.config.model_name,
            "last_error": None,
            "throughput": "N/A"  # Could track actual metrics
        }