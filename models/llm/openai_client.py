from openai import AsyncOpenAI, APIError
from loguru import logger
from typing import AsyncGenerator
from .base import LLMConfig, BaseLLM
from core.config import settings
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam
)


class OpenaiLLMConfig(LLMConfig):
    """Configuration for OpenAI LLM client."""
    
    temperature: float = settings['llm']['temperature']
    max_tokens: int = settings['llm']['max_tokens']
    top_p: float = settings['llm']['top_p']
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = settings['llm']['timeout']
    api_endpoint: str | None = None


class OpenaiClient(BaseLLM):
    """OpenAI LLM client implementation."""
    
    def __init__(self, config: OpenaiLLMConfig):
        """Initialize OpenAI LLM client.
        
        Args:
            config: OpenAI LLM configuration
        """
        self.config = config
        self.client = None
        self._is_running = False
    
    async def startup(self) -> None:
        """Initialize the OpenAI client."""
        try:
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_endpoint,
                timeout=self.config.timeout
            )
            self._is_running = True
            logger.info("OpenAI AsyncOpenAI client initialized")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            raise
        
    async def shutdown(self) -> None:
        """Shutdown the OpenAI client."""
        if self.client:
            await self.client.close()
        self._is_running = False
        logger.info("OpenAI client shutdown")

    
    async def chat(
        self,
        messages: list[dict[str, str]] | str,
        stream: bool = False,
        **kwargs
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None] | None:
        """Generate chat response with consistent return types.
        
        Args:
            messages: List of messages or single prompt string
            stream: Whether to stream the response
            **kwargs: Additional generation parameters
            
        Returns:
            - If not streaming: ChatCompletion object
            - If streaming: AsyncGenerator yielding ChatCompletionChunk objects
        """
        if not self._is_running:
            await self.startup()
        
        # Convert string to message list if needed
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Convert messages to proper format
        converted_messages = [self._convert_message(msg) for msg in messages]
        
        # Merge config with any overrides
        params = {
            "model": self.config.model_name,
            "messages": converted_messages,
            "temperature": kwargs.get('temperature', self.config.temperature),
            "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
            "top_p": kwargs.get('top_p', self.config.top_p),
            "frequency_penalty": kwargs.get('frequency_penalty', self.config.frequency_penalty),
            "presence_penalty": kwargs.get('presence_penalty', self.config.presence_penalty),
            "stream": stream,
        }
        
        try:
            response = await self.client.chat.completions.create(**params) # type: ignore
            return response
            
        except APIError as e:
            self.ERROR_COUNTER.labels(provider="openai").inc()
            logger.error(f"API Error: {e.code} - {e.message}")
            raise
        except Exception as e:
            self.ERROR_COUNTER.labels(provider="openai").inc()
            logger.error(f"Error generating chat response with OpenAI: {str(e)}")
            raise Exception(f"Error generating chat response with OpenAI: {str(e)}")

    def _convert_message(self, msg: dict):
        """Convert message dictionary to OpenAI message parameter object."""
        if msg['role'] == 'user':
            return ChatCompletionUserMessageParam(role='user', content=msg['content'])
        elif msg['role'] == 'system':
            return ChatCompletionSystemMessageParam(role='system', content=msg['content'])
        elif msg['role'] == 'assistant':
            return ChatCompletionAssistantMessageParam(role='assistant', content=msg['content'])
        else:
            raise ValueError(f"Unknown role: {msg['role']}")