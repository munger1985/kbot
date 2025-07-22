from openai import AsyncOpenAI, APIError
from loguru import logger
from typing import Dict, List, Optional, AsyncGenerator, Any
from .base import LLMConfig, BaseLLM
from core.config import settings
from openai.types.chat import (
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
    api_endpoint: Optional[str] = None
    # organization: Optional[str] = None

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
                # organization=self.config.organization,
                timeout=self.config.timeout
            )
            self._is_running = True
            logger.info("OpenAI AsyncOpenAI client initialized")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
        
    
    async def shutdown(self) -> None:
        """Shutdown the OpenAI client."""
        if self.client:
            await self.client.close()
        self._is_running = False
        logger.info("OpenAI client shutdown")
    
    async def generate(
        self,
        prompt: str,
        n: int = 1,
        **kwargs
    ) -> Optional[List[str]]:
        """Generate text based on the prompt.
        
        Args:
            prompt: Input prompt
            n: Number of responses to generate
            **kwargs: Additional parameters to override configuration
        
        Returns:
            Generated text
        """
        if not self._is_running:
            await self.startup()
        
        # Used for health check
        if prompt == "":
            return None
        
        # Merge config with any overrides
        params = {
            "model": self.config.model_name,
            "prompt": prompt,
            "temperature": kwargs.get('temperature', self.config.temperature),
            "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
            "n": n,
            "top_p": kwargs.get('top_p', self.config.top_p),
            "frequency_penalty": kwargs.get('frequency_penalty', self.config.frequency_penalty),
            "presence_penalty": kwargs.get('presence_penalty', self.config.presence_penalty)
        }
        
        try:
            response = await self.client.completions.create(**params) # type: ignore
            return [choice.text.strip() for choice in response.choices]
        except Exception as e:
            self.ERROR_COUNTER.labels(provider="openai").inc()
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            raise Exception(f"Error generating text with OpenAI: {str(e)}")
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Optional[AsyncGenerator[str, None] | str]:
        """Generate a chat response based on the conversation history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            stream: Whether to stream the response
            **kwargs: Additional parameters to override configuration
        
        Returns:
            If stream is True, returns an async generator of strings, else returns a single string.
            流式模式下返回异步生成器，非流式返回字符串
        """
        if not self._is_running:
            await self.startup()
        
        # Validate messages format
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must be a dictionary with 'role' and 'content' keys")
        
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
            
            if stream:
                return self._handle_stream(response)
            return response.choices[0].message.content
            
        except APIError as e:
            self.ERROR_COUNTER.labels(provider="openai").inc()
            logger.error(f"API Error: {e.code} - {e.message}")
            raise e
        except Exception as e:
            self.ERROR_COUNTER.labels(provider="openai").inc()
            logger.error(f"Error generating chat response with OpenAI: {str(e)}")
            raise Exception(f"Error generating chat response with OpenAI: {str(e)}")


    async def _handle_stream(
        self, 
        response_stream: Any
    ) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        full_response = ""
        async for chunk in response_stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content
        yield f"\n[Full response: {len(full_response)} characters]"

    def _convert_message(self, msg: dict):
            if msg['role'] == 'user':
                return ChatCompletionUserMessageParam(role='user', content=msg['content'])
            elif msg['role'] == 'system':
                return ChatCompletionSystemMessageParam(role='system', content=msg['content'])
            elif msg['role'] == 'assistant':
                return ChatCompletionAssistantMessageParam(role='assistant', content=msg['content'])
            else:
                raise ValueError(f"Unknown role: {msg['role']}")