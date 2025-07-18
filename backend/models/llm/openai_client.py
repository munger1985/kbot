import openai
from loguru import logger
from typing import Dict, List, Optional, Union, Any
from pydantic import Field
from .base import LLMConfig, BaseLLM
from core.config import settings
from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam
)

class OpenaiLLMConfig(LLMConfig):
    """Configuration for OpenAI LLM client."""
    
    model_name: str = "gpt-3.5-turbo"
    temperature: float = settings['llm']['temperature']
    max_tokens: int = settings['llm']['max_tokens']
    top_p: float = settings['llm']['top_p']
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = settings['llm']['timeout']
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)

class OpenaiClient(BaseLLM):
    """OpenAI LLM client implementation."""
    
    def __init__(self, config: OpenaiLLMConfig):
        """Initialize OpenAI LLM client.
        
        Args:
            config: OpenAI LLM configuration
        """
        super().__init__(config)
        self.client = None
    
    async def startup(self) -> None:
        """Initialize the OpenAI client."""
        openai.api_key = self.config.api_key
        # For newer versions of OpenAI Python client
        try:
            self.client = openai.AsyncOpenAI(api_key=self.config.api_key)
            logger.info("OpenAI AsyncOpenAI client initialized")
        except (ImportError, AttributeError):
            # Fall back to older client version
            self.client = openai
            logger.info("OpenAI client initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the OpenAI client."""
        # Nothing specific to do for OpenAI client shutdown
        logger.info("OpenAI client shutdown")
        pass
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text based on the prompt.
        
        Args:
            prompt: Input prompt
            model: Model name to use (overrides config)
            max_tokens: Maximum number of tokens to generate (overrides config)
            temperature: Sampling temperature (overrides config)
            **kwargs: Additional parameters to override configuration
        
        Returns:
            Generated text
        """
        if not self.client:
            await self.startup()
        
        # Used for health check
        if prompt == "":
            return ""
        
        # Merge config with any overrides
        params = {
            "model": self.config.model_name,
            "temperature": temperature or self.config.temperature, # type: ignore
            "max_tokens": max_tokens or self.config.max_tokens, # type: ignore
            "top_p": self.config.top_p, # type: ignore
            "frequency_penalty": self.config.frequency_penalty, # type: ignore
            "presence_penalty": self.config.presence_penalty, # type: ignore
            "timeout": self.config.timeout, # type: ignore
            **self.config.additional_kwargs, # type: ignore
            **kwargs
        }
        
        try:
            # Handle both new and old OpenAI client versions
            if hasattr(self.client, "completions"):
                # New client version
                response = await self.client.completions.create( # type: ignore
                    prompt=prompt,
                    **params
                )
                return response.choices[0].text.strip()
            else:
                # Old client version
                response = await self.client.Completion.acreate( # type: ignore
                    prompt=prompt,
                    **params
                )
                return response.choices[0].text.strip()
        except Exception as e:
            self.ERROR_COUNTER.labels(provider="openai").inc()
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            raise Exception(f"Error generating text with OpenAI: {str(e)}")
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate a chat response based on the conversation history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model name to use (overrides config)
            max_tokens: Maximum number of tokens to generate (overrides config)
            temperature: Sampling temperature (overrides config)
            **kwargs: Additional parameters to override configuration
        
        Returns:
            Generated response text
        """
        if not self.client:
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
            "temperature": temperature or self.config.temperature, # type: ignore
            "max_tokens": max_tokens or self.config.max_tokens, # type: ignore
            "top_p": self.config.top_p, # type: ignore
            "frequency_penalty": self.config.frequency_penalty, # type: ignore
            "presence_penalty": self.config.presence_penalty, # type: ignore
            "timeout": self.config.timeout, # type: ignore
            **self.config.additional_kwargs, # type: ignore
            **kwargs
        }
        
        try:
            # Handle both new and old OpenAI client versions
            if hasattr(self.client, "chat"):
                # New client version
                response = await self.client.chat.completions.create( # type: ignore
                    messages=converted_messages,
                    **params
                )
                return response.choices[0].message.content.strip()
            else:
                # Old client version
                response = await self.client.ChatCompletion.acreate( # type: ignore
                    messages=[msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in converted_messages], # type: ignore
                    **params
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            self.ERROR_COUNTER.labels(provider="openai").inc()
            logger.error(f"Error generating chat response with OpenAI: {str(e)}")
            raise Exception(f"Error generating chat response with OpenAI: {str(e)}")
        
    def _convert_message(self, msg: dict):
            if msg['role'] == 'user':
                return ChatCompletionUserMessageParam(role='user', content=msg['content'])
            elif msg['role'] == 'system':
                return ChatCompletionSystemMessageParam(role='system', content=msg['content'])
            elif msg['role'] == 'assistant':
                return ChatCompletionAssistantMessageParam(role='assistant', content=msg['content'])
            else:
                raise ValueError(f"Unknown role: {msg['role']}")