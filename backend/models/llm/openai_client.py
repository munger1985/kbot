import openai
from functools import lru_cache
from typing import Dict, List, Optional, Union, Any
from .base import LLMConfig, BaseLLM
from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam
)
class OpenaiLLMConfig(LLMConfig):
    """Configuration for OpenAI LLM client."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        timeout: int = 60,
        **kwargs
    ):
        """Initialize OpenAI LLM configuration.
        
        Args:
            api_key: OpenAI API key
            model_name: Model name to use (default: gpt-3.5-turbo)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum number of tokens to generate (default: 1000)
            top_p: Nucleus sampling parameter (default: 1.0)
            frequency_penalty: Frequency penalty parameter (default: 0.0)
            presence_penalty: Presence penalty parameter (default: 0.0)
            timeout: Request timeout in seconds (default: 60)
            **kwargs: Additional parameters to pass to the OpenAI API
        """
        super().__init__(provider="openai", model_name=model_name, api_key=api_key)
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.timeout = timeout
        self.additional_kwargs = kwargs

class OpenaiClient(BaseLLM):
    """OpenAI LLM client implementation."""
    
    def __init__(self, config: OpenaiLLMConfig):
        """Initialize OpenAI LLM client.
        
        Args:
            config: OpenAI LLM configuration
        """
        super().__init__()
        self.config = config
        self.client = None
    
    async def startup(self) -> None:
        """Initialize the OpenAI client."""
        openai.api_key = self.config.api_key
        # For newer versions of OpenAI Python client
        try:
            self.client = openai.AsyncOpenAI(api_key=self.config.api_key)
        except (ImportError, AttributeError):
            # Fall back to older client version
            self.client = openai
    
    async def shutdown(self) -> None:
        """Shutdown the OpenAI client."""
        # Nothing specific to do for OpenAI client shutdown
        pass
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on the prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters to override configuration
        
        Returns:
            Generated text
        """
        if not self.client:
            await self.startup()
        
        # Merge config with any overrides
        params = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "timeout": self.config.timeout,
            **self.config.additional_kwargs,
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
            raise Exception(f"Error generating text with OpenAI: {str(e)}")
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat response based on the conversation history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
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
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "timeout": self.config.timeout,
            **self.config.additional_kwargs,
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
                    messages=messages,
                    **params
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
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