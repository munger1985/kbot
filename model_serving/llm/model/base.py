"""
Base configuration and interface definitions for Large Language Models (LLMs).

This module contains:
1. LLMConfig: Base configuration class for LLM models
2. BaseLLM: Abstract base class defining the core LLM interface
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, AsyncGenerator, TypeVar, Generic
from openai.types.chat import ChatCompletion, ChatCompletionChunk


class LLMConfig(BaseModel):
    """Base configuration class for LLM models.
    
    Contains core configuration parameters required to initialize and interact
    with any Large Language Model implementation.
    """
    model_name: str = Field(..., description="Name of the LLM model (e.g., gpt-4, claude-3-sonnet)")
    provider: str = Field(..., description="Model provider/vendor (e.g., openai, anthropic, local)")
    max_tokens: int = Field(8192, description="Maximum number of tokens allowed for generation")

# Generic type variable bound to LLMConfig for type-safe configuration inheritance
T = TypeVar("T", bound=LLMConfig)

class BaseLLM(ABC, Generic[T]):
    """Abstract base class for all LLM implementations.
    
    Defines the core interface contract for LLM interactions, including
    resource management (startup/shutdown) and chat completion functionality.
    All concrete LLM implementations must inherit from this class and implement
    all abstract methods.
    
    Type Parameters:
        T: A subclass of LLMConfig containing provider-specific configuration
    """
    
    def __init__(self, config: T) -> None:
        """Initialize LLM instance with configuration.
        
        Args:
            config: Configuration object containing model-specific settings
        """
        self.config: T = config
        self.provider = config.provider
        self.model_name = config.model_name
        self.max_tokens = config.max_tokens
    
    @abstractmethod
    async def startup(self) -> None:
        """Asynchronously initialize required resources.
        
        This method should handle any provider-specific initialization such as:
        - Creating API clients
        - Establishing connections
        - Loading local models
        - Setting up authentication
        
        Called once during service initialization before any chat requests.
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Asynchronously release resources.
        
        This method should handle cleanup operations such as:
        - Closing API connections
        - Releasing model memory
        - Cleaning up temporary files
        
        Called once during service shutdown to ensure graceful termination.
        """
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]] | str,
        stream: bool = False,
        **kwargs: Any
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None] | None:
        """Generate chat completion response asynchronously.
        
        Core method for interacting with the LLM to generate responses based on
        input messages. Supports both single-response and streaming modes.
        
        Args:
            messages: Input messages for the chat completion. Can be either:
                - A list of message dictionaries (each with 'role' and 'content')
                - A single string representing the user prompt
            stream: If True, returns an async generator for streaming responses;
                if False, returns a complete ChatCompletion object
            **kwargs: Additional provider-specific generation parameters (e.g.,
                temperature, top_p, stop sequences, etc.)
                
        Returns:
            - ChatCompletion: Complete response object (when stream=False)
            - AsyncGenerator[ChatCompletionChunk]: Streaming response chunks (when stream=True)
            - None: If the request fails or is cancelled
            
        Raises:
            NotImplementedError: If the method is not implemented by the subclass
            LLMError: For provider-specific API errors or generation failures
        """
        raise NotImplementedError