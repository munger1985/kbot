"""
Basic configuration and interface definitions for VLM (Vision-Language Model).
Contains:
1. Base configuration class VLMConfig
2. Base interface class BaseVLM
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, AsyncGenerator, TypeVar, Generic
from openai.types.chat import ChatCompletion, ChatCompletionChunk


class VLMConfig(BaseModel):
    """Base configuration for VLM models
    
    Attributes:
        model_name: Name of the model
        provider: Provider of the model
        max_tokens: Maximum number of tokens
    """   
    model_name: str = Field(..., description="Model name")
    provider: str = Field(..., description="Model provider")
    max_tokens: int = Field(8192, description="Maximum number of tokens")

T = TypeVar("T", bound=VLMConfig)

class BaseVLM(ABC, Generic[T]):
    """Base class for VLM implementations
    
    This abstract base class defines the core interface that all VLM implementations
    must follow, including resource management, inference, and health check capabilities.
    """
    def __init__(self, config: T) -> None:
        self.config: T = config
        self.provider = config.provider
        self.model_name = config.model_name
        self.max_tokens = config.max_tokens
    
    
    @abstractmethod
    async def startup(self) -> None:
        """Asynchronously initialize resources
        
        This method should handle all necessary setup operations such as:
        - Establishing connections to model services
        - Loading model weights (for local models)
        - Initializing authentication
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Asynchronously release resources
        
        This method should clean up all resources such as:
        - Closing connections to model services
        - Releasing GPU/CPU memory
        - Terminating background processes
        """
        pass

    @abstractmethod
    async def inference(self, messages: list[dict[str, Any]], 
                        stream: bool = False, 
                        **kwargs) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None] | None:
        """Execute inference task
        
        Performs vision-language inference based on the provided messages containing
        both text and image inputs.
        
        Args:
            messages: List of message dictionaries, each containing:
                {
                    "role": str,   # One of "user", "system", or "assistant"
                    "content": [
                        {
                            "type": "text",
                            "text": str
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": str
                            }
                        }
                    ]
                }
            stream: If True, results will be returned as a streaming response
            **kwargs: Additional parameters for the inference model (e.g., temperature, top_p)
        
        Returns:
            Generated text completion or streaming output chunks
            Returns None if inference fails or no valid response is generated
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Perform health check on remote or local models
        
        Verifies the operational status of the model service/instance.
        
        Returns:
            Dictionary containing health status information, typically including:
            - status: "healthy" or "unhealthy"
            - latency: Response time in milliseconds (if applicable)
            - error: Error message (if any)
            - model_version: Version of the model (if available)
        """
        pass