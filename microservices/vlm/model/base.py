"""
VLM (Vision-Language Model) base configuration and interface definition.
Contains:
1. Base configuration class VLMConfig
2. Base interface class BaseVLM
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel
from prometheus_client import Counter, Histogram
from typing import Any, AsyncGenerator
from openai.types.chat import ChatCompletion, ChatCompletionChunk


class VLMConfig(BaseModel):
    """Base configuration for VLM models."""   
    model_name: str
    provider: str
    max_tokens: int = 512

class BaseVLM(ABC):
    """Base class for VLM implementations."""
    
    LATENCY_HIST = Histogram(
        'vlm_latency_seconds', 
        'vlm latency in seconds',
        ['model_type']
    )
    ERROR_COUNTER = Counter(
        'vlm_errors_total', 
        'Total number of vlm errors', 
        ['provider']
    )
    
    @abstractmethod
    async def startup(self) -> None:
        """Initialize resources asynchronously."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources asynchronously."""
        pass

    @abstractmethod
    async def inference(self, messages: list[dict[str, Any]], 
                        stream: bool = False, 
                        **kwargs) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None] | None:
        """
        1. messages: List of dictionaries, each dict contains:
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
            
        2. stream: If True, the result will be streamed.
        3. **kwargs: Extra parameters for inference model
        
        Returns:
           output: the generated text chunk
        """
        pass


    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Health check for a remote or local model"""
        pass
