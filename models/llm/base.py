"""
LLM base configuration and interface definition.
Contains:
1. Base configuration class LLMConfig
2. Base interface class BaseLLM
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, field_validator, ConfigDict
from typing import Any, AsyncGenerator
from prometheus_client import Counter, Histogram
from core.config import settings


class LLMConfig(BaseModel):
    """Base configuration for LLM models."""
    
    model_config = ConfigDict(extra='forbid')  # Forbid extra fields
    
    api_key: str
    model_name: str
    
    @field_validator('api_key')
    def validate_api_key(cls, v):
        """Validate API key is not empty."""
        if not v:
            raise ValueError("API key cannot be empty")
        return v

class BaseLLM(ABC):
    """Base class for LLM implementations."""
    
    ERROR_COUNTER = Counter('llm_errors', 'Errors by provider', ['provider'])
    LATENCY_HIST = Histogram('llm_latency', 'Generation latency', ['model_type'])
    
    def __init__(self, config: LLMConfig) -> None:
        """Initialize LLM with configuration.
        
        Args:
            config: LLM configuration
        """
        self.config = config
    
    @abstractmethod
    async def startup(self) -> None:
        """Initialize resources asynchronously."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources asynchronously."""
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]] | str,
        stream: bool = False,
        **kwargs: Any
    ) -> AsyncGenerator[str, None] | str | None:
        """Generate a chat response asynchronously."""
        raise NotImplementedError
    