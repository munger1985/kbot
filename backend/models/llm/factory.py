import asyncio
from enum import Enum
from typing import Dict, List, Union, Any
from .base import BaseLLM
from .openai_client import OpenaiClient, OpenaiLLMConfig

class LLMProvider(str, Enum):
    """Enum of supported LLM providers."""
    OPENAI = "openai"
    # Add more providers as they are implemented
    # AZURE = "azure"
    # LOCAL = "local"

def create_llm_model(config: Union[OpenaiLLMConfig, Dict[str, Any]]) -> BaseLLM:
    """Create an LLM model based on the provided configuration.
    
    Args:
        config: LLM configuration
    
    Returns:
        An instance of BaseLLM
    
    Raises:
        ValueError: If the provider is not supported
    """
    if isinstance(config, OpenaiLLMConfig):
        return OpenaiClient(config)
    
    # Add more providers as they are implemented
    # elif isinstance(config, AzureLLMConfig):
    #     return AzureClient(config)
    # elif isinstance(config, LocalLLMConfig):
    #     return LocalClient(config)
    
    raise ValueError(f"Unsupported LLM configuration type: {type(config)}")

def get_supported_providers() -> List[str]:
    """Get a list of supported LLM providers.
    
    Returns:
        List of supported provider names
    """
    return [provider.value for provider in LLMProvider]