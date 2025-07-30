from enum import Enum
from typing import Any
from .base import BaseLLM
from .openai_client import OpenaiClient, OpenaiLLMConfig
from .anthropic_client import AnthropicClient, AnthropicLLMConfig
from .huggingface_client import HuggingFaceClient, HuggingFaceLLMConfig

class LLMProvider(str, Enum):
    """Enum of supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    # Add more providers as they are implemented
    # AZURE = "azure"
    # LOCAL = "local"

def create_llm_model(config: OpenaiLLMConfig | AnthropicLLMConfig | HuggingFaceLLMConfig | dict[str, Any]) -> BaseLLM:
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
    elif isinstance(config, AnthropicLLMConfig):
        return AnthropicClient(config)
    elif isinstance(config, HuggingFaceLLMConfig):
        return HuggingFaceClient(config)
    
    # Add more providers as they are implemented
    # elif isinstance(config, AzureLLMConfig):
    #     return AzureClient(config)
    # elif isinstance(config, LocalLLMConfig):
    #     return LocalClient(config)
    
    raise ValueError(f"Unsupported LLM configuration type: {type(config)}")

def get_supported_providers() -> list[str]:
    """Get a list of supported LLM providers.
    
    Returns:
        List of supported provider names
    """
    return [provider.value for provider in LLMProvider]