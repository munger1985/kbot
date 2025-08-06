from enum import Enum
from .base import BaseVLM
from .openai_client import OpenAIVLM, OpenAIVLMConfig


class VLMProvider(str, Enum):
    """Enum of supported VLM providers."""
    OPENAI = "openai"     # Currently only supports OpenAI-compatible cloud providers
    # Add more providers as they are implemented

def create_vlm_model(config: OpenAIVLMConfig) -> BaseVLM:
    """Create a VLM model based on the provided configuration.
    
    Args:
        config: VLM configuration
    
    Returns:
        An instance of BaseVLM
    
    Raises:
        ValueError: If the provider is not supported
    """
    if config.provider == VLMProvider.OPENAI.value:
        return OpenAIVLM(config)
    else:
        # Add more providers as they are implemented
        raise ValueError(f"Unsupported VLM provider: {config.provider}")
    

def get_supported_providers() -> list[str]:
    """Get a list of supported VLM providers.
    
    Returns:
        List of supported provider names
    """
    return [provider.value for provider in VLMProvider]