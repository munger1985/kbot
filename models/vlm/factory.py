from enum import Enum
from .base import BaseVLM, LocalVLMConfig, RemoteVLMConfig
from .deepseek_vl_remote import DeepSeekVLCloud
from .qwen_vl_remote import QwenVLCloud
from .local_client import LocalVL

class VLMProvider(str, Enum):
    """Enum of supported VLM providers."""
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    LOCAL = "local"
    # Add more providers as they are implemented

def create_vlm_model(config: LocalVLMConfig | RemoteVLMConfig) -> BaseVLM:
    """Create a VLM model based on the provided configuration.
    
    Args:
        config: VLM configuration
    
    Returns:
        An instance of BaseVLM
    
    Raises:
        ValueError: If the provider is not supported
    """
    if config.provider == VLMProvider.DEEPSEEK.value:
        return DeepSeekVLCloud(config) # type: ignore
    elif config.provider == VLMProvider.QWEN.value:
        return QwenVLCloud(config) # type: ignore
    elif config.provider == VLMProvider.LOCAL.value:
        return LocalVL(config) # type: ignore 
    else:
        # Add more providers as they are implemented
        raise ValueError(f"Unsupported VLM provider: {config.provider}")
    

def get_supported_providers() -> list[str]:
    """Get a list of supported VLM providers.
    
    Returns:
        List of supported provider names
    """
    return [provider.value for provider in VLMProvider]