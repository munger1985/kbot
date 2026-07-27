from .model import *
from platform_core.dictionary import VLMProvider


def create_vlm_model(config: VLMConfig) -> BaseVLM:
    """
    Factory function: Create VLM (Vision-Language Model) instance based on configuration
    
    This factory method instantiates the appropriate VLM implementation based on the
    provider specified in the configuration object, ensuring type compatibility between
    the config and provider implementation.
    
    Args:
        config: VLM configuration object containing the provider field
        
    Returns:
        BaseVLM: VLM model instance for the specified provider
        
    Raises:
        ValueError: Raised when an unsupported provider is specified or the config type
            does not match the provider requirements
    """
    provider = config.provider.lower()
    
    # 1. OpenAI-compatible API interfaces (Qwen-VL API, GPT-4V, etc.)
    openai_vlm_providers = [
        VLMProvider.API_QWEN.value, 
        VLMProvider.CHATGPT.value
    ]
    
    if provider in openai_vlm_providers:
        if isinstance(config, OpenAIVLMConfig):
            return OpenAIVLM(config)
        else:
            raise ValueError(f"Provider {provider} requires an OpenAIVLMConfig configuration object")
            
    # 2. Extensible section for local VLMs or other providers
    # elif provider == VLMProvider.LOCAL_QWEN.value:
    #     if isinstance(config, LocalVLMConfig):
    #         return LocalVLM(config)
            
    else:
        raise ValueError(f"Unsupported VLM provider: {provider} (Model name: {config.model_name})")


def get_supported_providers() -> list[str]:
    """
    Get list of supported VLM providers
    
    Retrieves all valid VLM provider values from the VLMProvider enumeration,
    providing a single source of truth for supported providers.
    
    Returns:
        list[str]: List of supported provider names (string values)
    """
    return [provider.value for provider in VLMProvider]