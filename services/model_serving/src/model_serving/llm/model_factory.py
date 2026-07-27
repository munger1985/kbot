from .model import *
from platform_core.dictionary import LLMProvider


def create_llm_model(config: LLMConfig) -> BaseLLM:
    """Factory function to create LLM instances based on provider type.
    
    This factory pattern implementation abstracts the instantiation of different
    LLM provider implementations (OpenAI-compatible, OCI, etc.), ensuring proper
    configuration validation and type matching for each provider.
    
    Args:
        config: LLM configuration object containing provider identifier and
            provider-specific settings. Must be a subclass of LLMConfig matching
            the specified provider.
        
    Returns:
        BaseLLM: Properly initialized LLM instance for the specified provider,
            implementing the BaseLLM interface.
            
    Raises:
        ValueError: If:
            - The provider is not supported
            - The configuration object type does not match the provider requirements
    """
    provider = config.provider.lower()
    
    # 1. OpenAI-compatible providers (DeepSeek, Qwen API, ChatGPT, etc.)
    openai_providers = [
        LLMProvider.API_DEEPSEEK.value, 
        LLMProvider.API_QWEN.value, 
        LLMProvider.CHATGPT.value
    ]
    
    if provider in openai_providers:
        if isinstance(config, OpenaiLLMConfig):
            # Return concrete OpenAI client implementation
            return OpenaiClient(config) 
        else:
            raise ValueError(f"Provider {provider} requires OpenaiLLMConfig configuration object")
            
    # 2. Oracle Cloud Infrastructure (OCI) provider
    elif provider == LLMProvider.OCI.value:
        if isinstance(config, OCILLMConfig):
            return OCIClient(config)
        else:
            raise ValueError(f"Provider {provider} requires OCILLMConfig configuration object")
            
    # 3. Future provider extensions (example template)
    # elif provider == LLMProvider.AZURE.value:
    #     if isinstance(config, AzureLLMConfig):
    #         return AzureLLM(config)
            
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_supported_providers() -> list[str]:
    """Get list of supported LLM provider identifiers.
    
    Returns a list of all provider values defined in the LLMProvider enum
    that have corresponding implementations in the factory function.
    
    Returns:
        list[str]: List of supported provider name strings (e.g., "openai", "oci")
    """
    return [provider.value for provider in LLMProvider]