from .model import *
from core.dictionary import RerankerProvider


def create_reranker_model(config: RerankerConfig) -> BaseReranker:
    """Factory function to create Reranker model instances based on provider configuration.
    
    This factory pattern implementation abstracts the instantiation of different
    reranker provider implementations (local BGE, local Qwen3, etc.), ensuring proper
    configuration validation and type matching for each provider.
    
    Args:
        config: Reranker configuration object containing provider identifier and
            provider-specific settings. Must be a subclass of RerankerConfig matching
            the specified provider.
        
    Returns:
        BaseReranker: Properly initialized reranker instance for the specified provider,
            implementing the BaseReranker interface.
            
    Raises:
        ValueError: If:
            - The provider is not supported
            - The configuration object type does not match the provider requirements
    """
    provider = config.provider.lower()
    
    # 1. Local BGE reranker model
    if provider == RerankerProvider.LOCAL_BGE.value:
        if isinstance(config, BGERerankerConfig):
            return BGEReranker(config)
        else:
            raise ValueError(f"Provider {provider} requires BGERerankerConfig configuration object")
            
    # 2. Local Qwen3 reranker model
    elif provider == RerankerProvider.LOCAL_QWEN.value:
        if isinstance(config, Qwen3RerankerConfig):
            return Qwen3Reranker(config)
        else:
            raise ValueError(f"Provider {provider} requires Qwen3RerankerConfig configuration object")
            
    # 3. Qwen Reranker API (DashScope / 百炼)
    elif provider == RerankerProvider.API_QWEN.value:
        if isinstance(config, QwenRerankerConfig):
            return QwenReranker(config)
        else:
            raise ValueError(f"Provider {provider} requires QwenRerankerConfig configuration object")
            
    # 4. OpenAI-compatible APIs (Future implementation fallback)
    # elif provider == RerankerProvider.CHATGPT.value:
    #     if isinstance(config, OpenAIRerankerConfig):
    #         return OpenAIReranker(config)
    #     else:
    #         raise ValueError(f"Provider {provider} requires OpenAIRerankerConfig configuration object")
            
    # 5. Cohere reranker API - commented out for future implementation
    # elif provider == RerankerProvider.COHERE.value:
    #     if isinstance(config, CohereRerankerConfig):
    #         return CohereReranker(config)
    #     else:
    #         raise ValueError(f"Provider {provider} requires CohereRerankerConfig configuration object")
            
    else:
        raise ValueError(f"Unsupported Reranker provider: {provider} (model name: {config.model_name})")


def get_supported_providers() -> list[str]:
    """Get list of supported Reranker provider identifiers.
    
    Returns a list of all provider values defined in the RerankerProvider enum
    that have corresponding implementations in the factory function.
    
    Returns:
        list[str]: List of supported provider name strings (e.g., "local_bge", "local_qwen", "api_qwen")
    """
    return [provider.value for provider in RerankerProvider]