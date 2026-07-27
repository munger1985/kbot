from .model import *
from platform_core.dictionary import EmbeddingProvider


def create_embedding_model(config: EmbeddingConfig) -> BaseEmbedding:
    """
    Factory function: Create embedding model instance based on provider type.
    
    Args:
        config: Embedding model configuration object containing provider field
        
    Returns:
        BaseEmbedding: Embedding model instance corresponding to the specified provider
        
    Raises:
        ValueError: Raised when unsupported provider is specified or configuration type mismatch
    """
    provider = config.provider.lower()
    
    if provider == EmbeddingProvider.LOCAL_BGE.value:
        if isinstance(config, BGEEmbeddingConfig):
            return BGEEmbedding(config)
        else:
            raise ValueError("Invalid configuration for local BGE model")
        
    elif provider == EmbeddingProvider.LOCAL_QWEN.value:
        if isinstance(config, Qwen3EmbeddingConfig):
            return Qwen3Embedding(config)
        else:
            raise ValueError("Invalid configuration for local Qwen3 model")
        
    elif provider in [EmbeddingProvider.API_QWEN.value, EmbeddingProvider.CHATGPT.value]:
        if isinstance(config, OpenAIEmbeddingConfig):
            return OpenAIEmbedding(config)
        else:
            raise ValueError("Invalid OpenAI Embedding parameter configuration")
        
    elif provider == EmbeddingProvider.OCI.value:
        if isinstance(config, OCIEmbeddingConfig):
            return OCIEmbedding(config)
        else:
            raise ValueError("Invalid OCI Embedding parameter configuration")
        
    else:
        raise ValueError(f"Unsupported Embedding service provider: {provider}")


def get_supported_providers() -> list[str]:
    """
    Get list of supported embedding service providers.
    
    Returns:
        list[str]: List of supported provider names (string values)
    """
    return [provider.value for provider in EmbeddingProvider]