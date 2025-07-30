from enum import Enum
from models.embedding.base import BaseEmbedding, LocalEmbeddingConfig, RemoteEmbeddingConfig
from models.embedding.local_client import LocalEmbedding
from models.embedding.openai_client import OpenAIEmbedding
from models.embedding.azure_client import AzureEmbedding
from models.embedding.cohere_client import CohereEmbedding


class EmbeddingProvider(str, Enum):
    """Enumeration of supported embedding providers."""
    LOCAL = "local"
    OPENAI = "openai"
    AZURE = "azure"
    COHERE = "cohere"


def create_embedding_model(config: LocalEmbeddingConfig | RemoteEmbeddingConfig) -> BaseEmbedding:
    """Factory function to create embedding model based on provider"""
    provider = config.provider.lower()
    
    if provider == EmbeddingProvider.LOCAL:
        return LocalEmbedding(config) # type: ignore
    elif provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbedding(config) # type: ignore
    elif provider == EmbeddingProvider.AZURE:
        return AzureEmbedding(config) # type: ignore
    elif provider == EmbeddingProvider.COHERE:
        return CohereEmbedding(config) # type: ignore
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def get_supported_providers() -> list[str]:
    """Get a list of supported embedding providers"""
    return [provider.value for provider in EmbeddingProvider]