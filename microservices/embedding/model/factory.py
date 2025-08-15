from enum import Enum
from .base import BaseEmbedding, EmbeddingConfig
from .local_client import LocalEmbedding
from .openai_client import OpenAIEmbedding
from .azure_client import AzureEmbedding
from .cohere_client import CohereEmbedding
from .oci_client import OCIEmbedding


class EmbeddingProvider(str, Enum):
    """Enumeration of supported embedding providers."""
    LOCAL = "local"
    OPENAI = "openai"
    AZURE = "azure"
    COHERE = "cohere"
    OCI = "oci"


def create_embedding_model(config: EmbeddingConfig) -> BaseEmbedding:
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
    elif provider == EmbeddingProvider.OCI:
        return OCIEmbedding(config) # type: ignore
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def get_supported_providers() -> list[str]:
    """Get a list of supported embedding providers"""
    return [provider.value for provider in EmbeddingProvider]