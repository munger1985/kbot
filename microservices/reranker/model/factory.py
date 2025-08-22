from enum import Enum
from .base import BaseReranker, RerankerConfig
from .local_reranker import LocalReranker, LocalRerankerConfig
from .cohere_reranker import CohereReranker, CohereRerankerConfig

class RerankerProvider(str, Enum):
    """Enumeration of supported reranker models."""
    LOCAL = "local"
    COHERE = "cohere"


def create_reranker_model(config: RerankerConfig) -> BaseReranker:
    """
    Create a reranker instance based on the provided configuration.
    
    Args:
        config: Configuration for the reranker
        
    Returns:
        An instance of BaseReranker
    """
    model_name = config.model_name.lower()
    if config.provider == RerankerProvider.LOCAL.value:
        if isinstance(config, LocalRerankerConfig):
            return LocalReranker(config)
        else:
            raise ValueError("Invalid configuration for local reranker")
    elif config.provider == RerankerProvider.COHERE.value:
        if isinstance(config, CohereRerankerConfig):
            return CohereReranker(config)
        else:
            raise ValueError("Invalid configuration for Cohere reranker")
    else:
        raise ValueError(f"Unsupported reranker model: {config.model_name}")