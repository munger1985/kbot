from enum import Enum
from models.reranker.base import BaseReranker, RerankerConfig
from models.reranker.bgev2m3 import BGERerankerV2M3
from models.reranker.jinav2 import JinaRerankerV2

class RerankModels(str, Enum):
    """Enumeration of supported reranker models."""
    
    BGE_RERANKER_V2_M3 = "bge-reranker-v2-m3"
    JINA_RERANKER_V2 = "jina-reranker-v2"


def create_reranker_model(config: RerankerConfig) -> BaseReranker:
    """
    Create a reranker instance based on the provided configuration.
    
    Args:
        config: Configuration for the reranker
        
    Returns:
        An instance of BaseReranker
    """
    model_name = config.model_name.lower()
    
    if RerankModels.BGE_RERANKER_V2_M3.value in model_name:
        return BGERerankerV2M3(config)
    elif RerankModels.JINA_RERANKER_V2.value in model_name:
        return JinaRerankerV2(config)
    else:
        raise ValueError(f"Unsupported reranker model: {config.model_name}")