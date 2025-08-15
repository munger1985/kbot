from .reranker import Reranker
from .base import RerankerConfig

class JinaRerankerV2(Reranker):
    """Implementation of jinaai/jina-reranker-v2-base-multilingual model."""
    
    def __init__(self, config: RerankerConfig):
        super().__init__(
            config=config,
        )