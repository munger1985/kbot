from .reranker import Reranker
from .base import RerankerConfig

class BGERerankerV2M3(Reranker):
    """Implementation of BAAI/bge-reranker-v2-m3 model."""
    
    def __init__(self, config: RerankerConfig):
        super().__init__(
            config=config,
        )