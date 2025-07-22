from models.reranker.transformer import TransformerReranker
from models.reranker.base import RerankerConfig

class BGERerankerV2M3(TransformerReranker):
    """Implementation of BAAI/bge-reranker-v2-m3 model."""
    
    def __init__(self, config: RerankerConfig):
        super().__init__(
            config=config,
        )