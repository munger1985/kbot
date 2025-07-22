from models.reranker.transformer import TransformerReranker
from models.reranker.base import RerankerConfig

class JinaRerankerV2(TransformerReranker):
    """Implementation of jinaai/jina-reranker-v2-base-multilingual model."""
    
    def __init__(self, config: RerankerConfig):
        super().__init__(
            config=config,
        )