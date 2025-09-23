from enum import Enum
from .base import BaseReranker, RerankerConfig
from .local_reranker import LocalReranker, LocalRerankerConfig
from .cohere_reranker import CohereReranker, CohereRerankerConfig
from .jina_reranker import JinaReranker, JinaRerankerConfig
from .qwen3_reranker import Qwen3Reranker, Qwen3RerankerConfig


class RerankerProvider(str, Enum):
    """支持的 reranker 模型枚举"""
    LOCAL = "local"
    COHERE = "cohere"


def create_reranker_model(config: RerankerConfig) -> BaseReranker:
    """
    根据提供的配置创建 reranker 实例
    
    Args:
        config: reranker 配置
        
    Returns:
        BaseReranker 实例
        
    Raises:
        ValueError: 当配置无效或不支持时
    """
    model_name = config.model_name.lower()
    if config.provider == RerankerProvider.LOCAL.value:
        if isinstance(config, LocalRerankerConfig):
            return LocalReranker(config)
        elif isinstance(config, JinaRerankerConfig):
            return JinaReranker(config)
        elif isinstance(config, Qwen3RerankerConfig):
            return Qwen3Reranker(config)
        else:
            raise ValueError("本地 reranker 配置无效")
    elif config.provider == RerankerProvider.COHERE.value:
        if isinstance(config, CohereRerankerConfig):
            return CohereReranker(config)
        else:
            raise ValueError("Cohere reranker 配置无效")
    else:
        raise ValueError(f"不支持的 reranker 模型: {config.model_name}")