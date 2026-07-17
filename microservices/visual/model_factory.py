"""视觉嵌入模型工厂"""

from .model import *
from core.dictionary import VisualEmbeddingProvider


def create_visual_model(config: VisualModelConfig) -> BaseVisualEmbedding:
    """根据 provider 创建视觉嵌入模型实例"""
    provider = config.provider

    if provider == VisualEmbeddingProvider.LOCAL_QWEN.value:
        if isinstance(config, ColQwen2EmbeddingConfig):
            return ColQwen2Embedding(config)
        else:
            raise ValueError("Invalid configuration for ColQwen2 model")

    raise ValueError(f"不支持的视觉嵌入 Provider: {provider}")
