from .model import *
from core.dictionary import EmbeddingProvider


def create_embedding_model(config: EmbeddingConfig) -> BaseEmbedding:
    """
    工厂函数：根据提供商创建嵌入模型
    
    Args:
        config: 嵌入模型配置对象，包含 provider 字段
        
    Returns:
        BaseEmbedding: 对应提供商的嵌入模型实例
        
    Raises:
        ValueError: 当提供不支持的提供商时抛出
    """
    provider = config.provider.lower()
    
    if provider == EmbeddingProvider.LOCAL_BGE.value:
        if isinstance(config, BGEEmbeddingConfig):
            return BGEEmbedding(config)
        else:
            raise ValueError("本地 BGE 模型配置无效")
        
    elif provider == EmbeddingProvider.LOCAL_QWEN.value:
        if isinstance(config, Qwen3EmbeddingConfig):
            return Qwen3Embedding(config)
        else:
            raise ValueError("本地 Qwen3 模型配置无效")
        
    elif provider in [EmbeddingProvider.API_QWEN.value, EmbeddingProvider.CHATGPT.value]:
        if isinstance(config, OpenAIEmbeddingConfig):
            return OpenAIEmbedding(config)
        else:
            raise ValueError("OpenAI Embedding 参数配置无效")
        
    elif provider == EmbeddingProvider.OCI.value:
        if isinstance(config, OCIEmbeddingConfig):
            return OCIEmbedding(config)
        else:
            raise ValueError("OCI Embedding 参数配置无效")
        
    else:
        raise ValueError(f"不支持的Embedding服务提供商: {provider}")


def get_supported_providers() -> list[str]:
    """
    获取支持的嵌入服务提供商列表
    
    Returns:
        list[str]: 支持的提供商名称列表
    """
    return [provider.value for provider in EmbeddingProvider]