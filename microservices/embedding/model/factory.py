from enum import Enum
from .base import BaseEmbedding, EmbeddingConfig
from .local_client import LocalEmbedding
from .openai_client import OpenAIEmbedding
from .azure_client import AzureEmbedding
from .cohere_client import CohereEmbedding
from .oci_client import OCIEmbedding


class EmbeddingProvider(str, Enum):
    """支持的嵌入服务提供商枚举"""
    LOCAL = "local"
    OPENAI = "openai"
    AZURE = "azure"
    COHERE = "cohere"
    OCI = "oci"


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
        raise ValueError(f"不支持的嵌入服务提供商: {provider}")


def get_supported_providers() -> list[str]:
    """
    获取支持的嵌入服务提供商列表
    
    Returns:
        list[str]: 支持的提供商名称列表
    """
    return [provider.value for provider in EmbeddingProvider]