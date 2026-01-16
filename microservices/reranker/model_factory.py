from .model import *
from core.dictionary import RerankerProvider


def create_reranker_model(config: RerankerConfig) -> BaseReranker:
    """
    工厂函数：根据配置创建 Reranker 模型实例
    
    Args:
        config: Reranker 配置对象，包含 provider 字段
        
    Returns:
        BaseReranker: 对应提供商的 Reranker 模型实例
        
    Raises:
        ValueError: 当提供不支持的提供商或配置类型与提供商不匹配时抛出
    """
    provider = config.provider.lower()
    
    # 1. 本地 BGE 模型
    if provider == RerankerProvider.LOCAL_BGE.value:
        if isinstance(config, BGERerankerConfig):
            return BGEReranker(config)
        else:
            raise ValueError(f"提供商 {provider} 需要 BGERerankerConfig 配置对象")
            
    # 2. 本地 Qwen3 模型
    elif provider == RerankerProvider.LOCAL_QWEN.value:
        if isinstance(config, Qwen3RerankerConfig):
            return Qwen3Reranker(config)
        else:
            raise ValueError(f"提供商 {provider} 需要 Qwen3RerankerConfig 配置对象")
            
    # 3. OpenAI 兼容接口 (Qwen API, ChatGPT 等)
    elif provider in [RerankerProvider.API_QWEN.value, RerankerProvider.CHATGPT.value]:
        if isinstance(config, OpenAIRerankerConfig):
            return OpenAIReranker(config)
        else:
            raise ValueError(f"提供商 {provider} 需要 OpenAIRerankerConfig 配置对象")
            
    # 4. Cohere 模型
    elif provider == RerankerProvider.COHERE.value:
        if isinstance(config, CohereRerankerConfig):
            return CohereReranker(config)
        else:
            raise ValueError(f"提供商 {provider} 需要 CohereRerankerConfig 配置对象")
            
    else:
        raise ValueError(f"不支持的 Reranker 提供商: {provider} (模型名称: {config.model_name})")


def get_supported_providers() -> list[str]:
    """
    获取支持的 Reranker 提供商列表
    
    Returns:
        list[str]: 支持的提供商名称列表
    """
    return [provider.value for provider in RerankerProvider]