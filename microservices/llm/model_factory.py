from .model import *
from core.dictionary import LLMProvider


def create_llm_model(config: LLMConfig) -> BaseLLM:
    """
    工厂函数：根据提供商创建大语言模型 (LLM) 实例
    
    Args:
        config: LLM 配置对象，包含 provider 字段
        
    Returns:
        BaseLLM: 对应提供商的 LLM 实例
        
    Raises:
        ValueError: 当提供不支持的提供商或配置类型不匹配时抛出
    """
    provider = config.provider.lower()
    
    # 1. OpenAI 兼容接口 (DeepSeek, Qwen API, ChatGPT 等)
    openai_providers = [
        LLMProvider.API_DEEPSEEK.value, 
        LLMProvider.API_QWEN.value, 
        LLMProvider.CHATGPT.value
    ]
    
    if provider in openai_providers:
        if isinstance(config, OpenaiLLMConfig):
            # 确保这里返回的是实现类，例如 OpenaiLLM 或 OpenaiClient
            return OpenaiClient(config) 
        else:
            raise ValueError(f"提供商 {provider} 需要 OpenaiLLMConfig 配置对象")
            
    # 2. Oracle Cloud Infrastructure (OCI)
    elif provider == LLMProvider.OCI.value:
        if isinstance(config, OCILLMConfig):
            return OCIClient(config)
        else:
            raise ValueError(f"提供商 {provider} 需要 OCILLMConfig 配置对象")
            
    # 3. 待扩展的提供商
    # elif provider == LLMProvider.AZURE.value:
    #     if isinstance(config, AzureLLMConfig):
    #         return AzureLLM(config)
            
    else:
        raise ValueError(f"不支持的LLM提供商: {provider}")


def get_supported_providers() -> list[str]:
    """
    获取支持的 LLM 提供商列表
    
    Returns:
        list[str]: 支持的提供商名称列表
    """
    return [provider.value for provider in LLMProvider]