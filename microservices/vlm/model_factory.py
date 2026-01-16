from .model import *
from core.dictionary import VLMProvider


def create_vlm_model(config: VLMConfig) -> BaseVLM:
    """
    工厂函数：根据配置创建 VLM (视觉语言模型) 实例
    
    Args:
        config: VLM 配置对象，包含 provider 字段
        
    Returns:
        BaseVLM: 对应提供商的 VLM 模型实例
        
    Raises:
        ValueError: 当提供不支持的提供商或配置类型不匹配时抛出
    """
    provider = config.provider.lower()
    
    # 1. OpenAI 兼容协议接口 (Qwen-VL API, GPT-4V 等)
    openai_vlm_providers = [
        VLMProvider.API_QWEN.value, 
        VLMProvider.CHATGPT.value
    ]
    
    if provider in openai_vlm_providers:
        if isinstance(config, OpenAIVLMConfig):
            return OpenAIVLM(config)
        else:
            raise ValueError(f"提供商 {provider} 需要 OpenAIVLMConfig 配置对象")
            
    # 2. 待扩展的本地 VLM 或其他厂商
    # elif provider == VLMProvider.LOCAL_QWEN.value:
    #     if isinstance(config, LocalVLMConfig):
    #         return LocalVLM(config)
            
    else:
        raise ValueError(f"不支持的 VLM 提供商: {provider} (模型名称: {config.model_name})")


def get_supported_providers() -> list[str]:
    """
    获取支持的 VLM 提供商列表
    
    Returns:
        list[str]: 支持的提供商名称列表
    """
    return [provider.value for provider in VLMProvider]