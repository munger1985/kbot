from enum import Enum
from .base import BaseVLM
from .openai_client import OpenAIVLM, OpenAIVLMConfig
from core.dictionary import VLMProvider


def create_vlm_model(config: OpenAIVLMConfig) -> BaseVLM:
    """根据提供的配置创建 VLM 模型
    
    参数:
        config: VLM 配置
    
    返回:
        BaseVLM 的实例
    
    异常:
        ValueError: 如果不支持该提供商
    """
    if config.provider == VLMProvider.OPENAI.value:
        return OpenAIVLM(config)
    else:
        # 随着实现添加更多提供商
        raise ValueError(f"不支持的 VLM 提供商: {config.provider}")
    

def get_supported_providers() -> list[str]:
    """获取支持的 VLM 提供商列表
    
    返回:
        支持的提供商名称列表
    """
    return [provider.value for provider in VLMProvider]