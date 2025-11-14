from .base import BaseLLM, LLMConfig
from .openai_client import OpenaiClient
from .oci_client import OCIClient
from core.dictionary import LLMProvider


def create_llm_model(config: LLMConfig) -> BaseLLM:
    """根据提供的配置创建LLM模型
    
    Args:
        config: LLM配置对象
    
    Returns:
        BaseLLM实例
    
    Raises:
        ValueError: 如果提供商不被支持
    """
    if config.provider == LLMProvider.OPENAI.value:
        return OpenaiClient(config)  # type: ignore
    elif config.provider == LLMProvider.OCI.value:
        return OCIClient(config)  # type: ignore
    # TODO: 添加更多提供商
    # 随着实现逐步添加更多提供商
    # elif isinstance(config, AzureLLMConfig):
    #     return AzureClient(config)
    # elif isinstance(config, LocalLLMConfig):
    #     return LocalClient(config)
    else:
        raise ValueError(f"不支持的LLM提供商: {config.provider}")


def get_supported_providers() -> list[str]:
    """获取支持的LLM提供商列表
    
    Returns:
        支持的提供商名称列表
    """
    return [provider.value for provider in LLMProvider]