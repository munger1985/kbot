from enum import Enum
from .base import BaseLLM, LLMConfig
from .openai_client import OpenaiClient
from .oci_client import OCIClient

class LLMProvider(str, Enum):
    """Enum of supported LLM providers."""
    OPENAI = "openai"
    OCI = "oci"
    # Add more providers as they are implemented
    # AZURE = "azure"
    # HUGGINGFACE = "huggingface"
    # LOCAL = "local"

def create_llm_model(config: LLMConfig) -> BaseLLM:
    """Create an LLM model based on the provided configuration.
    
    Args:
        config: LLM configuration
    
    Returns:
        An instance of BaseLLM
    
    Raises:
        ValueError: If the provider is not supported
    """
    if config.provider == LLMProvider.OPENAI.value:
        return OpenaiClient(config) # type: ignore
    elif config.provider == LLMProvider.OCI.value:
        return OCIClient(config) # type: ignore
    # TODO: add more providers
    # Add more providers as they are implemented
    # elif isinstance(config, AzureLLMConfig):
    #     return AzureClient(config)
    # elif isinstance(config, LocalLLMConfig):
    #     return LocalClient(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")


def get_supported_providers() -> list[str]:
    """Get a list of supported LLM providers.
    
    Returns:
        List of supported provider names
    """
    return [provider.value for provider in LLMProvider]