from core.config.settings import get_llm_config
from core.dictionary import ModelCategory, LLMProvider
from loguru import logger
from datetime import datetime
from typing import Any
from microservices.common.model_pool import BaseModelPool
from .model import *


class LLMModelPool(BaseModelPool[BaseLLM]):
    """LLM 模型池"""
    
    def __init__(self, health_check_interval: int = 600):
        super().__init__(health_check_interval)
        self._providers: dict[int, str] = {}
        self._max_tokens: dict[int, int] = {}

    def _get_model_category(self) -> str:
        return ModelCategory.LLM.value

    async def _shutdown_model_instance(self, model: BaseLLM):
        await model.shutdown()

    async def _perform_model_health_check(self, model_id: int, model: BaseLLM):
        await model.chat("hello", False, **{"max_tokens": 16})
        logger.debug(f"模型 {self._model_names.get(model_id, str(model_id))} 健康检查成功")

    async def _start_model(self, model_id: int, model_data: dict[str, Any]) -> BaseLLM:
        display_name = model_data.get("display_name")
        model_name = model_data.get("model_name")
        provider = model_data.get("provider")
        model_params = model_data.get("model_params", {})
        
        if not all([model_name, provider]):
            raise ValueError(f"模型 {display_name or model_id} 缺少必要参数")

        config = get_llm_config()
        
        if provider == LLMProvider.OPENAI.value:
            api_endpoint = model_data.get("api_endpoint")
            api_key = model_data.get("api_key")

            if api_key is None or api_endpoint is None:
                raise ValueError(f"模型 {display_name or model_name} 缺少api_key或api_endpoint")
            
            model_config = OpenaiLLMConfig(
                provider=provider,
                api_key=api_key,
                api_endpoint=api_endpoint,
                model_name=model_name,
                temperature=model_params.get("temperature", config.temperature),
                max_tokens=model_params.get("max_tokens", config.max_tokens),
                top_p=model_params.get("top_p", config.top_p),
                frequency_penalty=model_params.get("frequency_penalty", config.frequency_penalty),
                presence_penalty=model_params.get("presence_penalty", config.presence_penalty),
                timeout=model_params.get("timeout", config.timeout)
            )
        elif provider == LLMProvider.OCI.value:
            compartment_id = model_params.get("compartment_id")
            config_file = model_params.get("config_file")
            api_endpoint = model_data.get("api_endpoint")

            if not all([api_endpoint, compartment_id, config_file]):
                raise ValueError(f"模型 {display_name or model_name} 缺少必要参数")

            model_config = OCILLMConfig(
                provider=provider,
                api_endpoint=api_endpoint,
                model_name=model_name,
                temperature=model_params.get("temperature", config.temperature),
                compartment_id=compartment_id,
                max_tokens=model_params.get("max_tokens", config.max_tokens),
                top_p=model_params.get("top_p", config.top_p),
                top_k=model_params.get("top_k", config.top_k),
                frequency_penalty=model_params.get("frequency_penalty", config.frequency_penalty),
                presence_penalty=model_params.get("presence_penalty", config.presence_penalty),
                config_file=config_file
            )
        else:
            raise ValueError(f"模型 {display_name or model_name} 使用了不支持的提供商 {provider}")
        
        model = create_llm_model(model_config)
        await model.startup()
        self._models[model_id] = model
        self._model_names[model_id] = display_name or model_name
        self._providers[model_id] = provider
        self._last_used[model_id] = datetime.now()
        self._max_tokens[model_id] = model_config.max_tokens
        logger.success(f"模型 {display_name or model_name} 加载成功")
        return model

    def get_provider_in_pool(self, model_id: int) -> str | None:
        """获取模型池中指定模型的提供商"""
        return self._providers.get(model_id, None)