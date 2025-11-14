import asyncio
from core.dictionary import ModelCategory, VLMProvider
from loguru import logger
from datetime import datetime
from typing import Any
from microservices.common.model_pool import BaseModelPool
from .model import *


class VLMModelPool(BaseModelPool[BaseVLM]):
    """VLM 模型池"""
    
    def _get_model_category(self) -> str:
        return ModelCategory.VLM.value

    async def _shutdown_model_instance(self, model: BaseVLM):
        await model.shutdown()

    async def _perform_model_health_check(self, model_id: int, model: BaseVLM):
        if asyncio.iscoroutinefunction(model.health_check):
            status = await model.health_check()
        else:
            status = await asyncio.to_thread(model.health_check)
        
        if isinstance(status, dict):
            if status.get('initialized', False):
                logger.info(f"模型 {self._model_names.get(model_id, str(model_id))} 健康检查通过")
            else:
                logger.warning(f"模型 {self._model_names.get(model_id, str(model_id))} 健康检查失败")
                raise RuntimeError("健康检查失败")
        else:
            if getattr(status, 'initialized', False):
                logger.info(f"模型 {self._model_names.get(model_id, str(model_id))} 健康检查通过")
            else:
                logger.warning(f"模型 {self._model_names.get(model_id, str(model_id))} 健康检查失败")
                raise RuntimeError("健康检查失败")

    async def _start_model(self, model_id: int, model_data: dict[str, Any]) -> BaseVLM:
        display_name = model_data.get("display_name")
        model_name = model_data.get("model_name")
        provider = model_data.get("provider")
        model_params = model_data.get("model_params", {})
        
        if not all([model_name, provider]):
            raise ValueError(f"模型 {display_name or model_id} 缺少必要参数")

        if provider == VLMProvider.OPENAI.value:
            api_endpoint = model_data.get("api_endpoint")
            api_key = model_data.get("api_key")

            if not api_endpoint or not api_key:
                raise ValueError(f"模型 {display_name or model_name} 没有 API 端点或 API 密钥")
            
            model_config = OpenAIVLMConfig(
                model_name=model_name,
                provider=provider,
                max_tokens=model_params.get("max_tokens", 512),
                api_key=api_key,
                api_endpoint=api_endpoint,
                api_version=model_params.get("api_version", ""),
                timeout=model_params.get("timeout", 30),
                max_retries=model_params.get("max_retries", 3),
                temperature=model_params.get("temperature", 0.1)
            )
        else:
            raise NotImplementedError(f"不支持的模型提供者: {provider}")

        model = create_vlm_model(model_config)
        await model.startup()
        self._models[model_id] = model
        self._model_names[model_id] = display_name or model_name
        self._last_used[model_id] = datetime.now()
        logger.success(f"模型 {display_name or model_name} 加载成功")
        return model