from core.config.settings import get_embed_config
from core.dictionary import ModelCategory, EmbeddingProvider
from loguru import logger
from datetime import datetime
from typing import Any
from microservices.common.model_pool import BaseModelPool
from .model import *


class EmbeddingModelPool(BaseModelPool[BaseEmbedding]):
    """Embedding 模型池"""
    
    def _get_model_category(self) -> int:
        return ModelCategory.TXT_EMBEDDING.value

    async def _shutdown_model_instance(self, model: BaseEmbedding):
        await model.shutdown()

    async def _perform_model_health_check(self, model_id: int, model: BaseEmbedding):
        await model.embed(["健康检查"], batch_size=1)
        logger.debug(f"模型 {self._model_names.get(model_id, str(model_id))} 健康检查通过")

    async def _start_model(self, model_id: int, model_data: dict[str, Any]) -> BaseEmbedding:
        display_name = model_data.get("display_name")
        model_name = model_data.get("model_name")
        provider = model_data.get("provider")
        model_params = model_data.get("model_params", {})
        
        if not all([model_name, provider]):
            raise ValueError(f"模型 {display_name or model_id} 缺少必要参数")

        config = get_embed_config()
        
        if provider == EmbeddingProvider.LOCAL.value:
            model_config = LocalEmbeddingConfig(
                model_name=model_name, # type: ignore
                provider=provider,
                max_tokens=model_params.get("max_tokens", config.max_tokens),
                batch_size=model_params.get("batch_size", 2),
                model_path=model_params.get("model_path"),
                device=model_params.get("device"),
                device_map=model_params.get("device_map"),
                max_memory=model_params.get("max_memory"),
                trust_remote_code=model_params.get("trust_remote_code", False),
                use_fp16=model_params.get("use_fp16", False),
                local_files_only=model_params.get("local_files_only", False),
                compile_model=model_params.get("compile_model", True),
                cache_dir=config.cache_dir
            )
        elif provider == EmbeddingProvider.OCI.value:
            compartment_id = model_params.get("compartment_id")
            config_file = model_params.get("config_file")
            api_endpoint = model_data.get("api_endpoint")

            if not all([model_name, api_endpoint, compartment_id, config_file]):
                raise ValueError(f"模型 {display_name or model_name} 缺少必要参数")
            
            model_config = OCIEmbeddingConfig(
                model_name=model_name, # type: ignore
                provider=provider,
                max_tokens=model_params.get("max_tokens", config.max_tokens),
                batch_size=model_params.get("batch_size", 2),
                api_endpoint=api_endpoint, # type: ignore
                compartment_id=compartment_id,
                config_file=config_file
            )
        else:
            raise ValueError(f"不支持的提供者: {provider}")

        model = create_embedding_model(model_config)
        await model.startup()
        self._models[model_id] = model
        self._model_names[model_id] = display_name or model_name # type: ignore
        self._last_used[model_id] = datetime.now()
        logger.success(f"模型 {display_name or model_name} 加载成功")
        return model