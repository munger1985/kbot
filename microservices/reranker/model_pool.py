from core.config.settings import get_reranker_config
from core.dictionary import ModelCategory, RerankerProvider
from loguru import logger
from datetime import datetime
from typing import Any
from microservices.common.model_pool import BaseModelPool
from .model import *


class RerankerModelPool(BaseModelPool[BaseReranker]):
    """Reranker 模型池"""
    
    def _get_model_category(self) -> str:
        return ModelCategory.RERANKER.value

    async def _shutdown_model_instance(self, model: BaseReranker):
        await model.shutdown()

    async def _perform_model_health_check(self, model_id: int, model: BaseReranker):
        await model.rerank(query="test", documents=["test"], top_k=1)
        logger.success(f"模型 {self._model_names.get(model_id, str(model_id))} 健康检查通过")

    async def _start_model(self, model_id: int, model_data: dict[str, Any]) -> BaseReranker:
        display_name = model_data.get("display_name")
        model_name = model_data.get("model_name")
        provider = model_data.get("provider")
        model_params = model_data.get("model_params", {})
        
        if not all([model_name, provider]):
            raise ValueError(f"模型 {display_name or model_id} 缺少必要参数")

        config = get_reranker_config()
        
        if provider == RerankerProvider.LOCAL.value:
            if "jina" in model_name.lower():
                model_config = JinaRerankerConfig(
                    provider=provider,
                    model_name=model_name,
                    model_path=model_params.get("model_path", None),
                    device=model_params.get("device", None),
                    device_map=model_params.get("device_map", None),
                    max_tokens=model_params.get("max_tokens", 512),
                    batch_size=model_params.get("batch_size", 16),
                    compile_model=model_params.get("compile_model", True),
                    use_fp16=model_params.get("use_fp16", True),
                    trust_remote_code=model_params.get("trust_remote_code", True),
                    local_files_only=model_params.get("local_files_only", False),
                    max_memory=model_params.get("max_memory", None),
                    cache_dir=config.cache_dir
                )
            elif "qwen" in model_name.lower():
                model_config = Qwen3RerankerConfig(
                    provider=provider,
                    model_name=model_name,
                    model_path=model_params.get("model_path", None),
                    device=model_params.get("device", None),
                    max_tokens=model_params.get("max_tokens", 8192),
                    batch_size=1,
                    use_fp16=model_params.get("use_fp16", True),
                    use_flash_attention=model_params.get("use_flash_attention", True),
                    instruction=model_params.get("instruction", None)
                )
            else:
                model_config = LocalRerankerConfig(
                    provider=provider,
                    model_name=model_name,
                    model_path=model_params.get("model_path", None),
                    device=model_params.get("device", None),
                    device_map=model_params.get("device_map", None),
                    max_tokens=model_params.get("max_tokens", 8192),
                    batch_size=model_params.get("batch_size", 16),
                    compile_model=model_params.get("compile_model", True),
                    use_fp16=model_params.get("use_fp16", False),
                    trust_remote_code=model_params.get("trust_remote_code", True),
                    local_files_only=model_params.get("local_files_only", False),
                    max_memory=model_params.get("max_memory", None),
                    cache_dir=config.cache_dir or "./cached_models"
                )
            
        elif provider == RerankerProvider.COHERE.value:
            api_endpoint = model_data.get("api_endpoint")
            api_key = model_data.get("api_key")

            if not api_endpoint or not api_key:
                raise ValueError(f"模型 {display_name or model_name} 缺少 API 端点或 API 密钥")
            
            model_config = CohereRerankerConfig(
                provider=provider,
                model_name=model_name,
                max_tokens=model_params.get("max_tokens", 8192),
                api_key=api_key,
                api_endpoint=api_endpoint,
                timeout=model_params.get("timeout", 10)
            )
        else:
            raise ValueError(f"不支持的 reranker 模型: {provider}")
        
        model = create_reranker_model(model_config)
        await model.startup()
        self._models[model_id] = model
        self._model_names[model_id] = display_name or model_name
        self._last_used[model_id] = datetime.now()
        logger.success(f"模型 {display_name or model_name} 加载成功")
        return model