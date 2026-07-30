from model_serving.config import get_embedding_config
from platform_core.dictionary import ModelCategory, EmbeddingProvider
from loguru import logger
from typing import Any
from model_serving.common.model_pool import BaseModelPool
from .model import *
from .model_factory import create_embedding_model


class EmbeddingModelPool(BaseModelPool[BaseEmbedding[Any]]):
    """
    Embedding model pool implementation.
    Responsible for lifecycle management and configuration mapping of specific Embedding models.
    """

    def _get_model_category(self) -> int:
        """Define the database model category managed by this pool"""
        return ModelCategory.TXT_EMBEDDING.value

    async def _shutdown_model_instance(self, model: BaseEmbedding[Any]) -> None:
        """Invoke specific model's shutdown logic"""
        await model.shutdown()

    async def _perform_model_health_check(self, served_model_name: str, model: BaseEmbedding[Any]) -> None:
        """
        Perform lightweight inference health check.
        Note: Wrapped with try-except in parent class, only focus on check action here.
        """
        # Use an extremely short text for health check
        await model.embed(["ping"], batch_size=1)
        logger.debug(f"Embedding 模型 {served_model_name} 健康检查通过")

    async def _start_model(self, served_model_name: str, model_data: dict[str, Any]) -> BaseEmbedding[Any]:
        """
        Construct configuration and start model instance.
        Improvement: Extract configuration construction logic to ensure single responsibility principle.
        """
        provider = model_data.get("provider")
        if not provider:
            raise ValueError(f"模型 {served_model_name} 缺少 provider")

        # 1. Get base global configuration (as fallback)
        global_config = get_embedding_config()
        
        # 2. Construct Provider-specific Config object
        model_config = self._build_config(served_model_name, provider, model_data, global_config)

        # 3. Create model via factory function
        model = create_embedding_model(model_config)
        
        # 4. Initialize model resources
        try:
            await model.startup()
            # 父类在启动成功后统一登记实例，子类只负责构建模型。
            logger.success(
                f"Embedding 模型 {served_model_name}（{provider}）启动成功"
            )
            return model
        except Exception as e:
            logger.error(f"Embedding 模型 {served_model_name} 启动失败：{e}")
            raise

    def _build_config(self, name: str, provider: str, data: dict[str, Any], global_cfg: Any) -> EmbeddingConfig:
        """
        Map database data to specific Pydantic Config objects.
        """
        params = data.get("model_params", {})
        provider_model_name = data["provider_model_name"]
        path = data.get("model_path", None)
        api_key = data.get("api_key", None)
        api_endpoint = data.get("api_endpoint", None)
        
        # Extract common parameters (high reuse frequency)
        common_kwargs = {
            "model_name": provider_model_name,
            "provider": provider,
            "max_tokens": params.get("max_tokens", global_cfg.max_tokens),
            "batch_size": params.get("batch_size", 2),
        }

        # Map based on Provider type
        if provider == EmbeddingProvider.LOCAL_QWEN.value:
            if not path: raise ValueError(f"{name} missing model_path")
            return Qwen3EmbeddingConfig(
                **common_kwargs,
                model_path=path,
                device=params.get("device"),
                use_fp16=params.get("use_fp16", True),
                instruction=params.get("instruction")
            )

        if provider == EmbeddingProvider.LOCAL_BGE.value:
            if not path: raise ValueError(f"{name} missing model_path")
            return BGEEmbeddingConfig(
                **common_kwargs,
                model_path=path,
                device=params.get("device"),
                use_fp16=params.get("use_fp16", False),
                query_instruction=params.get("query_instruction"),
                pooling_strategy=params.get("pooling_strategy", "cls")
            )

        if provider in [EmbeddingProvider.API_QWEN.value, EmbeddingProvider.CHATGPT.value]:
            if not api_key: raise ValueError(f"{name} missing api_key")
            # Remove trailing "/embeddings" from api_endpoint to avoid path duplication
            if api_endpoint and api_endpoint.endswith("/embeddings"):
                api_endpoint = api_endpoint[:-11]
            return OpenAIEmbeddingConfig(
                **common_kwargs,
                api_key=api_key,
                api_base=api_endpoint,
                # 模型目录中的声明维度是 KC、模型服务和实际供应商请求的唯一契约。
                dimensions=params.get("embedding_dimension"),
                timeout=params.get("timeout", global_cfg.timeout),
                max_retries=params.get("max_retries", 3)
            )

        if provider == EmbeddingProvider.OCI.value:
            compartment_id = params.get("compartment_id")
            config_file = params.get("config_file")
            
            if not all([api_endpoint, compartment_id, config_file]):
                raise ValueError(f"OCI model {name} missing required parameters (compartment_id/config_file/endpoint)")

            return OCIEmbeddingConfig(
                **common_kwargs,
                compartment_id=compartment_id,
                config_file=config_file,
                api_endpoint=api_endpoint, # type: ignore
                input_type_doc=params.get("input_type_doc", "search_document"),
                input_type_query=params.get("input_type_query", "search_query")
            )

        raise ValueError(f"Unimplemented or unsupported Provider: {provider}")
