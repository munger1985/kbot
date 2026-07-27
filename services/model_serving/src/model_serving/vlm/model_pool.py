import asyncio
from typing import Any
from loguru import logger
from model_serving.config import get_vlm_config
from platform_core.dictionary import ModelCategory, VLMProvider
from model_serving.common.model_pool import BaseModelPool
from .model import *
from .model_factory import create_vlm_model


class VLMModelPool(BaseModelPool[BaseVLM[Any]]):
    """
    VLM Model Pool
    
    Manages the lifecycle of Vision-Language Models (e.g., Qwen-VL API, GPT-4V, etc.)
    including initialization, health checking, resource management, and shutdown.
    Inherits from BaseModelPool to leverage common model pooling functionality while
    implementing VLM-specific lifecycle management logic.
    """

    def _get_model_category(self) -> int:
        """Return VLM category enumeration value
        
        Overrides the base class method to identify this pool as managing VLM models.
        
        Returns:
            int: Integer value of the VLM model category from ModelCategory enum
        """
        return ModelCategory.VLM.value

    async def _shutdown_model_instance(self, model: BaseVLM[Any]):
        """Execute VLM resource cleanup
        
        Overrides base class method to handle VLM-specific resource release by
        calling the model's shutdown method.
        
        Args:
            model: VLM model instance to shut down
        """
        await model.shutdown()

    async def _perform_model_health_check(self, served_model_name: str, model: BaseVLM[Any]):
        """
        Perform model health check
        
        Optimization: Simplified status judgment logic, assuming BaseVLM has a unified
        health_check interface implementation across all subclasses.
        
        Args:
            served_model_name: Name of the model to check
            model: VLM model instance to perform health check on
            
        Raises:
            RuntimeError: If model is not initialized/ready
            Exception: Propagates any errors encountered during health check (triggering
                model reload by the base class)
        """
        try:
            # Compatibility for async/sync health check methods (await directly if properly defined in BaseVLM)
            if asyncio.iscoroutinefunction(model.health_check):
                status = await model.health_check()
            else:
                status = await asyncio.to_thread(model.health_check)

            # Unified judgment logic: Support dict return or object attribute return
            is_ready = False
            if isinstance(status, dict):
                is_ready = status.get('initialized', False)
            else:
                is_ready = getattr(status, 'initialized', False)

            if not is_ready:
                raise RuntimeError(f"模型 {served_model_name} 尚未就绪")
                
            logger.debug(f"VLM 模型 {served_model_name} 健康检查通过")
            
        except Exception as e:
            logger.warning(f"VLM 模型 {served_model_name} 健康检查失败：{e}")
            raise  # Raise exception to trigger reload_model in base class

    async def _start_model(self, served_model_name: str, model_data: dict[str, Any]) -> BaseVLM[Any]:
        """
        Create and start VLM instance
        
        Initializes a new VLM model instance using the factory pattern, configures it
        with provider-specific settings, and starts it up.
        
        Args:
            served_model_name: Name of the model to create
            model_data: Dictionary containing model configuration parameters
            
        Returns:
            BaseVLM[Any]: Initialized and started VLM model instance
            
        Raises:
            ValueError: If required provider parameter is missing
            Exception: If model startup fails
        """
        provider = model_data.get("provider")
        if not provider:
            raise ValueError(f"模型 {served_model_name} 缺少 provider")

        # 1. Get global VLM default configuration
        global_vlm_config = get_vlm_config()

        # 2. Build provider-specific configuration object
        model_config = self._build_config(served_model_name, provider, model_data, global_vlm_config)

        # 3. Create and start model
        model = create_vlm_model(model_config)
        try:
            await model.startup()
            # State management handled by base class BaseModelPool
            logger.success(f"VLM 模型 {served_model_name}（{provider}）启动成功")
            return model
        except Exception as e:
            logger.error(f"VLM 模型 {served_model_name} 启动失败：{e}")
            raise

    def _build_config(self, name: str, provider: str, data: dict[str, Any], global_cfg: Any) -> VLMConfig:
        """
        Configuration transformation mapper
        
        Converts raw model configuration data into a typed VLMConfig subclass instance
        appropriate for the specified provider.
        
        Args:
            name: Model name
            provider: Model provider identifier
            data: Raw model configuration dictionary
            global_cfg: Global VLM configuration defaults
            
        Returns:
            VLMConfig: Typed configuration object for the specified provider
            
        Raises:
            ValueError: If required API parameters are missing
            NotImplementedError: If the provider is not supported
        """
        # Extract model parameters with fallback to empty dict
        params = data.get("model_params", {})
        provider_model_name = data["provider_model_name"]
        api_endpoint = data.get("api_endpoint")
        api_key = data.get("api_key")

        # Extract common parameters
        common_kwargs = {
            "model_name": provider_model_name,
            "provider": provider,
            "max_tokens": params.get("max_tokens", 512),
            "temperature": params.get("temperature", 0.1),
        }

        # 1. OpenAI-compatible VLM APIs (e.g., Qwen-VL-Plus/Max)
        if provider == VLMProvider.API_QWEN.value:
            if not api_endpoint or not api_key:
                raise ValueError(f"Model {name} missing API parameters (endpoint/key)")

            return OpenAIVLMConfig(
                **common_kwargs,
                api_key=api_key,
                api_endpoint=api_endpoint,
                api_version=params.get("api_version", ""),
                timeout=params.get("timeout", global_cfg.timeout),
                max_retries=params.get("max_retries", 3)
            )

        # Extend here for other providers (e.g., LocalVLMConfig for Llava/Qwen-VL local deployments)
        
        raise NotImplementedError(f"Unsupported VLM provider: {provider}")
