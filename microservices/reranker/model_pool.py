from core.config.settings import get_reranker_config
from core.dictionary import ModelCategory, RerankerProvider
from loguru import logger
from typing import Any

from microservices.common.model_pool import BaseModelPool
from .model import *
from .model_factory import create_reranker_model

class RerankerModelPool(BaseModelPool[BaseReranker[Any]]):
    """
    Specialized model pool for managing Reranker model instances.
    
    Key optimizations:
    1. Unified configuration mapping logic across all reranker providers
    2. Removed duplicate self._models assignments in subclasses (managed by parent load_model)
    3. Enhanced type hints for better compatibility with generic base class
    4. Centralized health check and lifecycle management
    
    Type Parameters:
        BaseReranker[Any]: Generic reranker base class with provider-specific configuration
    """
    
    def _get_model_category(self) -> int:
        """Return the enum value for Reranker model category.
        
        Required implementation from BaseModelPool to identify this pool's
        model type for metadata management and configuration.
        
        Returns:
            int: Numeric value of ModelCategory.RERANKER enum
        """
        return ModelCategory.RERANKER.value

    async def _shutdown_model_instance(self, model: BaseReranker[Any]):
        """Shut down reranker model instance and release resources.
        
        Implements provider-specific cleanup logic by delegating to the
        model's own shutdown method.
        
        Args:
            model: Reranker instance to shut down
        """
        await model.shutdown()

    async def _perform_model_health_check(self, model_name: str, model: BaseReranker[Any]):
        """
        Perform health check on reranker model instance.
        
        Validates model functionality with a minimal reranking task (ping/pong)
        to ensure the model is operational and responsive.
        
        Args:
            model_name: Name of the model to check
            model: Reranker instance to perform health check on
            
        Raises:
            Exception: If health check request fails (model unresponsive)
        """
        await model.rerank(query="ping", documents=["pong"], top_k=1)
        logger.debug(f"🔍 Reranker model {model_name} health check passed")

    async def _start_model(self, model_name: str, model_data: dict[str, Any]) -> BaseReranker[Any]:
        """
        Construct configuration and initialize a reranker model instance.
        
        Orchestrates the complete reranker instantiation process:
        1. Retrieves global reranker configuration defaults
        2. Builds provider-specific configuration object
        3. Creates model instance via factory pattern
        4. Initializes the model (loads weights/connects to API)
        
        Args:
            model_name: Unique technical name of the model
            model_data: Dictionary containing model configuration from database/storage
            
        Returns:
            BaseReranker[Any]: Fully initialized reranker instance ready for reranking
            
        Raises:
            ValueError: If required configuration parameters are missing
            Exception: If model creation or initialization fails
        """
        # Extract required provider identifier
        provider = model_data.get("provider")
        if not provider:
            raise ValueError(f"Missing required parameter 'provider' for model {model_name}")

        # Step 1: Get global default reranker configuration
        global_config = get_reranker_config()
        
        # Step 2: Build provider-specific configuration object
        model_config = self._build_config(model_name, provider, model_data, global_config)

        # Step 3: Create model instance via factory pattern and initialize
        model = create_reranker_model(model_config)
        try:
            await model.startup()
            # Important: Do not manually assign to self._models - parent class load_model
            # manages instance state to ensure consistency
            logger.success(f"🚀 Reranker model {model_name} ({provider}) initialized successfully")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to initialize Reranker model {model_name}: {e}")
            raise

    def _build_config(self, name: str, provider: str, data: dict[str, Any], global_cfg: Any) -> RerankerConfig:
        """
        Map raw database configuration to strongly-typed reranker config objects.
        
        Converts generic configuration data into provider-specific configuration
        objects with proper validation of required parameters (paths, API keys, etc.).
        
        Args:
            name: Model technical name
            provider: Reranker provider identifier (e.g., "local_qwen", "cohere")
            data: Raw model configuration data (from database/storage)
            global_cfg: Global reranker configuration defaults
            
        Returns:
            RerankerConfig: Provider-specific configuration object (BGERerankerConfig/Qwen3RerankerConfig/etc.)
            
        Raises:
            ValueError: If required provider-specific parameters are missing
            ValueError: If provider is not recognized/supported
        """
        # Extract model parameters with fallback to empty dict
        params = data.get("model_params", {})
        model_tech_name = data.get("model_tech_name", name)
        # Extract common connection/configuration parameters
        model_path = data.get("model_path")
        api_key = data.get("api_key")
        api_endpoint = data.get("api_endpoint")

        # Common reranker parameters with sensible defaults
        common_kwargs = {
            "provider": provider,
            "model_name": model_tech_name,
            "max_tokens": params.get("max_tokens", 8192),
            "batch_size": params.get("batch_size", 16),
        }

        # 1. Local model handling (Qwen3/BGE)
        if provider in [RerankerProvider.LOCAL_QWEN.value, RerankerProvider.LOCAL_BGE.value]:
            # Validate required local model parameter
            if not model_path:
                raise ValueError(f"Local reranker model {name} missing required parameter: model_path")
            
            # Select appropriate config class based on provider
            config_class: Any = Qwen3RerankerConfig if provider == RerankerProvider.LOCAL_QWEN.value else BGERerankerConfig
            
            # Create provider-specific configuration
            return config_class(
                **common_kwargs,
                model_path=model_path,
                device=params.get("device"),
                use_fp16=params.get("use_fp16", provider == RerankerProvider.LOCAL_QWEN.value),
                # Qwen3-specific instruction parameter (None for BGE)
                instruction=params.get("instruction") if provider == RerankerProvider.LOCAL_QWEN.value else None
            )

        # 2. Qwen Reranker API handling (DashScope / 阿里云百炼)
        elif provider == RerankerProvider.API_QWEN.value:
            # Validate required API key
            if not api_key:
                raise ValueError(f"Qwen API reranker model {name} missing required parameter: api_key")
            
            # 允许数据库覆盖默认的百炼 Endpoint，如果没有则使用我们类里定义的 default
            config_kwargs = {
                **common_kwargs,
                "api_key": api_key,
                "timeout": params.get("timeout", getattr(global_cfg, "timeout", 30))
            }
            if api_endpoint:
                config_kwargs["api_endpoint"] = api_endpoint

            return QwenRerankerConfig(**config_kwargs)
        
        # # 3. Cohere API handling
        # if provider == RerankerProvider.COHERE.value:
        #     # Validate required Cohere parameter
        #     if not api_key:
        #         raise ValueError(f"Cohere reranker model {name} missing required parameter: api_key")
            
        #     # Create Cohere-specific configuration
        #     return CohereRerankerConfig(
        #         **common_kwargs,
        #         api_key=api_key,
        #         batch_size=params.get("batch_size", 1000),  # Cohere supports large batch sizes
        #         timeout=params.get("timeout", global_cfg.timeout)
        #     )

        # # 4. Generic OpenAI-compatible API handling
        # if provider in [RerankerProvider.API_QWEN.value, RerankerProvider.CHATGPT.value]:
        #     # Validate required API parameters
        #     if not api_key or not api_endpoint:
        #         raise ValueError(f"OpenAI-compatible reranker model {name} missing required parameters: api_key or api_endpoint")
            
        #     # Create OpenAI-compatible configuration
        #     return OpenAIRerankerConfig(
        #         **common_kwargs,
        #         api_key=api_key,
        #         api_endpoint=api_endpoint,
        #         timeout=params.get("timeout", global_cfg.timeout)
        #     )

        # Unsupported provider fallback
        raise ValueError(f"Unknown/unsupported Reranker provider: {provider}")