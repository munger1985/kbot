from core.config.settings import get_llm_config
from core.dictionary import ModelCategory, LLMProvider
from loguru import logger
from typing import Any
from microservices.common.model_pool import BaseModelPool
from .model import *
from .model_factory import create_llm_model


class LLMModelPool(BaseModelPool[BaseLLM[Any]]):
    """
    Model pool for managing Large Language Model (LLM) instances.
    
    This specialized model pool handles lifecycle management for LLM implementations
    including OpenAI-compatible APIs, OCI Generative AI, and other providers. It
    extends the base model pool with LLM-specific initialization, health checking,
    and resource cleanup logic.
    
    Type Parameters:
        BaseLLM[Any]: Generic LLM base class with provider-specific configuration
    """

    def _get_model_category(self) -> int:
        """Return the enum value for LLM model category."""
        return ModelCategory.LLM.value

    async def _shutdown_model_instance(self, model: BaseLLM[Any]):
        """Shut down LLM instance and release associated resources.
        
        Implements provider-specific cleanup logic including closing HTTP client
        sessions, releasing connections, and cleaning up any open resources.
        
        Args:
            model: LLM instance to shut down
        """
        await model.shutdown()

    async def _perform_model_health_check(self, model_name: str, model: BaseLLM[Any]):
        """
        Perform health check on LLM instance using minimal chat completion.
        
        Validates API connectivity and basic functionality with a minimal request
        (short prompt, minimal tokens, non-streaming) to minimize token usage and
        latency while ensuring the model is operational.
        
        Args:
            model_name: Name of the model to check
            model: LLM instance to perform health check on
            
        Raises:
            Exception: If health check request fails (API error, timeout, etc.)
        """
        # Execute minimal health check request
        await model.chat(
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            max_tokens=2
        )
        logger.debug(f"🔍 LLM model {model_name} health check passed")

    async def _start_model(self, model_name: str, model_data: dict[str, Any]) -> BaseLLM[Any]:
        """
        Create and initialize an LLM instance from configuration data.
        
        Orchestrates the complete LLM instantiation process:
        1. Retrieves global LLM configuration defaults
        2. Builds provider-specific configuration object
        3. Creates model instance via factory pattern
        4. Initializes the model (connects to API)
        5. Returns ready-to-use model instance
        
        Args:
            model_name: Unique technical name of the model
            model_data: Dictionary containing model configuration from database/storage
            
        Returns:
            BaseLLM[Any]: Fully initialized LLM instance ready for chat completions
            
        Raises:
            ValueError: If required configuration parameters are missing
            Exception: If model creation or initialization fails
        """
        # Extract required provider identifier
        provider = model_data.get("provider")
        if not provider:
            raise ValueError(f"Missing required parameter 'provider' for model {model_name}")

        # Step 1: Get global default LLM configuration
        global_config = get_llm_config()
        
        # Step 2: Build provider-specific configuration object
        model_config = self._build_config(model_name, provider, model_data, global_config)

        # Step 3: Create model instance via factory pattern
        model = create_llm_model(model_config)
        
        try:
            # Initialize model (connect to API, create clients, etc.)
            await model.startup()
            
            # Instance state managed by parent class load_model method
            logger.success(f"🚀 LLM model {model_name} ({provider}) loaded into pool successfully")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to start LLM model {model_name}: {e}")
            raise

    def _build_config(self, name: str, provider: str, data: dict[str, Any], global_cfg: Any) -> LLMConfig:
        """
        Map raw configuration data to strongly-typed provider-specific config objects.
        
        Combines global defaults with model-specific parameters and validates
        provider-specific requirements (API keys, endpoints, etc.).
        
        Args:
            name: Model technical name
            provider: LLM provider identifier (e.g., "openai", "oci")
            data: Raw model configuration data (from database/storage)
            global_cfg: Global LLM configuration defaults
            
        Returns:
            LLMConfig: Provider-specific configuration object (OpenaiLLMConfig/OCILLMConfig)
            
        Raises:
            ValueError: If required provider-specific parameters are missing
            ValueError: If provider is not supported
        """
        # Extract model parameters with fallback to empty dict
        params = data.get("model_params", {})
        model_tech_name = data.get("model_tech_name", name)
        # Extract common connection parameters
        api_key = data.get("api_key")
        api_endpoint = data.get("api_endpoint")

        # Common LLM parameters with global defaults fallback
        common_kwargs = {
            "model_name": model_tech_name,
            "provider": provider,
            "temperature": params.get("temperature", global_cfg.temperature),
            "max_tokens": params.get("max_tokens", global_cfg.max_tokens),
            "top_p": params.get("top_p", global_cfg.top_p),
            "frequency_penalty": params.get("frequency_penalty", global_cfg.frequency_penalty),
            "presence_penalty": params.get("presence_penalty", global_cfg.presence_penalty),
        }

        # 1. OpenAI-compatible providers (DeepSeek, Qwen API, ChatGPT)
        openai_providers = [
            LLMProvider.API_DEEPSEEK.value, 
            LLMProvider.API_QWEN.value, 
            LLMProvider.CHATGPT.value
        ]
        if provider in openai_providers:
            # Validate required OpenAI-compatible parameters
            if not api_key or not api_endpoint:
                raise ValueError(f"OpenAI-compatible model {name} missing required parameters: api_key or api_endpoint")
            
            # Create OpenAI-specific configuration
            return OpenaiLLMConfig(
                **common_kwargs,
                api_key=api_key,
                api_endpoint=api_endpoint,
                timeout=params.get("timeout", global_cfg.timeout)
            )

        # 2. Oracle Cloud Infrastructure (OCI) provider
        if provider == LLMProvider.OCI.value:
            # Extract OCI-specific parameters
            compartment_id = params.get("compartment_id")
            config_file = params.get("config_file")
            
            # Validate OCI-specific requirements
            if not all([api_endpoint, compartment_id, config_file]):
                raise ValueError(f"OCI model {name} missing required parameters (compartment_id/config_file/endpoint)")

            # Create OCI-specific configuration
            return OCILLMConfig(
                **common_kwargs,
                api_endpoint=api_endpoint,  # type: ignore
                compartment_id=compartment_id,
                config_file=config_file,
                top_k=params.get("top_k", global_cfg.top_k)
            )

        # Unsupported provider fallback
        raise ValueError(f"Unsupported LLM provider: {provider}")
