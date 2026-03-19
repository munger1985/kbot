import json
from typing import Any
from loguru import logger
from .model_pool import LLMModelPool
from .model import BaseLLM
from core.dictionary import LLMProvider


class LLMService:
    """Central service for managing LLM model interactions.
    
    Provides a unified interface for:
    - Model pool initialization and lifecycle management
    - Model instance retrieval and warmup
    - Standardized chat completion with cross-provider support
    - Tool call integration (MCP compatible)
    - Stream and non-stream response handling
    - Provider-specific response normalization
    
    This service abstracts provider-specific implementations (OpenAI, OCI, etc.)
    behind a consistent API while maintaining provider-specific optimizations.
    """

    def __init__(self) -> None:
        """Initialize LLM service with empty model pool."""
        self._model_pool = LLMModelPool()
        self._initialized = False

    async def initialize(self):
        """Initialize LLM service and all model pools asynchronously.
        
        Idempotent method that safely initializes the model pool only once.
        Called automatically by other methods if not explicitly initialized.
        
        Logs successful initialization and ensures all model metadata is loaded.
        """
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True
            logger.info("LLM service initialized successfully.")
        
    async def shutdown(self):
        """Shut down LLM service and clean up all model resources.
        
        Gracefully closes all model client connections and releases resources.
        Resets initialization state to allow re-initialization if needed.
        """
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False
            logger.info("LLM service shut down successfully.")
            
    async def get_llm_model(self, model_name: str) -> BaseLLM:
        """Retrieve an LLM model instance by its unique technical name.
        
        Ensures the service is initialized and loads the model if not already in the pool.
        
        Args:
            model_name: Unique technical name of the model (e.g., "gpt-4", "llama3-70b")
            
        Returns:
            BaseLLM: Initialized model instance ready for chat completions
            
        Raises:
            RuntimeError: If model cannot be loaded from the pool
        """
        if not self._initialized:
            await self.initialize()

        return await self._model_pool.load_model(model_name)


    async def chat(
        self,
        model_name: str,
        messages: list[dict[str, str]] | str,
        stream: bool = False,
        timeout: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ):
        """Generate chat completion responses with MCP tool call support.
        
        Unified interface for chat completions across all supported LLM providers.
        Handles message normalization, parameter aggregation, provider-specific
        streaming logic, and error handling.
        
        Args:
            model_name: Technical name of the model to use
            messages: Input messages - either a list of role/content dictionaries
                or a single string (interpreted as user message)
            stream: If True, returns async generator of response chunks;
                if False, returns complete response object
            timeout: Request timeout in seconds (provider-specific handling)
            max_tokens: Maximum number of tokens to generate (overrides model config)
            temperature: Sampling temperature (0-2, higher = more random)
            top_p: Nucleus sampling parameter (0-1)
            frequency_penalty: Penalty for repetitive text (-2 to 2)
            presence_penalty: Penalty for new topics (-2 to 2)
            tools: List of MCP-compatible tool definitions for function calling
            tool_choice: Tool selection strategy ("auto", "none", or specific tool name)

        Returns:
            Union[Any, AsyncGenerator[Any, None]]:
                - If stream=True: Async generator yielding response chunks (provider-specific format)
                - If stream=False: Complete response object with content and usage stats
                
        Raises:
            RuntimeError: If model retrieval or response generation fails
            Exception: Provider-specific errors propagated with context
        """
        try:
            # Step 1: Get model instance (ensures model is loaded and initialized)
            model = await self.get_llm_model(model_name)
            current_provider = model.config.provider  # Unified provider access
        except Exception as e:
            raise RuntimeError(f"从模型池中获取模型 {model_name} 失败: {e}")
        
        # Step 2: Prepare and filter generation parameters (remove None values)
        kwargs = {
            "timeout": timeout,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }

        # Add tool call parameters if provided
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        # Filter out None values to use model config defaults
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        # Step 3: Standardize message format
        try:
            processed_messages = []
            if isinstance(messages, str):
                processed_messages = [{"role": "user", "content": messages}]
            else:
                for msg in messages:
                    if isinstance(msg, dict):
                        processed_messages.append(msg)
                    else:
                        # Convert object-style messages to standard dict format
                        processed_messages.append({
                            "role": getattr(msg, "role", "user"),
                            "content": getattr(msg, "content", "")
                        })
            
            # Log request details for debugging
            logger.debug(f"Sending messages to model {model_name}: {processed_messages}")
            if tools:
                logger.debug(f"Tool call config - Tool count: {len(tools)}, Tool choice: {tool_choice}")

            # Step 4: Handle streaming responses (provider-specific implementation)
            if stream:
                # OpenAI-compatible providers (ChatGPT, Qwen, DeepSeek)
                openai_compatible_providers = [
                    LLMProvider.CHATGPT.value,
                    LLMProvider.API_QWEN.value,
                    LLMProvider.API_DEEPSEEK.value
                ]

                if current_provider in openai_compatible_providers:
                    response = await model.chat(processed_messages, stream=True,** kwargs)
                    logger.debug(f"Received OpenAI-compatible stream response ({current_provider})")
                    
                    async def generate_openai_stream():
                        """Wrapper for OpenAI-compatible streaming responses with error handling."""
                        try:
                            async for chunk in response:  # type: ignore
                                yield chunk
                        except Exception as e:
                            logger.exception(f"OpenAI stream response error: {e}")
                            raise
                    return generate_openai_stream()

                # OCI provider (native streaming implementation)
                elif current_provider == LLMProvider.OCI.value:
                    response = await model.chat(processed_messages, stream=True,** kwargs)
                    logger.debug("Received OCI native stream response")
                    
                    async def generate_oci_stream():
                        """Wrapper for OCI streaming responses with JSON parsing."""
                        try:
                            # OCI SDK uses event stream with JSON payloads
                            for event in response.data.events():  # type: ignore
                                output = json.loads(event.data)
                                yield output
                        except Exception as e:
                            logger.exception(f"OCI stream response error: {e}")
                            raise   
                    return generate_oci_stream()

            # Step 5: Handle non-streaming responses
            else:
                response = await model.chat(processed_messages, stream=False, **kwargs)
                logger.debug(f"Received non-stream response ({current_provider})")
                return response
                
        # General error handling with context
        except Exception as e:
            error_context = (
                f"Model: {model_name}, Messages: {messages}, "
                f"Stream: {stream}, Tools: {len(tools) if tools else 0}"
            )
            logger.exception(f"Error generating chat response - {error_context}: {e}")
            raise RuntimeError(f"Failed to generate chat response: {e}. Context: {error_context}")


    async def warmup(self):
        """
        Preload all models in the pool to memory for faster first requests.
        
        Initializes all configured models upfront to eliminate cold start latency
        for production workloads. Safe to call multiple times (idempotent).
        """
        if not self._initialized:
            await self.initialize()
        
        await self._model_pool.warmup()

    async def load_model(self, model_name: str) -> bool:
        """Load a specific model into memory by its technical name.
        
        Explicitly loads a model into the pool if not already present.
        
        Args:
            model_name: Technical name of the model to load
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.reload_model(model_name)

        
    async def unload_model(self, model_name: str) -> bool:
        """Unload a specific model from memory to free resources.
        
        Gracefully shuts down the model client and removes it from the pool.
        
        Args:
            model_name: Technical name of the model to unload
            
        Returns:
            bool: True if model unloaded successfully, False otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.unload_model(model_name)
    
    async def get_model_instance(self, model_name: str) -> BaseLLM:
        """
        Get a model instance with guaranteed initialization.
        
        Ensures the model is loaded (from config and initialized) if not already
        present in the pool.
        
        Args:
            model_name: Technical name of the model to retrieve
            
        Returns:
            BaseLLM: Fully initialized model instance
            
        Raises:
            RuntimeError: If model fails to load or does not exist
        """
        if not self._initialized:
            await self.initialize()
        
        # load_model handles both retrieval (if exists) and loading (if not)
        instance = await self._model_pool.load_model(model_name)
        if not instance:
            raise RuntimeError(f"Model {model_name} failed to load or does not exist")
        return instance

    def get_provider(self, model_name: str) -> str | None:
        """
        Get the provider of a model (optimized for performance).
        
        First checks loaded models for immediate lookup, then falls back to
        cached model metadata to avoid unnecessary loading.
        
        Args:
            model_name: Technical name of the model
            
        Returns:
            str | None: Provider name (e.g., "openai", "oci") or None if unknown
        """
        # 1. Fast path: Check already loaded models
        model = self._model_pool._models.get(model_name)
        if model:
            return model.config.provider
            
        # 2. Fallback: Check cached model metadata (loaded during initialization)
        metadata = getattr(self._model_pool, '_model_metadata', {}).get(model_name)
        if metadata:
            return metadata.get("provider")
            
        return None

    async def get_max_tokens_limit(self, model_name: str) -> int:
        """
        Get the maximum token limit for a model (async to ensure config is loaded).
        
        Retrieves the configured max_tokens value with safe fallback to 4096
        if model configuration cannot be loaded.
        
        Args:
            model_name: Technical name of the model
            
        Returns:
            int: Maximum token limit for the model (default: 4096)
        """
        try:
            model = await self.get_model_instance(model_name)
            return getattr(model.config, "max_tokens", 4096)
        except Exception:
            # Safe fallback for unknown models or configuration errors
            logger.warning(f"Could not retrieve max tokens for {model_name}, using default 4096")
            return 4096

    async def get_model_config(self, model_name: str) -> Any:
        """Get the complete configuration object for a loaded model.
        
        Ensures the model is loaded before returning its configuration,
        providing access to all provider-specific settings.
        
        Args:
            model_name: Technical name of the model
            
        Returns:
            Any: Model configuration object (subclass of LLMConfig)
        """
        model = await self.get_model_instance(model_name)
        return model.config