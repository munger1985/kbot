import json
from typing import Any, Callable
from loguru import logger
from .model_pool import LLMModelPool
from .model import BaseLLM
from platform_core.dictionary import LLMProvider


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

    def bind_session_factory(self, session_factory: Callable) -> None:
        self._model_pool.set_session_factory(session_factory)

    async def initialize(self):
        """Initialize LLM service and all model pools asynchronously.
        
        Idempotent method that safely initializes the model pool only once.
        Called automatically by other methods if not explicitly initialized.
        
        Logs successful initialization and ensures all model metadata is loaded.
        """
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True
            logger.info("LLM 服务初始化成功。")
        
    async def shutdown(self):
        """Shut down LLM service and clean up all model resources.
        
        Gracefully closes all model client connections and releases resources.
        Resets initialization state to allow re-initialization if needed.
        """
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False
            logger.info("LLM 服务已正常停止。")

    async def get_llm_model(self, model_name: str) -> BaseLLM:
        """Get llm model instance by its unique name."""
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
        response_format: str | dict[str, str] | None = None,
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
            "presence_penalty": presence_penalty,
            "response_format": response_format,
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
            logger.debug(f"正在向模型 {model_name} 发送消息：{processed_messages}")
            if tools:
                logger.debug(f"工具调用配置 - 工具数量：{len(tools)}，工具选择：{tool_choice}")

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
                    logger.debug(f"收到 OpenAI 兼容流式响应（{current_provider}）")
                    
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
                    logger.debug("收到 OCI 原生流式响应")
                    
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
                logger.debug(f"收到非流式响应（{current_provider}）")
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
    
    
