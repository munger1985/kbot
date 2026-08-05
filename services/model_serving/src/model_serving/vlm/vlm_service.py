import json
from loguru import logger
from typing import Any, AsyncGenerator, Callable
from .model_pool import VLMModelPool
from .model import BaseVLM


class VLMService:
    """
    Unified VLM (Vision-Language Model) Service

    Manages the lifecycle and inference operations for different VLM models through
    a centralized model pool, providing a consistent interface for model interaction.
    """

    def __init__(self):
        """
        Initialize VLM Service

        Sets up the model pool instance and initialization state tracking.
        """
        self._model_pool = VLMModelPool()
        self._initialized = False

    def bind_session_factory(self, session_factory: Callable) -> None:
        self._model_pool.set_session_factory(session_factory)

    async def initialize(self):
        """Initialize VLM service and model pool

        Performs one-time setup for the VLM service, including initializing the
        underlying model pool with all configured models.
        """
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True
            logger.info("VLM 服务初始化成功")

    async def shutdown(self):
        """Shut down VLM service and all managed models

        Gracefully terminates all model instances in the pool and cleans up
        associated resources.
        """
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False
            logger.info("VLM 服务已正常停止")

    async def invalidate_model(self, served_model_name: str) -> None:
        """配置变更后立即移除旧实例，不触发模型服务初始化。"""
        if self._initialized:
            await self._model_pool.unload_model(served_model_name)

    def is_model_loaded(self, served_model_name: str) -> bool:
        return self._model_pool.is_model_loaded(served_model_name)

    async def get_vlm_model(self, served_model_name: str) -> BaseVLM:
        """Retrieve VLM model instance by unique model name

        Args:
            served_model_name: Unique technical name of the VLM model

        Returns:
            BaseVLM: Initialized VLM model instance from the pool
        """
        if not self._initialized:
            await self.initialize()

        return await self._model_pool.load_model(served_model_name)

    async def inference(self,
                        served_model_name: str,
                        messages: list[dict[str, Any]],
                        stream: bool = False,
                        timeout: int | None = None,
                        max_tokens: int | None = None,
                        temperature: float | None = None,
                        top_p: float | None = None,
                        frequency_penalty: float | None = None,
                        presence_penalty: float | None = None
                    ) -> dict[str, Any] | AsyncGenerator[str, None]:
        """
        Execute inference with a VLM model

        Performs vision-language inference using the specified model with the given
        parameters, supporting both streaming and non-streaming response modes.

        Args:
            served_model_name: Technical name of the target VLM model
            messages: List of message dictionaries (supports text + image content)
            stream: If True, returns an async generator of response chunks
            timeout: Request timeout in seconds
            max_tokens: Maximum number of tokens to generate (0-4096 typically)
            temperature: Sampling temperature (0.0-2.0, lower = more deterministic)
            top_p: Nucleus sampling parameter (0.0-1.0)
            frequency_penalty: Penalty for repeated token generation (-2.0-2.0)
            presence_penalty: Penalty for new topic introduction (-2.0-2.0)

        Returns:
            Union[dict[str, Any], AsyncGenerator[str, None]]:
                - If stream=False: Complete response dict with content and token usage
                - If stream=True: Async generator yielding response chunks as strings

        Raises:
            RuntimeError: If inference fails at any stage
            ValueError: If response format is invalid or incomplete
        """

        try:
            # Retrieve model instance from pool
            model = await self.get_vlm_model(served_model_name)

            # Prepare inference parameters (filter out None values)
            kwargs = {
                "timeout": timeout,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            # Execute inference with the model
            try:
                logger.debug(f"正在向模型发送消息：{served_model_name}")
                response = await model.inference(messages, stream=stream, **kwargs)
                logger.debug(f"收到响应类型：{type(response)}")
            except Exception as e:
                logger.error(f"生成响应失败：{e}")
                raise RuntimeError(f"Failed to generate chat response: {e}")

            if stream:
                # Handle streaming response
                async def generate_stream():
                    try:
                        content_parts = []
                        last_chunk = None

                        async for chunk in response: # type: ignore
                            logger.debug(f"收到响应块类型：{type(chunk)}")
                            last_chunk = chunk

                            if not hasattr(chunk, 'choices'):
                                logger.warning("收到无效响应块：缺少 'choices' 属性")
                                continue

                            if not chunk.choices:
                                logger.warning("收到 choices 为空的响应块")
                                continue

                            if not hasattr(chunk.choices[0], 'delta'):
                                logger.warning("收到无效 choice：缺少 'delta' 属性")
                                continue

                            delta = chunk.choices[0].delta
                            if delta and hasattr(delta, 'content'):
                                content = delta.content
                                if content is not None:
                                    content_parts.append(str(content))
                                    yield str(content)
                                else:
                                    logger.debug("收到 content 为空的 delta")
                            else:
                                logger.debug("收到不含 content 字段的 delta")

                        # Append token usage statistics after stream completion
                        if hasattr(last_chunk, 'usage'):
                            yield "\n\n=== USAGE ===\n" + json.dumps({
                                "total_tokens": int(last_chunk.usage.total_tokens), # type: ignore
                                "prompt_tokens": int(last_chunk.usage.prompt_tokens), # type: ignore
                                "completion_tokens": int(last_chunk.usage.completion_tokens) # type: ignore
                            })
                        elif hasattr(response, 'usage'):
                            yield "\n\n=== USAGE ===\n" + json.dumps({
                                "total_tokens": int(response.usage.total_tokens), # type: ignore
                                "prompt_tokens": int(response.usage.prompt_tokens), # type: ignore
                                "completion_tokens": int(response.usage.completion_tokens) # type: ignore
                            })

                        # Append full response metadata after usage stats
                        if last_chunk:
                            yield "\n\n=== FULL RESPONSE ===\n" + json.dumps({
                                "id": last_chunk.id,
                                "choices": [{
                                    "delta": {
                                        "content": last_chunk.choices[0].delta.content,
                                        "role": last_chunk.choices[0].delta.role,
                                        "function_call": last_chunk.choices[0].delta.function_call,
                                        "tool_calls": last_chunk.choices[0].delta.tool_calls
                                    },
                                    "finish_reason": last_chunk.choices[0].finish_reason,
                                    "index": last_chunk.choices[0].index
                                }],
                                "created": last_chunk.created,
                                "model": last_chunk.model,
                                "object": last_chunk.object,
                                "system_fingerprint": last_chunk.system_fingerprint if hasattr(last_chunk, 'system_fingerprint') else None,
                                "service_tier": last_chunk.service_tier if hasattr(last_chunk, 'service_tier') else None
                            })

                    except Exception as e:
                        logger.error(f"处理流式响应失败：{e}")
                        raise

                return generate_stream()
            else:
                # Handle non-streaming response
                logger.debug(f"收到响应类型：{type(response)}")

                if not hasattr(response, 'choices'):
                    raise ValueError("Invalid response format - missing 'choices' attribute")

                if not response.choices: # type: ignore
                    raise ValueError("Invalid completion format - no choices available")

                if not hasattr(response.choices[0], 'message'): # type: ignore
                    raise ValueError("Invalid choice format - missing 'message' attribute")

                message = response.choices[0].message # type: ignore
                if not message or not hasattr(message, 'content'):
                    raise ValueError("Invalid message format - missing 'content' attribute")

                if not message.content:
                    raise ValueError("Invalid message format - empty content")

                if not hasattr(response, 'usage'):
                    raise ValueError("Invalid completion format - missing 'usage' attribute")

                if not response.usage: # type: ignore
                    raise ValueError("Invalid completion format - no usage data available")

                if not all(hasattr(response.usage, attr)  # type: ignore
                          for attr in ['total_tokens', 'prompt_tokens', 'completion_tokens']):
                    raise ValueError("Invalid usage format - missing required attributes")

                # Construct standardized response format
                return {
                    "id": response.id, # type: ignore
                    "object": "chat.completion",
                    "created": response.created, # type: ignore
                    "model": response.model, # type: ignore
                    "choices": [{
                        "index": response.choices[0].index, # type: ignore
                        "message": {
                            "role": message.role,
                            "content": str(message.content),
                            "function_call": message.function_call,
                            "tool_calls": message.tool_calls
                        },
                        "finish_reason": response.choices[0].finish_reason # type: ignore
                    }],
                    "usage": {
                        "prompt_tokens": int(response.usage.prompt_tokens), # type: ignore
                        "completion_tokens": int(response.usage.completion_tokens), # type: ignore
                        "total_tokens": int(response.usage.total_tokens) # type: ignore
                    },
                    "system_fingerprint": response.system_fingerprint if hasattr(response, 'system_fingerprint') else None, # type: ignore
                    "service_tier": response.service_tier if hasattr(response, 'service_tier') else None # type: ignore
                }

        except Exception as e:
            logger.error(f"生成对话响应失败：{e}")
            raise RuntimeError(f"Failed to generate chat response: {e}")

    async def warmup(self):
        """
        Warm up all models in the model pool

        Pre-initializes all configured VLM models to reduce first-request latency.
        This method should be called during service startup for optimal performance.
        """
        if not self._initialized:
            await self.initialize()

        await self._model_pool.warmup()

    async def load_model(self, served_model_name: str) -> bool:
        """Load model into memory by technical name

        Explicitly loads (or reloads) a specific model into the pool, initializing
        all required resources for inference.

        Args:
            served_model_name: Technical name of the model to load

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not self._initialized:
            await self.initialize()

        return await self._model_pool.reload_model(served_model_name)


    async def unload_model(self, served_model_name: str) -> bool:
        """Unload model from memory by technical name

        Gracefully shuts down and removes a specific model from the pool,
        freeing up system resources.

        Args:
            served_model_name: Technical name of the model to unload

        Returns:
            bool: True if model unloaded successfully, False otherwise
        """
        if not self._initialized:
            await self.initialize()

        return await self._model_pool.unload_model(served_model_name)
