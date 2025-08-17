"""LLM service implementation."""

from loguru import logger
import os
import sys
import json
from model_pool import ModelPool
from model import BaseLLM, LLMProvider

# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)




class LLMService:
    """LLM service class."""

    def __init__(self) -> None:
        """Initialize LLM service. //初始化LLM服务"""
        self._model_pool = ModelPool()
        self._initialized = False
        

    async def initialize(self):
        """
        Initialize LLM service and all model pools.//初始化LLM服务和所有模型池
        """
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True
            logger.info("LLM service initialized")
        
    async def shutdown(self):
        """
        Close all LLM service and model pools.//关闭LLM服务和所有模型池
        """
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False
            logger.info("LLM service has been shutdown")
            
    async def get_llm_model(self, model_unique_name: str) -> BaseLLM:
        """Retrieve a LLM model by ID. //获取指定ID的LLM模型

        Args:
            model_unique_name: 模型ID

        Returns:
            LLM model instance //LLM模型实例
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.load_model(model_unique_name)

    async def chat(
        self,
        model_unique_name: str,
        messages: list[dict[str, str]] | str,
        stream: bool = False,
        timeout: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None
    ):
        """Generate a chat response.

        Args:
            model_unique_name: Model ID
            messages: list of messages or single prompt string
            stream: Whether to stream the response
            timeout: Timeout in seconds
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty

        Returns:
            If stream is True: AsyncGenerator yielding text chunks
            If stream is False: Dictionary with content and usage stats
        """
        try:
            # Get model from pool
            model = await self.get_llm_model(model_unique_name)
            
            # Prepare parameters
            kwargs = {
                "timeout": timeout,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            
            # Get response from model
            try:
                # Ensure messages are in correct format
                processed_messages = []
                if isinstance(messages, str):
                    processed_messages = [{"role": "user", "content": messages}]
                else:
                    for msg in messages:
                        if isinstance(msg, dict):
                            processed_messages.append(msg)
                        else:
                            # Convert message objects to dict if needed
                            processed_messages.append({
                                "role": getattr(msg, "role", "user"),
                                "content": getattr(msg, "content", "")
                            })
                
                logger.debug(f"Sending messages to model: {processed_messages}")

                # OpenAI stream mode
                if stream and model.provider == LLMProvider.OPENAI.value:
                    response = await model.chat(processed_messages, stream=True, **kwargs)
                    logger.debug("OpenAI streaming response received.")
                    async def generate_openai_stream():
                        try:
                            async for chunk in response: # type: ignore
                                    yield chunk
                                
                        except Exception as e:
                            logger.exception(f"Error in OpenAI streaming response: {e}")
                            raise
                            
                    return generate_openai_stream()
                
                # OCI stream mode
                elif stream and model.provider == LLMProvider.OCI.value:
                    response = await model.chat(processed_messages, stream=True, **kwargs)
                    logger.debug("Non-openai streaming response received.")
                    async def generate_oci_stream():
                        try:
                            for event in response.data.events(): # type: ignore
                                output =  json.loads(event.data)
                                yield output
                        except Exception as e:
                            logger.exception(f"Error in streaming response: {e}")
                            raise   

                    return generate_oci_stream()
                
                # non-stream mode
                elif not stream:
                    response = await model.chat(processed_messages, stream=False, **kwargs) # type: ignore
                    logger.debug("Non-stream response received")
                    return response
                # Unknown response type
                else:
                    logger.warning(f"Unknown response type.")
                    return None
                
            except Exception as e:
                logger.exception(f"Error generating chat response: {e}")
                raise RuntimeError("Error generating chat response") from e

            
                
        except Exception as e:
            logger.exception(f"Error generating chat response: {e}")
            raise RuntimeError(f"Failed to generate chat response: {e}")