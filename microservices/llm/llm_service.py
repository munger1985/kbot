"""LLM service implementation."""

from loguru import logger
import os
import sys
import json
from typing import Dict, List, Union, Optional, Any, AsyncGenerator

# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from microservices.llm.model_pool import ModelPool
from models.llm import BaseLLM


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
        messages: Union[List[Dict[str, str]], str],
        stream: bool = False,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Generate a chat response.

        Args:
            model_unique_name: Model ID
            messages: List of messages or single prompt string
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
                response = await model.chat(processed_messages, stream=stream, **kwargs)
                logger.debug(f"Received response type: {type(response)}")
            except Exception as e:
                logger.error(f"Error generating chat response: {e}")

            if stream:
                # Stream response processing
                async def generate_stream():
                    try:
                        content_parts = []
                        last_chunk = None
                        
                        async for chunk in response: # type: ignore
                            logger.debug(f"Received chunk type: {type(chunk)}")
                            last_chunk = chunk
                            
                            if not hasattr(chunk, 'choices'):
                                logger.warning("Received invalid chunk format")
                                continue
                            
                            if not chunk.choices:
                                logger.warning("Received chunk with no choices")
                                continue
                            
                            if not hasattr(chunk.choices[0], 'delta'):
                                logger.warning("Received invalid choice format")
                                continue
                            
                            delta = chunk.choices[0].delta
                            if delta and hasattr(delta, 'content'):
                                content = delta.content
                                if content is not None:
                                    content_parts.append(str(content))
                                    yield str(content)
                                else:
                                    logger.debug("Received delta with null content")
                            else:
                                logger.debug("Received delta with no content")
                        
                        # After stream ends, check for usage data
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
                            
                    except Exception as e:
                        logger.error(f"Error in streaming response: {e}")
                        raise
                        
                return generate_stream()
            else:
                # Non-stream response processing
                logger.debug(f"Received response type: {type(response)}")
                
                if not hasattr(response, 'choices'):
                    raise ValueError("Invalid response format: no choices attribute")
                
                if not response.choices: # type: ignore
                    raise ValueError("Invalid completion format: no choices available")
                
                if not hasattr(response.choices[0], 'message'): # type: ignore
                    raise ValueError("Invalid choice format: no message attribute")
                
                message = response.choices[0].message # type: ignore
                if not message or not hasattr(message, 'content'):
                    raise ValueError("Invalid message format: no content attribute")
                
                if not message.content:
                    raise ValueError("Invalid message format: empty content")
                
                if not hasattr(response, 'usage'):
                    raise ValueError("Invalid completion format: no usage attribute")
                
                if not response.usage: # type: ignore
                    raise ValueError("Invalid completion format: no usage data available")
                
                if not all(hasattr(response.usage, attr)  # type: ignore
                          for attr in ['total_tokens', 'prompt_tokens', 'completion_tokens']):
                    raise ValueError("Invalid usage format: missing required attributes")
                
                return {
                    "content": str(message.content),
                    "usage": {
                        "total_tokens": int(response.usage.total_tokens), # type: ignore
                        "prompt_tokens": int(response.usage.prompt_tokens), # type: ignore
                        "completion_tokens": int(response.usage.completion_tokens) # type: ignore
                    }
                }
                
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            raise RuntimeError(f"Failed to generate chat response: {e}")