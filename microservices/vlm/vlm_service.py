import os
import sys
import json
from loguru import logger
from typing import Any, AsyncGenerator    
from .model_pool import ModelPool
from .model import BaseVLM


class VLMService:
    """
    统一的VLM服务，用于管理和使用不同的VLM模型
    """
    
    def __init__(self):
        """
        Initialize VLM service // 初始化VLM服务
        """
        self._model_pool = ModelPool()
        self._initialized = False
        
    async def initialize(self):
        """
        Initialize VLM service and model pool // 初始化VLM服务和模型池
        """
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True
            logger.info("VLM service initialized")
        
    async def shutdown(self):
        """
        Shutdown VLM service and all models // 关闭VLM服务和所有模型
        """
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False
            logger.info("VLM service has been shutdown")
    
    async def get_vlm_model(self, model_unique_name: str) -> BaseVLM:
        """
        Get a VLM model by unique name // 获取指定唯一名的VLM模型

        Args:
            model_unique_name: The unique name of the model to get // 要获取的模型ID

        Returns:
            VLM model instance // VLM模型实例

        Raises:
            ValueError: If model_unique_name is not found in database // 如果模型ID在数据库中不存在
            RuntimeError: If model creation fails // 如果模型创建失败
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.load_model(model_unique_name)
    
    async def unload_model(self, model_unique_name: str):
        """
        从模型池中卸载模型

        Args:
            model_unique_name: 要卸载的模型ID
        """
        if self._initialized:
            await self._model_pool.unload_model(model_unique_name)
            logger.info(f"Model {model_unique_name} has been unloaded.")
    
    async def reload_model(self, model_unique_name: str) -> BaseVLM:
        """
        Reload a model from the pool // 重新加载模型

        Args:
            model_unique_name: The unique name of the model to reload // 要重新加载的模型ID

        Returns:
            The reloaded VLM model instance // 重新加载的VLM模型实例
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.reload_model(model_unique_name)
    
    async def inference(self, 
                        model_unique_name: str, 
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
        调用VLM模型进行推理

        Args:
            model_unique_name: 数据库内的模型唯一名
            messages: 消息列表
            stream: 是否开启流式输出，如果是，则输出 AsyncGenerator
            timeout: 超时时间，单位：秒
            max_tokens: 生成的最大 token 数量
            temperature: 模型生成的温度，0~2.0
            top_p: 生成概率 top p 的值，0~1.0
            frequency_penalty: 生成惩罚参数
            presence_penalty: 存在惩罚参数

        Returns:
            If stream is True: AsyncGenerator yielding text chunks
            If stream is False: Dictionary with content and usage stats
        """

        try:
            # Get model from pool
            model = await self.get_vlm_model(model_unique_name)
            
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
                logger.debug(f"Sending messages to model: {model_unique_name}")
                response = await model.inference(messages, stream=stream, **kwargs)
                logger.debug(f"Received response type: {type(response)}")
            except Exception as e:
                logger.error(f"Error generating response: {e}")

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
                        
                        # Return the full response structure for the last chunk
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
            logger.error(f"Error generating chat response: {e}")
            raise RuntimeError(f"Failed to generate chat response: {e}")
