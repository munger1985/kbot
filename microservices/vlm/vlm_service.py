import json
from loguru import logger
from typing import Any, AsyncGenerator    
from .model_pool import VLMModelPool
from .model import BaseVLM


class VLMService:
    """
    统一的VLM服务，用于管理和使用不同的VLM模型
    """
    
    def __init__(self):
        """
        初始化VLM服务
        """
        self._model_pool = VLMModelPool()
        self._initialized = False
        
    async def initialize(self):
        """初始化VLM服务和模型池。"""
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True
            logger.info("VLM 服务已初始化")
        
    async def shutdown(self):
        """关闭VLM服务和所有模型。"""
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False
            logger.info("VLM 服务已关闭")
    
    async def get_vlm_model(self, model_id: int) -> BaseVLM:
        """获取指定唯一名的VLM模型。"""
        if not self._initialized:
            await self.initialize()

        return await self._model_pool.load_model(model_id)
    
    async def inference(self, 
                        model_id: int, 
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

        参数:
            model_id: 模型唯一标识符
            messages: 消息列表
            stream: 是否开启流式输出，如果是，则输出 AsyncGenerator
            timeout: 超时时间，单位：秒
            max_tokens: 生成的最大 token 数量
            temperature: 模型生成的温度，0~2.0
            top_p: 生成概率 top p 的值，0~1.0
            frequency_penalty: 生成惩罚参数
            presence_penalty: 存在惩罚参数

        返回:
            如果 stream 为 True: 生成文本块的异步生成器
            如果 stream 为 False: 包含内容和使用统计信息的字典
        """

        try:
            # 从池中获取模型
            model = await self.get_vlm_model(model_id)
            
            # 准备参数
            kwargs = {
                "timeout": timeout,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            
            # 从模型获取响应
            try:
                logger.debug(f"向模型发送消息: {self._model_pool._model_names.get(model_id, str(model_id))}")
                response = await model.inference(messages, stream=stream, **kwargs)
                logger.debug(f"收到响应类型: {type(response)}")
            except Exception as e:
                logger.error(f"生成响应时出错: {e}")

            if stream:
                # 流式响应处理
                async def generate_stream():
                    try:
                        content_parts = []
                        last_chunk = None
                        
                        async for chunk in response: # type: ignore
                            logger.debug(f"收到块类型: {type(chunk)}")
                            last_chunk = chunk
                            
                            if not hasattr(chunk, 'choices'):
                                logger.warning("收到无效的块格式")
                                continue
                            
                            if not chunk.choices:
                                logger.warning("收到没有选择的块")
                                continue
                            
                            if not hasattr(chunk.choices[0], 'delta'):
                                logger.warning("收到无效的选择格式")
                                continue
                            
                            delta = chunk.choices[0].delta
                            if delta and hasattr(delta, 'content'):
                                content = delta.content
                                if content is not None:
                                    content_parts.append(str(content))
                                    yield str(content)
                                else:
                                    logger.debug("收到内容为空的 delta")
                            else:
                                logger.debug("收到没有内容的 delta")
                        
                        # 流结束后，检查使用数据
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
                        
                        # 返回最后一个块的完整响应结构
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
                        logger.error(f"流式响应处理错误: {e}")
                        raise
                        
                return generate_stream()
            else:
                # 非流式响应处理
                logger.debug(f"收到响应类型: {type(response)}")
                
                if not hasattr(response, 'choices'):
                    raise ValueError("无效的响应格式: 没有 choices 属性")
                
                if not response.choices: # type: ignore
                    raise ValueError("无效的完成格式: 没有可用的选择")
                
                if not hasattr(response.choices[0], 'message'): # type: ignore
                    raise ValueError("无效的选择格式: 没有 message 属性")
                
                message = response.choices[0].message # type: ignore
                if not message or not hasattr(message, 'content'):
                    raise ValueError("无效的消息格式: 没有 content 属性")
                
                if not message.content:
                    raise ValueError("无效的消息格式: 内容为空")
                
                if not hasattr(response, 'usage'):
                    raise ValueError("无效的完成格式: 没有 usage 属性")
                
                if not response.usage: # type: ignore
                    raise ValueError("无效的完成格式: 没有使用数据")
                
                if not all(hasattr(response.usage, attr)  # type: ignore
                          for attr in ['total_tokens', 'prompt_tokens', 'completion_tokens']):
                    raise ValueError("无效的使用格式: 缺少必需的属性")
                
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
            logger.error(f"生成聊天响应时出错: {e}")
            raise RuntimeError(f"生成聊天响应失败: {e}")
        
    async def warmup(self):
        """
        预热模型池中的所有模型
        """
        if not self._initialized:
            await self.initialize()
        
        await self._model_pool.warmup()

    async def load_model(self, model_id: int) -> bool:
        """通过模型唯一标识符加载模型到内存中
        
        Args:
            model_id: 模型唯一标识符
            
        Returns:
            bool: 加载是否成功
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.reload_model(model_id)

        
    async def unload_model(self, model_id: int) -> bool:
        """通过模型唯一标识符卸载模型到内存中。
        
        Args:
            model_id: 模型唯一标识符
            
        Returns:
            bool: 卸载是否成功
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.unload_model(model_id)