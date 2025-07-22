"""LLM service implementation."""

from loguru import logger
import os
import sys
from typing import Dict, List, Optional, AsyncGenerator

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
            
    async def get_llm_model(self, model_id: int) -> BaseLLM:
        """Retrieve a LLM model by ID. //获取指定ID的LLM模型

        Args:
            model_id: 模型ID

        Returns:
            LLM model instance //LLM模型实例
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.load_model(model_id)

    async def generate(
        self,
        model_id: int,
        prompt: str,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        n: Optional[int] = 1,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None
    ) -> Optional[List[str]]:
        """Generate text from a prompt. //根据提示词生成文本

        Args:
            model_id: Model ID
            prompt: Input prompt
            timeout: Timeout in seconds
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            n: Number of responses to generate
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty

        Returns:
            Generated text list

        Raises:
            RuntimeError: If an error occurs while generating the text
        """
        try:
            # Get model from pool
            model = await self.get_llm_model(model_id)
            # Generate text
            kwargs = {}
            if timeout:
                kwargs["timeout"] = timeout
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if temperature:
                kwargs["temperature"] = temperature
            if top_p:
                kwargs["top_p"] = top_p
            if frequency_penalty:
                kwargs["frequency_penalty"] = frequency_penalty
            if presence_penalty:
                kwargs["presence_penalty"] = presence_penalty
                
            return await model.generate(prompt=prompt, n=n, **kwargs)
        except Exception as e:
            logger.exception(f"Error generating text: {e}")
            raise RuntimeError(f"Error generating text: {e}")

    async def chat(
        self,
        model_id: int,
        messages: List[Dict[str, str]],
        stream: bool = False,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None
    ) -> Optional[AsyncGenerator[str, None] | Dict[str, str]]:
        """Generate a chat response. //生成聊天响应

        Args:
            model_id: Model ID
            messages: List of messages
            stream: Whether to stream the response
            timeout: Timeout in seconds
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty

        Returns:
            If stream is True, returns an async generator of strings, else returns a dictionary with role and content.
            流式模式下返回异步生成器，非流式返回包含角色和内容的字典

        Raises:
            RuntimeError: If an error occurs while generating the chat response
        """
        try:
            # Get model from pool
            model = await self.get_llm_model(model_id)
            # Generate chat response
            kwargs = {}
            if timeout:
                kwargs["timeout"] = timeout
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if temperature:
                kwargs["temperature"] = temperature
            if top_p:
                kwargs["top_p"] = top_p
            if frequency_penalty:
                kwargs["frequency_penalty"] = frequency_penalty
            if presence_penalty:
                kwargs["presence_penalty"] = presence_penalty
                
            if stream:
                # 直接返回流式生成器
                return await model.chat(messages=messages, stream=stream, **kwargs) # type: ignore
            else:
                # 非流式模式，返回完整响应
                response = await model.chat(messages=messages, stream=stream, **kwargs)
                return {"role": "assistant", "content": response} # type: ignore
        except Exception as e:
            logger.exception(f"Error generating chat response: {e}")
            raise RuntimeError(f"Error generating chat response: {e}")