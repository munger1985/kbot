"""LLM service implementation."""

from loguru import logger
import os
import sys
from typing import Dict, List, Optional

# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from microservices.llm.model_pool import ModelPool
from models.llm import LLMProvider, BaseLLM


class LLMService:
    """LLM service class."""

    def __init__(self) -> None:
        """Initialize LLM service."""
        self._model_pool = ModelPool()
        self._initialized = False

    async def initialize(self):
        """
        初始化LLM服务和所有模型池
        """
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True
            logger.info("LLM service initialized")
        
    async def shutdown(self):
        """
        关闭LLM服务和所有模型池
        """
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False
            logger.info("LLM service has been shutdown")
            
    async def get_llm_model(self, model_id: int) -> BaseLLM:
        """获取指定ID的LLM模型

        Args:
            model_id: 模型ID

        Returns:
            LLM模型实例
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.load_model(model_id)

    async def generate(
        self,
        model_id: int,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            model_id: Model ID
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text

        Raises:
            RuntimeError: If an error occurs while generating the text
        """
        try:
            # Get model from pool
            model = await self.get_llm_model(model_id)
            # Generate text
            kwargs = {}
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if temperature:
                kwargs["temperature"] = temperature
                
            return await model.generate(prompt=prompt, **kwargs)
        except Exception as e:
            logger.exception(f"生成文本时出错: {e}")
            raise RuntimeError(f"生成文本时出错: {e}")

    async def chat(
        self,
        model_id: int,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, str]:
        """Generate a chat response.

        Args:
            model_id: Model ID
            messages: List of messages
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Chat response message

        Raises:
            RuntimeError: If an error occurs while generating the chat response
        """
        try:
            # Get model from pool
            model = await self.get_llm_model(model_id)
            # Generate chat response
            kwargs = {}
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if temperature:
                kwargs["temperature"] = temperature
                
            response = await model.chat(messages=messages, **kwargs)
            return {"role": "assistant", "content": response}
        except Exception as e:
            logger.exception(f"生成聊天响应时出错: {e}")
            raise RuntimeError(f"生成聊天响应时出错: {e}")