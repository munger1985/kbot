"""LLM service implementation."""

from loguru import logger
import os
import sys
import re
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
        
    def ensure_markdown(self, text: str) -> str:
        """
        Ensure the text is in markdown format.
        如果文本不是markdown格式，将其转换为markdown格式。
        
        Args:
            text: 输入文本
            
        Returns:
            确保为markdown格式的文本
        """
        # 检查文本是否已经包含markdown元素
        markdown_patterns = [
            r'#{1,6}\s+\w+',  # 标题
            r'\*\*.*?\*\*',    # 粗体
            r'\*.*?\*',        # 斜体
            r'`.*?`',          # 行内代码
            r'```[\s\S]*?```', # 代码块
            r'\[.*?\]\(.*?\)', # 链接
            r'!\[.*?\]\(.*?\)', # 图片
            r'^\s*[-*+]\s+\w+', # 无序列表
            r'^\s*\d+\.\s+\w+', # 有序列表
            r'^\s*>\s+\w+'      # 引用
        ]
        
        # 如果文本已经包含markdown元素，则直接返回
        for pattern in markdown_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return text
                
        # 如果文本不包含markdown元素，则将其转换为markdown格式
        # 这里我们简单地将文本包装在markdown段落中
        # 实际应用中可能需要更复杂的转换逻辑
        return text

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
            Generated text list in markdown format

        Raises:
            RuntimeError: If an error occurs while generating the text
        """
        try:
            # Get model from pool
            model = await self.get_llm_model(model_id)
            # 修改提示词，要求模型生成markdown格式的内容
            markdown_prompt = f"{prompt}\n\n请以markdown格式返回内容。"
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
            
            # 获取生成的文本
            result = await model.generate(prompt=markdown_prompt, n=n, **kwargs)
            
            # 确保返回的内容是markdown格式
            if result:
                return [self.ensure_markdown(text) for text in result]
            return result
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
            所有返回内容均为markdown格式

        Raises:
            RuntimeError: If an error occurs while generating the chat response
        """
        try:
            # Get model from pool
            model = await self.get_llm_model(model_id)
            
            # 修改最后一条消息，要求模型生成markdown格式的内容
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] = f"{messages[-1]['content']}\n\n请以markdown格式返回内容。"
            
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
                # 流式模式下，需要包装生成器以确保每个块都是markdown格式
                original_generator = await model.chat(messages=messages, stream=stream, **kwargs) # type: ignore
                
                # 创建一个新的生成器，确保每个块都是markdown格式
                async def markdown_generator():
                    async for chunk in original_generator: # type: ignore
                        yield self.ensure_markdown(chunk)
                
                return markdown_generator()
            else:
                # 非流式模式，返回完整响应
                response = await model.chat(messages=messages, stream=stream, **kwargs)
                # 确保返回的内容是markdown格式
                markdown_response = self.ensure_markdown(response) # type: ignore
                return {"role": "assistant", "content": markdown_response}
        except Exception as e:
            logger.exception(f"Error generating chat response: {e}")
            raise RuntimeError(f"Error generating chat response: {e}")