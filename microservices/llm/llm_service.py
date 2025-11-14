import json
from typing import Any
from loguru import logger
from .model_pool import LLMModelPool
from .model import BaseLLM
from core.dictionary import LLMProvider


class LLMService:
    """LLM服务类"""

    def __init__(self) -> None:
        """初始化LLM服务。"""
        self._model_pool = LLMModelPool()
        self._initialized = False

    async def initialize(self):
        """初始化LLM服务和所有模型池。 """
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True
            logger.info("LLM服务初始化完成")
        
    async def shutdown(self):
        """关闭LLM服务和所有模型池。"""
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False
            logger.info("LLM服务已关闭")
            
    async def get_llm_model(self, model_id: int) -> BaseLLM:
        """获取指定唯一名称的嵌入模型实例。"""
        if not self._initialized:
            await self.initialize()

        return await self._model_pool.load_model(model_id)


    async def chat(
        self,
        model_id: int,
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
        """生成聊天响应，支持MCP工具调用

        Args:
            model_id: 模型唯一标识符
            messages: 消息列表或单个提示字符串
            stream: 是否流式传输响应
            timeout: 超时时间（秒）
            max_tokens: 要生成的最大令牌数
            temperature: 采样温度
            top_p: Top-p采样参数
            frequency_penalty: 频率惩罚
            presence_penalty: 存在惩罚
            tools: MCP工具列表，支持工具调用功能
            tool_choice: 工具选择策略，可选值为"auto"或"none"

        Returns:
            如果stream为True: 异步生成器，产生文本块
            如果stream为False: 包含内容和使用统计信息的字典
        """
        try:
            model = await self.get_llm_model(model_id)
        except Exception as e:
            raise RuntimeError(f"获取模型 {self._model_pool._model_names.get(model_id, str(model_id))} 失败: {e}")
        
        # 准备参数
        kwargs = {
            "timeout": timeout,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }

        # 添加工具调用参数（如果提供）
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        # 从模型获取响应
        try:
            # 确保消息格式正确
            processed_messages = []
            if isinstance(messages, str):
                processed_messages = [{"role": "user", "content": messages}]
            else:
                for msg in messages:
                    if isinstance(msg, dict):
                        processed_messages.append(msg)
                    else:
                        # 如果需要，将消息对象转换为字典
                        processed_messages.append({
                            "role": getattr(msg, "role", "user"),
                            "content": getattr(msg, "content", "")
                        })
            
            logger.debug(f"发送消息到模型: {processed_messages}")
            if tools:
                logger.debug(f"工具调用配置 - 工具数量: {len(tools)}, 工具选择: {tool_choice}")

            # OpenAI流模式
            if stream and model.provider == LLMProvider.OPENAI.value:
                response = await model.chat(processed_messages, stream=True, **kwargs)
                logger.debug("收到OpenAI流式响应")
                async def generate_openai_stream():
                    try:
                        async for chunk in response: # type: ignore
                                yield chunk
                            
                    except Exception as e:
                        logger.exception(f"OpenAI流式响应错误: {e}")
                        raise
                        
                return generate_openai_stream()
            
            # OCI流模式
            elif stream and model.provider == LLMProvider.OCI.value:
                response = await model.chat(processed_messages, stream=True, **kwargs)
                logger.debug("收到非OpenAI流式响应")
                async def generate_oci_stream():
                    try:
                        for event in response.data.events(): # type: ignore
                            output =  json.loads(event.data)
                            yield output
                    except Exception as e:
                        logger.exception(f"流式响应错误: {e}")
                        raise   

                return generate_oci_stream()
            
            # 非流模式
            elif not stream:
                response = await model.chat(processed_messages, stream=False, **kwargs) # type: ignore
                logger.debug("收到非流式响应")
                return response
            # 未知响应类型
            else:
                logger.warning(f"未知的响应类型")
                return None
            
        except Exception as e:
            logger.exception(f"生成聊天响应时出错: {e}")
            logger.error(f"详细错误上下文 - 模型: {model_id}, 消息: {messages}, 流式: {stream}, 工具: {len(tools) if tools else 0}")
            raise RuntimeError(f"生成聊天响应失败: {e}. 上下文: 模型 {model_id}, 消息：{messages}, 流式：{stream}")


    async def warmup(self):
        """
        预热池中的所有模型
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
    
    def get_provider(self, model_id: int) -> str | None:
        """获取指定模型的提供者。"""
        if not self._initialized:
            raise RuntimeError("LLM服务未初始化")
        
        return self._model_pool.get_provider_in_pool(model_id)