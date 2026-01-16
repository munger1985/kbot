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
            
    async def get_llm_model(self, model_name: str) -> BaseLLM:
        """获取指定唯一名称的嵌入模型实例。"""
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
    ):
        """生成聊天响应，支持MCP工具调用

        Args:
            model_name: 模型技术名称
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
            # 1. 获取模型实例（这会自动确保模型已加载）
            model = await self.get_llm_model(model_name)
            current_provider = model.config.provider  # 统一获取方式
        except Exception as e:
            raise RuntimeError(f"从模型池中获取模型 {model_name} 失败: {e}")
        
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

            if stream:
                # 检查是否属于 OpenAI 协议家族
                openai_compatible_providers = [
                    LLMProvider.CHATGPT.value,
                    LLMProvider.API_QWEN.value,
                    LLMProvider.API_DEEPSEEK.value
                ]

                if current_provider in openai_compatible_providers:
                    response = await model.chat(processed_messages, stream=True, **kwargs)
                    logger.debug(f"收到 OpenAI 兼容流式响应 ({current_provider})")
                    
                    async def generate_openai_stream():
                        try:
                            async for chunk in response: # type: ignore
                                yield chunk
                        except Exception as e:
                            logger.exception(f"OpenAI流式响应错误: {e}")
                            raise
                    return generate_openai_stream()

                elif current_provider == LLMProvider.OCI.value:
                    response = await model.chat(processed_messages, stream=True, **kwargs)
                    logger.debug("收到 OCI 原生流式响应")
                    
                    async def generate_oci_stream():
                        try:
                            # OCI SDK 特有的事件流处理
                            for event in response.data.events():  # type: ignore
                                output = json.loads(event.data)
                                yield output
                        except Exception as e:
                            logger.exception(f"OCI流式响应错误: {e}")
                            raise   
                    return generate_oci_stream()

            # 2. 非流模式 (Non-Stream)
            else:
                response = await model.chat(processed_messages, stream=False, **kwargs)
                logger.debug(f"收到非流式响应 ({current_provider})")
                return response
            
        except Exception as e:
            logger.exception(f"生成聊天响应时出错: {e}")
            logger.error(f"详细错误上下文 - 模型: {model_name}, 消息: {messages}, 流式: {stream}, 工具: {len(tools) if tools else 0}")
            raise RuntimeError(f"生成聊天响应失败: {e}. 上下文: 模型 {model_name}, 消息：{messages}, 流式：{stream}")


    async def warmup(self):
        """
        预热池中的所有模型
        """
        if not self._initialized:
            await self.initialize()
        
        await self._model_pool.warmup()

    async def load_model(self, model_name: str) -> bool:
        """通过模型技术名称加载模型到内存中
        
        Args:
            model_name: 模型技术名称
            
        Returns:
            bool: 加载是否成功
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.reload_model(model_name)

        
    async def unload_model(self, model_name: str) -> bool:
        """通过模型技术名称卸载模型到内存中。
        
        Args:
            model_name: 模型技术名称
            
        Returns:
            bool: 卸载是否成功
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.unload_model(model_name)
    
    async def get_model_instance(self, model_name: str) -> BaseLLM:
        """
        获取模型实例。
        如果模型未加载，load_model 会负责从数据库读取配置并初始化。
        """
        if not self._initialized:
            await self.initialize()
        
        # load_model 在基类中已经实现了：存在则返回，不存在则加载
        instance = await self._model_pool.load_model(model_name)
        if not instance:
            raise RuntimeError(f"模型 {model_name} 加载失败或不存在")
        return instance

    def get_provider(self, model_name: str) -> str | None:
        """
        获取模型的提供商。
        优化点：如果池中没有，尝试去缓存的元数据中查找。
        """
        # 1. 尝试从已加载的模型实例中获取
        model = self._model_pool._models.get(model_name)
        if model:
            return model.config.provider
            
        # 2. 如果模型还没加载，从 model_pool 缓存的元数据中获取
        # 假设 BaseModelPool 在 initialize 时已经 fetch 了所有元数据到 self._model_metadata
        metadata = getattr(self._model_pool, '_model_metadata', {}).get(model_name)
        if metadata:
            return metadata.get("provider")
            
        return None

    async def get_max_tokens_limit(self, model_name: str) -> int:
        """
        获取模型的最大 Token 限制。
        由于需要确保模型配置已加载，这里建议改为异步。
        """
        try:
            model = await self.get_model_instance(model_name)
            return getattr(model.config, "max_tokens", 4096)
        except:
            return 4096

    async def get_model_config(self, model_name: str) -> Any:
        """获取已加载模型的配置对象（异步确保加载）"""
        model = await self.get_model_instance(model_name)
        return model.config