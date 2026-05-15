from typing import AsyncGenerator, Any
from loguru import logger
from skills import BaseSkill
from utils.clients import AIModelClient
from core.dictionary import PacketType
from agent.common import ContextMemory


class ChitChatSkill(BaseSkill):
    """
    轻量级闲聊技能：用于处理非业务类对话。
    直接利用 get_llm_stream_parsed 获取解析后的流式数据。
    """
    
    def __init__(self):
        super().__init__()
        self.name = "ChitChatSkill"
        self.description = "处理通用对话、问候及非 RAG 业务的开放式提问。"
        self.model_client = AIModelClient()

    async def run_stream(
        self, 
        context: ContextMemory,  # 统一使用 ContextMemory
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        全异步流式调用：改写后的问题 -> 解析后的内容包
        """
        # 从 context 中提取参数
        task_input = context["current_execution"] or context["standalone_query"] or context["question"]
        model_name = context["llm_model"]

        messages = [
            {"role": "system", "content": "你是一个智能助手。请根据用户的问题提供简洁、准确的回答。"},
            {"role": "user", "content": task_input}
        ]
        logger.info(f"[{self.name}] 触发闲聊流，使用模型: {model_name}")

        try:
            # 2. 调用已有的解析方法，自动处理 JSON 解析和字段提取
            async for chunk in self.model_client.get_llm_stream_parsed(
                model_name=model_name,
                prompt=messages,
                **kwargs
            ):
                # A. 优先处理推理流（Thought/Reasoning）
                if chunk.reasoning_content:
                    yield {
                        "type": PacketType.THOUGHT,
                        "content": chunk.reasoning_content
                    }
                
                # B. 处理标准答案流内容
                if chunk.content:
                    yield {
                        "type": PacketType.ANSWER,
                        "content": chunk.content
                    }
                            
        except Exception as e:
            logger.error(f"[{self.name}] 运行异常: {str(e)}", exc_info=True)
            yield {"type": PacketType.ERROR, "content": f"对话生成遇到一点问题，请稍后再试。"}