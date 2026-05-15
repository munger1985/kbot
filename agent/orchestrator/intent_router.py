"""意图识别与路由模块。"""

from loguru import logger
from pydantic import BaseModel

from utils.clients import AIModelClient
from core.dictionary import IntentType
from core.config.settings import get_prompt_config
from agent.prompt import default_prompt


class IntentAnalysis(BaseModel):
    intent: IntentType
    reason: str
    confidence: float
    # --- 新增字段 ---
    requires_context: bool = False  # 是否是依赖上下文的追问
    detected_entities: list[str] = [] # 路由阶段顺便提取的关键词/实体

class IntentRouter:
    def __init__(self):
        self.model_client = AIModelClient()

    async def route(self, llm_model_name: str, query: str) -> IntentAnalysis:
        """识别意图并返回多维路由结果"""
        try:
            # 1. 获取配置好的重构版 Prompt
            # 确保 get_prompt_config().intent_router 已经更新为我们刚刚重写的那段
            prompt = await default_prompt.generate(
                get_prompt_config().intent_router, 
                query=query
            )
            
            # 2. 调用模型，温度设为 0 以保证分类稳定性
            data = await self.model_client.get_llm_json(
                model_name=llm_model_name,
                prompt=prompt,
                temperature=0
            )

            # 3. 实例化模型，Pydantic 会自动处理字段映射
            analysis = IntentAnalysis(**data)
            
            # 4. 核心逻辑增强：低置信度修正
            # 如果是业务意图但信心极低（比如 < 0.4），我们可以打个日志，
            # 或者在这里逻辑介入，将其标记为需要反问。
            if analysis.confidence < 0.5:
                logger.warning(f"意图识别置信度较低: {analysis.confidence} | Reason: {analysis.reason}")

            logger.info(
                f"意图路由成功 | Query: {query} | "
                f"Intent: {analysis.intent} | "
                f"Ctx_Required: {analysis.requires_context} | "
                f"Entities: {analysis.detected_entities}"
            )
            
            return analysis
            
        except Exception as e:
            # 降级逻辑：默认转为最稳妥的意图
            logger.error(f"意图路由失败: {e}")
            return IntentAnalysis(
                intent=IntentType.CHITCHAT, 
                reason=f"路由异常降级: {str(e)}", 
                confidence=0.0
            )