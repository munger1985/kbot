"""意图识别与路由模块。"""

from loguru import logger
from pydantic import BaseModel

from utils.clients import AIModelClient
from core.dictionary import IntentType
from core.config import get_prompt_config
from agent.prompt import default_prompt


class IntentAnalysis(BaseModel):
    intent: IntentType
    reason: str
    confidence: float
    requires_context: bool = False
    detected_entities: list[str] = []
    # --- 工作流/SOP 匹配 ---
    workflow_id: str | None = None       # 命中的预定义 SOP 工作流 ID（ES 语义检索匹配）
    workflow_name: str | None = None     # 命中工作流的可读名称

class IntentRouter:
    def __init__(self):
        self.model_client = AIModelClient()
        self._workflow_service = None

    @property
    def workflow_service(self):
        """延迟加载 WorkflowService，避免循环依赖"""
        if self._workflow_service is None:
            from services.basic import WorkflowService
            self._workflow_service = WorkflowService()
        return self._workflow_service

    async def route(self, llm_model_name: str, query: str, agent_id: str | None = None) -> IntentAnalysis:
        """识别意图并返回多维路由结果（含可选的工作流 SOP 匹配）"""
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

            # 5. 工作流 SOP 语义匹配：对复杂业务意图，检索最匹配的预定义工作流
            if (
                agent_id
                and analysis.intent in (
                    IntentType.KNOWLEDGE_QUERY,
                    IntentType.DATA_ANALYSIS,
                    IntentType.TASK_EXECUTION,
                    IntentType.COMPLEX_HYBRID,
                )
            ):
                try:
                    matches = await self.workflow_service.search_workflow(
                        agent_id=agent_id,
                        query=query,
                        top_k=1
                    )
                    if matches and matches[0].get("score", 0) >= 0.7:
                        analysis.workflow_id = matches[0]["workflow_id"]
                        analysis.workflow_name = matches[0]["name"]
                        logger.info(
                            f"🎯 SOP 工作流命中: [{analysis.workflow_name}] "
                            f"(Score: {matches[0]['score']:.3f})"
                        )
                except Exception as wf_err:
                    logger.warning(f"工作流 SOP 检索降级: {wf_err}")

            logger.info(
                f"意图路由成功 | Query: {query} | "
                f"Intent: {analysis.intent} | "
                f"Ctx_Required: {analysis.requires_context} | "
                f"Entities: {analysis.detected_entities} | "
                f"Workflow: {analysis.workflow_id or 'None'}"
            )

            return analysis

        except Exception as e:
            logger.error(f"意图路由失败: {e}")
            return IntentAnalysis(
                intent=IntentType.CHITCHAT,
                reason=f"路由异常降级: {str(e)}",
                confidence=0.0
            )