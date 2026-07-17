from loguru import logger
from datetime import datetime
from typing import Any
from agent.common import ExecutionPlan
from services.basic import WorkflowService
from skills import SkillManager
from .workflow_compiler import WorkflowCompiler


class WorkflowPlanner:
    """
    SOP 工作流规划器。

    职责:
    1. load_sop_context() — 加载 SOP 数据并线性化为 LLM 友好的步骤摘要（主路径）
    2. generate_plan() — 确定性编译 DAG → ExecutionPlan（降级兜底路径）
    """

    def __init__(self):
        self.workflow_service = WorkflowService()
        self.compiler = WorkflowCompiler(SkillManager())

    async def load_sop_context(self, workflow_id: str) -> dict[str, Any]:
        """
        加载 SOP 上下文，用于注入 LLMPlanner 的 prompt。

        Returns:
            {
                "name": str,           # SOP 名称
                "description": str,    # SOP 描述
                "mode": str,           # strict / guided / suggested
                "steps": list[TaskStep],  # 确定性编译的完整步骤（降级用）
                "summary": list[dict],    # LLM 友好的步骤摘要
            }
        """
        logger.info(f"加载 SOP 上下文: {workflow_id}")

        workflow_data = await self.workflow_service.get_workflow(workflow_id)

        sop_name = workflow_data.get("name", "未命名流程")
        sop_description = workflow_data.get("description", "")
        sop_mode = workflow_data.get("mode", "guided")

        # 线性化为 LLM 友好的摘要
        summary = self.compiler.linearize_for_llm(workflow_data)

        # 同时保留确定性编译的完整步骤（降级用）
        full_plan = self.compiler.compile(workflow_data, "")
        fallback_steps = full_plan.get("steps", [])

        logger.info(
            f"SOP [{sop_name}] 上下文就绪: mode={sop_mode}, "
            f"{len(summary)} LLM摘要步骤, {len(fallback_steps)} 降级步骤"
        )

        return {
            "name": sop_name,
            "description": sop_description,
            "mode": sop_mode,
            "steps": fallback_steps,   # 降级用的完整 TaskStep
            "summary": summary,        # LLM 用的轻量摘要
        }

    async def generate_plan(self, workflow_id: str, query: str) -> ExecutionPlan:
        """
        确定性编译 DAG → ExecutionPlan（降级兜底路径）。

        仅在 LLM 规划多次失败时使用。
        """
        logger.info(f"降级: 确定性编译 WorkflowID: {workflow_id}")

        try:
            workflow_data = await self.workflow_service.get_workflow(workflow_id)
            execution_plan = self.compiler.compile(workflow_data, query)

            execution_plan["inputs"] = {
                "user_query": query,
                "plan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "plan_type": "workflow",
                "model_name": None,
                "intent": None,
                "context_vars": [],
                "agent_id": workflow_data.get("agent_id"),
                "workflow_name": workflow_data.get("name"),
                "workflow_id": workflow_id,
            }

            logger.info(f"SOP 【{workflow_data.get('name')}】 确定性编译完毕")
            return execution_plan

        except Exception as e:
            logger.exception(f"Workflow 降级编译异常: {e}")
            return self._generate_error_plan(query, str(e))

    def _generate_error_plan(self, query: str, reason: str) -> ExecutionPlan:
        return {
            "thought": f"无法加载预定义流程，原因：{reason}",
            "steps": [],
            "final_goal": query,
            "plan_type": "error",
            "workflow_id": None,
            "inputs": {
                "user_query": query,
                "plan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "plan_type": "error",
                "model_name": None,
                "intent": None,
                "context_vars": [],
                "agent_id": None,
                "workflow_name": None,
                "workflow_id": None,
            }
        }
