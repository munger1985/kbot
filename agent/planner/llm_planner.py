from datetime import datetime
from typing import Any
from loguru import logger
from utils.clients import AIModelClient
from agent.common import ExecutionPlan, TaskStep
from agent.prompt import default_prompt
from skills import SkillManager
from core.config.settings import get_prompt_config

class LLMPlanner:
    def __init__(self, skill_manager: SkillManager):
        self.model_client = AIModelClient()
        self.skill_manager = skill_manager

    async def generate_plan(
        self, 
        standalone_query: str, 
        model_name: str, 
        intent_type: str | None = None,   # 新增：意图子类
        variables: dict[str, Any] | None = None # 新增：上下文变量
    ) -> ExecutionPlan:
        """根据意图、上下文变量和问题动态生成执行计划"""
        logger.info(f"正在生成动态执行计划. 意图: {intent_type}, 查询: {standalone_query}")
        
        try:
            # 1. 工具裁剪逻辑：根据 intent_type 过滤技能列表
            # 如果是 knowledge_query，只传 RAG 类技能；如果是 data_analysis，只传 SQL 类
            skills_list_str = self.skill_manager.get_skill_list_for_planner(
                category_filter=intent_type 
            )

            # 2. 准备上下文变量摘要
            # 告诉 LLM 当前已经有哪些变量可用（比如上一步生成的 {{sql_data}}）
            var_summary = ", ".join(variables.keys()) if variables else "None"

            # 3. 渲染 Prompt (确保你的 task_planner prompt 模版支持这些新变量)
            final_prompt_content = await default_prompt.generate(
                get_prompt_config().task_planner,
                skills_list=skills_list_str,
                standalone_query=standalone_query,
                intent_type=intent_type or "general",
                existing_variables=var_summary 
            )

            # 4. 获取 LLM 生成结果
            plan_data = await self.model_client.get_llm_json(
                model_name=model_name,
                prompt=[{"role": "system", "content": final_prompt_content}]
            )

            # 5. 构造 ExecutionPlan
            plan: ExecutionPlan = {
                "thought": plan_data.get("thought", "正在按步骤执行任务..."),
                "steps": self._validate_steps(plan_data.get("steps", [])),
                "final_goal": plan_data.get("final_goal", "执行任务并得出结论"),
                "plan_type": "dynamic",
                "workflow_id": None,
                "inputs": {
                    "user_query": standalone_query,
                    "plan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_name": model_name,
                    "intent": intent_type,
                    "context_vars": list(variables.keys()) if variables else []
                }
            }
            
            logger.success(f"动态计划生成成功，意图驱动共 {len(plan['steps'])} 步")
            return plan

        except Exception as e:
            logger.error(f"生成动态计划失败: {str(e)}")
            return self._get_fallback_plan(standalone_query)

    def _validate_steps(self, raw_steps: list[dict]) -> list[TaskStep]:
        """确保符合 TaskStep 结构 (保持不变)"""
        validated = []
        for i, s in enumerate(raw_steps):
            step: TaskStep = {
                "step_id": s.get("step_id", i + 1),
                "skill": s.get("skill", "reasoning"),
                "task_description": s.get("task_description", ""),
                "output_var": s.get("output_var", f"step_{i+1}_result"),
                "condition": s.get("condition")
            }
            validated.append(step)
        return validated

    def _get_fallback_plan(self, query: str) -> ExecutionPlan:
        """兜底方案 (保持不变)"""
        return {
            "thought": "规划引擎异常，降级为单步通用处理。",
            "steps": [{
                "step_id": 1, 
                "skill": "ReasoningSkill", 
                "task_description": query, 
                "output_var": "final_result",
                "condition": None
            }],
            "final_goal": "直接回答用户",
            "plan_type": "dynamic",
            "workflow_id": None,
            "inputs": {"user_query": query, "error_fallback": True}
        }