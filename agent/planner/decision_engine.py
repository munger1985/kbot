from datetime import datetime
from typing import AsyncGenerator, Any
from loguru import logger

from agent.common import ContextMemory, ExecutionPlan
from agent.planner.workflow_planner import WorkflowPlanner
from agent.planner.llm_planner import LLMPlanner
from agent.planner.plan_validator import PlanValidator, format_validation_errors
from skills import SkillManager
from core.dictionary import PacketType

# 最大重试次数
MAX_PLAN_RETRIES = 2


class PlanningEngine:
    """
    计划决策引擎 v2（统一入口版）。

    所有业务意图统一走 LLMPlanner 生成计划，SOP 作为约束注入 Prompt。
    生成的计划经过统一程序化校验，失败则反馈错误给 LLM 重新生成。

    降级策略:
    - 校验失败 ≥ MAX_PLAN_RETRIES 次 → 确定性编译 SOP (如有) 或兜底 ReasoningSkill
    """

    def __init__(self):
        self.skill_manager = SkillManager()
        self.workflow_planner = WorkflowPlanner()
        self.llm_planner = LLMPlanner(self.skill_manager)
        self.validator = PlanValidator(
            skill_registry=set(self.skill_manager._skills.keys())
        )

    async def decide_stream(
        self,
        context: ContextMemory
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        统一决策流：LLM 生成计划 → 校验 → 通过/重试/降级。
        """
        # --- 1. 提取上下文 ---
        intent_info = context.get("intent_context", {})
        intent_type = intent_info.get("intent")
        workflow_id = intent_info.get("workflow_id")

        query = context.get("standalone_query") or context.get("question")
        model_name = context.get("llm_model")
        if not model_name:
            raise ValueError("llm_model is required in context but was not set")
        variables = context.get("variables", {})

        logger.info(
            f"[PlanningEngine] 统一决策入口 | 意图: {intent_type} | "
            f"SOP: {bool(workflow_id)}"
        )

        # --- 2. 准备 SOP 上下文（如果有） ---
        sop_steps = None
        sop_summary = None
        sop_name = None
        sop_description = None
        sop_mode = "guided"

        if workflow_id:
            try:
                sop_context = await self.workflow_planner.load_sop_context(workflow_id)
                sop_steps = sop_context["steps"]
                sop_summary = sop_context["summary"]
                sop_name = sop_context["name"]
                sop_description = sop_context["description"]
                sop_mode = sop_context["mode"]
                logger.info(f"SOP [{sop_name}] 已加载, mode={sop_mode}, {len(sop_steps)} 核心步骤")
            except Exception as e:
                logger.warning(f"SOP 加载失败，降级为纯 LLM 规划: {e}")

        # --- 3. 生成计划 + 校验循环 ---
        # 首次 generate，后续 regenerate_with_errors（携带前次校验失败信息）
        plan: ExecutionPlan | None = None
        last_error: str | None = None
        last_plan: ExecutionPlan | None = None

        for attempt in range(MAX_PLAN_RETRIES + 1):
            try:
                if attempt == 0:
                    # 首次：从零生成计划
                    if sop_summary and sop_name:
                        plan = await self.llm_planner.generate_plan_with_sop(
                            standalone_query=query,
                            model_name=model_name,
                            intent_type=intent_type,
                            variables=variables,
                            sop_summary=sop_summary,
                            sop_name=sop_name,
                            sop_description=sop_description,
                            sop_mode=sop_mode,
                        )
                    else:
                        plan = await self.llm_planner.generate_plan(
                            standalone_query=query,
                            model_name=model_name,
                            intent_type=intent_type,
                            variables=variables,
                        )
                else:
                    # 重试：基于前次校验错误重新生成
                    plan = await self.llm_planner.regenerate_with_errors(
                        standalone_query=query,
                        model_name=model_name,
                        intent_type=intent_type,
                        variables=variables,
                        previous_plan=last_plan,
                        validation_errors=last_error,
                        sop_summary=sop_summary,
                        sop_name=sop_name,
                        sop_description=sop_description,
                        sop_mode=sop_mode,
                    )

                # 校验
                result = self.validator.validate(
                    steps=plan.get("steps", []),
                    sop_steps=sop_steps,
                    initial_vars=variables,
                )

                if result.passed:
                    # 校验通过 — 注入 workflow 元数据
                    if workflow_id:
                        plan["plan_type"] = "workflow"
                        plan["workflow_id"] = workflow_id
                        plan["inputs"]["workflow_id"] = workflow_id
                        plan["inputs"]["workflow_name"] = sop_name

                    logger.success(
                        f"[PlanningEngine] 计划校验通过 (尝试 {attempt + 1}/{MAX_PLAN_RETRIES + 1})"
                    )
                    break

                # 校验失败 — 保存错误信息供下一轮 regenerate 使用
                last_error = format_validation_errors(result)
                last_plan = plan
                logger.warning(
                    f"[PlanningEngine] 计划校验失败 (尝试 {attempt + 1}):\n{last_error}"
                )

                if attempt >= MAX_PLAN_RETRIES:
                    # 最后一次尝试也失败 → 降级
                    logger.error(f"[PlanningEngine] 已达最大重试次数，触发降级")
                    plan = self._get_fallback_plan(query, sop_steps)

            except Exception as e:
                logger.error(f"[PlanningEngine] 计划生成异常 (尝试 {attempt + 1}): {e}")
                if attempt >= MAX_PLAN_RETRIES:
                    plan = self._get_fallback_plan(query, sop_steps)

        # --- 4. 写入 ContextMemory ---
        context["runtime_plan"] = plan

        # 将依赖图写入 temp 供后续执行调度使用
        if plan and plan.get("steps"):
            deps = self.validator._build_dependency_graph(
                plan["steps"], variables
            )
            context["temp"]["_dep_graph"] = deps

        # --- 5. 流式输出思考过程 ---
        thought = plan.get("thought", "正在依据规划部署任务流...") if plan else "规划失败，使用兜底方案"
        prefix_thought = f"【当前意图：{intent_type}】\n{thought}\n"

        yield {"type": PacketType.THOUGHT, "content": prefix_thought}

        logger.success(f"[PlanningEngine] 计划分发完成, PlanType: {plan.get('plan_type') if plan else 'fallback'}")

    def _get_fallback_plan(
        self, query: str, sop_steps: list[dict] | None
    ) -> ExecutionPlan:
        """
        降级策略:
        1. 如果有 SOP → 确定性编译
        2. 否则 → ReasoningSkill 单步兜底
        """
        if sop_steps:
            logger.info("降级为 SOP 确定性编译")
            return {
                "thought": "LLM 规划多次失败，降级为 SOP 确定性编译执行。",
                "steps": sop_steps,
                "final_goal": query,
                "plan_type": "workflow",
                "workflow_id": None,
                "inputs": {
                    "user_query": query,
                    "plan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "plan_type": "fallback_workflow",
                    "error_fallback": True,
                }
            }

        logger.info("降级为 ReasoningSkill 单步兜底")
        return {
            "thought": "由于动态规划引擎不可用，基座被迫转入安全兜底模式。",
            "final_goal": "通过基本回答回复用户",
            "plan_type": "dynamic",
            "workflow_id": None,
            "inputs": {
                "user_query": query,
                "plan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "plan_type": "dynamic",
                "model_name": None,
                "intent": None,
                "context_vars": [],
                "agent_id": None,
                "workflow_name": None,
                "workflow_id": None,
            },
            "steps": [{
                "step_id": 999,
                "skill": "ReasoningSkill",
                "task_description": query,
                "output_var": "fallback_output",
                "condition": None
            }]
        }
