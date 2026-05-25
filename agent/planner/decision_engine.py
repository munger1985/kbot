from typing import AsyncGenerator, Any
from loguru import logger

from agent.common import ContextMemory, ExecutionPlan
from agent.planner.llm_planner import LLMPlanner
from skills import SkillManager
from core.dictionary import PacketType
from utils.simulate_stream import simulate_stream


class PlanningEngine:
    """
    计划决策引擎（纯控制平面 Class）：
    负责根据细化意图上下文生成（LLM）或加载（Workflow SOP）执行计划，不参与具体搬砖。
    """

    def __init__(self):
        # 1. 保持无参初始化，延迟加载技能管理器
        skill_manager = SkillManager() 
        # 将 skill_manager 传给 LLMPlanner 用于读取当前系统注册了哪些可用组件的元数据
        self.llm_planner = LLMPlanner(skill_manager)

    async def decide_stream(
        self, 
        context: ContextMemory
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        核心决策决策流：依据上下文环境动态生成蓝图，并流式输出决策思考。
        """
        # --- 1. 提取多维基础上下文 ---
        intent_info = context.get("intent_context", {})
        intent_type = intent_info.get("intent")  # 获取具体的业务细化子意图

        
        # 优先使用改写后的 standalone_query，保证规划的准确性
        query = context.get("standalone_query") or context.get("question")
        model_name = context.get("llm_model", "gpt-4o")

        logger.info(
            f"[PlanningEngine] 开始决策。精细意图: {intent_type} | "
        )

        plan: ExecutionPlan | None = None

        # --- 2. 决策路径：智能自主规划 (LLM Dynamic Planning) ---
        if not plan:
            logger.info(f"激活执行路径：LLM 动态编排引擎 [当前意图子类: {intent_type}]...")
            try:
                # 将整个 intent_context 和 variables 扔给动态规划器
                plan = await self.llm_planner.generate_plan(
                    standalone_query=query,
                    model_name=model_name,
                    intent_type=intent_type,  # 让提示词自动裁剪工具
                    variables=context.get("variables", {})  # 让 Planner 知道上一轮遗留了什么
                )
            except Exception as e:
                logger.error(f"LLM 动态规划路径遭遇不可逆溃败: {str(e)}")
                # 极端兜底：构建一个只带最稳妥兜底手段的极其精简的计划结构
                plan = {
                    "thought": "由于动态规划引擎不可用，基座被迫转入安全兜底模式。",
                    "final_goal": "通过基本回答回复用户",
                    "plan_type": "dynamic",
                    "workflow_id": None,
                    "inputs": {"query": query},
                    "steps": [{
                        "step_id": 999,
                        "skill": "ReasoningSkill",
                        "task_description": query,
                        "output_var": "fallback_output",
                        "condition": None
                    }]
                }

        # --- 3. 统一处理结果并写入控制平面（Class 内部直接回填内存） ---
        context["runtime_plan"] = plan
        
        # --- 5. 向前端/用户输出“思考”过程流 ---
        thought = plan.get("thought", "正在依据规划部署任务流...")
        prefix_thought = f"【current intent: {intent_type}】\n{thought}\n"
        
        async for char in simulate_stream(prefix_thought):
            yield {"type": PacketType.THOUGHT, "content": char}

        logger.success(f"[PlanningEngine] 计划分发完成，最终决策 PlanType: {plan.get('plan_type')}")