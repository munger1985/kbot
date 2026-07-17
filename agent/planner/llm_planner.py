from datetime import datetime
from typing import Any
from loguru import logger
from utils.clients import AIModelClient
from agent.common import ExecutionPlan, TaskStep
from agent.prompt import default_prompt
from skills import SkillManager
from core.config import get_prompt_config
from skills import SkillDomain
from .workflow_compiler import WorkflowCompiler


class LLMPlanner:
    def __init__(self, skill_manager: SkillManager):
        self.model_client = AIModelClient()
        self.skill_manager = skill_manager
        self._compiler = WorkflowCompiler(skill_manager)

    async def generate_plan(
        self,
        standalone_query: str,
        model_name: str,
        intent_type: str | None = None,
        variables: dict[str, Any] | None = None,
        user_language: str = "English"
    ) -> ExecutionPlan:
        """根据意图、上下文变量和问题动态生成执行计划（无 SOP 约束）"""
        logger.info(f"正在生成动态执行计划. 意图: {intent_type}, 查询: {standalone_query}")

        try:
            skills_list_str = self.skill_manager.get_skill_list_for_planner(domain_filter=SkillDomain.BUSINESS)
            var_summary = ", ".join(variables.keys()) if variables else "None"

            final_prompt_content = await default_prompt.generate(
                get_prompt_config().task_planner,
                skills_list=skills_list_str,
                standalone_query=standalone_query,
                intent_type=intent_type or "general",
                existing_variables=var_summary,
                user_language=user_language
            )

            plan_data = await self.model_client.get_llm_json(
                model_name=model_name,
                prompt=[{"role": "system", "content": final_prompt_content}]
            )

            plan = self._build_plan(plan_data, standalone_query, model_name, intent_type, variables)
            logger.success(f"动态计划生成成功，意图驱动共 {len(plan['steps'])} 步")
            return plan

        except Exception as e:
            logger.error(f"生成动态计划失败: {str(e)}")
            return self._get_fallback_plan(standalone_query)

    async def generate_plan_with_sop(
        self,
        standalone_query: str,
        model_name: str,
        intent_type: str | None,
        variables: dict[str, Any] | None,
        sop_summary: list[dict[str, Any]],
        sop_name: str,
        sop_description: str,
        sop_mode: str,
        user_language: str = "English",
    ) -> ExecutionPlan:
        """
        基于 SOP 约束生成增强执行计划。

        SOP 步骤作为 LLM 的强制参考，LLM 需要:
        1. 保留 SOP 的所有核心步骤及其顺序
        2. 将每步的通用 instruction 具象化（注入用户 query 中的实体）
        3. 在末尾补充缺失的分析总结/可视化步骤
        """
        logger.info(f"正在基于 SOP [{sop_name}] 生成增强计划. mode={sop_mode}")

        try:
            skills_list_str = self.skill_manager.get_skill_list_for_planner(domain_filter=SkillDomain.BUSINESS)
            var_summary = ", ".join(variables.keys()) if variables else "None"

            # 构建 SOP 摘要文本
            sop_text = self._format_sop_for_llm(sop_summary, sop_name, sop_description, sop_mode)

            # 构建约束描述
            constraints = self._get_mode_constraints(sop_mode)

            final_prompt_content = await default_prompt.generate(
                get_prompt_config().task_planner,
                skills_list=skills_list_str,
                standalone_query=standalone_query,
                intent_type=intent_type or "general",
                existing_variables=var_summary,
                user_language=user_language
            )

            # 在 prompt 末尾注入 SOP 约束
            sop_section = self._build_sop_section(sop_text, constraints, sop_mode)
            final_prompt_content = final_prompt_content + "\n\n" + sop_section

            plan_data = await self.model_client.get_llm_json(
                model_name=model_name,
                prompt=[{"role": "system", "content": final_prompt_content}]
            )

            plan = self._build_plan(plan_data, standalone_query, model_name, intent_type, variables)
            logger.success(f"SOP 增强计划生成成功，包含 {len(plan['steps'])} 步")
            return plan

        except Exception as e:
            logger.error(f"SOP 增强计划生成失败: {str(e)}")
            return self._get_fallback_plan(standalone_query)

    async def regenerate_with_errors(
        self,
        standalone_query: str,
        model_name: str,
        intent_type: str | None,
        variables: dict[str, Any] | None,
        previous_plan: ExecutionPlan,
        validation_errors: str,
        sop_summary: list[dict[str, Any]] | None = None,
        sop_name: str | None = None,
        sop_description: str | None = None,
        sop_mode: str | None = None,
        user_language: str = "English",
    ) -> ExecutionPlan:
        """
        校验失败后重新生成计划。

        将上次失败的 plan 和校验错误注入 prompt，引导 LLM 修正。
        """
        logger.info(f"校验失败，正在重新生成计划。错误: {validation_errors[:200]}...")

        try:
            skills_list_str = self.skill_manager.get_skill_list_for_planner(domain_filter=SkillDomain.BUSINESS)
            var_summary = ", ".join(variables.keys()) if variables else "None"

            final_prompt_content = await default_prompt.generate(
                get_prompt_config().task_planner,
                skills_list=skills_list_str,
                standalone_query=standalone_query,
                intent_type=intent_type or "general",
                existing_variables=var_summary,
                user_language=user_language
            )

            # 注入 SOP 约束（如果有）
            if sop_summary and sop_name:
                sop_text = self._format_sop_for_llm(sop_summary, sop_name, sop_description or "", sop_mode or "guided")
                constraints = self._get_mode_constraints(sop_mode or "guided")
                sop_section = self._build_sop_section(sop_text, constraints, sop_mode or "guided")
                final_prompt_content = final_prompt_content + "\n\n" + sop_section

            # 追加前次失败信息
            retry_section = (
                "\n\n---\n\n"
                "## ⚠️ 上一次生成的计划校验未通过，请修正后重新输出\n\n"
                f"{validation_errors}\n\n"
                "请修正上述所有 FATAL 错误，重新输出完整的执行计划 JSON。"
            )
            final_prompt_content = final_prompt_content + retry_section

            plan_data = await self.model_client.get_llm_json(
                model_name=model_name,
                prompt=[{"role": "system", "content": final_prompt_content}]
            )

            plan = self._build_plan(plan_data, standalone_query, model_name, intent_type, variables)
            logger.success(f"计划重新生成成功，{len(plan['steps'])} 步")
            return plan

        except Exception as e:
            logger.error(f"计划重新生成失败: {str(e)}")
            return self._get_fallback_plan(standalone_query)

    # ═══════════════════════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════════════════════

    def _build_plan(
        self,
        plan_data: dict,
        standalone_query: str,
        model_name: str,
        intent_type: str | None,
        variables: dict[str, Any] | None,
    ) -> ExecutionPlan:
        """从 LLM 返回的 JSON 构建 ExecutionPlan"""
        plan: ExecutionPlan = {
            "thought": plan_data.get("thought", "正在按步骤执行任务..."),
            "steps": self._validate_steps(plan_data.get("steps", [])),
            "final_goal": plan_data.get("final_goal", "执行任务并得出结论"),
            "plan_type": "dynamic",
            "workflow_id": None,
            "inputs": {
                "user_query": standalone_query,
                "plan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "plan_type": "dynamic",
                "model_name": model_name,
                "intent": intent_type,
                "context_vars": list(variables.keys()) if variables else [],
                "agent_id": None,
                "workflow_name": None,
                "workflow_id": None,
            }
        }
        return plan

    def _validate_steps(self, raw_steps: list[dict]) -> list[TaskStep]:
        """确保符合 TaskStep 结构"""
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
        """兜底方案"""
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
                "error_fallback": True
            }
        }

    def _format_sop_for_llm(
        self,
        sop_summary: list[dict[str, Any]],
        sop_name: str,
        sop_description: str,
        sop_mode: str,
    ) -> str:
        """将 SOP 步骤摘要格式化为 LLM 友好的 Markdown 文本"""
        mode_labels = {
            "strict": "严格遵循 (strict) — 必须严格按照 SOP 步骤执行，不允许增删改",
            "guided": "引导增强 (guided) — 保留 SOP 核心步骤，允许增强描述并补充缺失步骤",
            "suggested": "建议参考 (suggested) — SOP 作为参考，可自由调整",
        }
        mode_desc = mode_labels.get(sop_mode, mode_labels["guided"])

        lines = [
            "## 🎯 匹配到的标准作业程序 (SOP)",
            "",
            f"**名称**: {sop_name}",
            f"**描述**: {sop_description}",
            f"**遵循模式**: {mode_desc}",
            "",
            "### 核心步骤（顺序不可变）",
            "",
            "| 步骤 | 技能 | 说明 | 结果变量 |",
            "|------|------|------|----------|",
        ]

        for step in sop_summary:
            lines.append(
                f"| {step['step_id']} | `{step['skill']}` | {step['instruction']} | {step['output_var']} |"
            )

        return "\n".join(lines)

    def _get_mode_constraints(self, sop_mode: str) -> str:
        """根据模式返回约束描述"""
        if sop_mode == "strict":
            return (
                "- 必须严格按照 SOP 的核心步骤及其顺序执行，不允许增删改任何步骤\n"
                "- 每步的 task_description 必须结合用户问题中的具体实体进行具象化\n"
                "- 不允许添加 SOP 之外的任何步骤"
            )
        elif sop_mode == "suggested":
            return (
                "- SOP 步骤作为参考建议，你可以根据用户实际问题自由调整步骤和顺序\n"
                "- 如果 SOP 步骤与用户问题不相关，可以跳过\n"
                "- 根据实际需要自由补充分析和可视化步骤"
            )
        else:  # guided (default)
            return (
                "- 必须保留 SOP 中的所有核心步骤，且顺序不可变\n"
                "- 每步的 task_description 必须融入用户问题中的具体实体\n"
                "- 可以在末尾追加分析总结、可视化等收尾步骤\n"
                "- 如果用户问题已包含 SOP 某步骤所需的信息，该步骤仍需执行以获取完整上下文"
            )

    def _build_sop_section(self, sop_text: str, constraints: str, sop_mode: str) -> str:
        """构建注入 prompt 的 SOP 约束段落"""
        return (
            f"{sop_text}\n\n"
            f"### 约束\n"
            f"{constraints}\n\n"
            f"### 输出要求\n"
            f"请基于以上 SOP 规范生成执行计划 JSON。SOP 中的 instruction 是通用描述，"
            f"你需要结合用户的具体问题将其改写为精确的 task_description。"
        )
