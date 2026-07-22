"""
统一程序化计划校验器 (Plan Validator)

对所有 ExecutionPlan（纯 LLM 生成 / SOP 增强生成）进行零成本程序化校验。
校验分为 FATAL（阻断执行）和 WARNING（记录但放行）两级。

同时计算步骤间的依赖图，供 ExecutionScheduler 进行波次分组。
"""

import re
from dataclasses import dataclass, field
from typing import Callable
from loguru import logger

from agent.common.skill_context import TaskStep
from .skill_io_map import get_skill_io


# ── 系统预置变量（始终可用）──
SYSTEM_VARS: set[str] = {
    "standalone_query",
    "question",
    "user_query",       # alias → standalone_query
    "query",            # alias → standalone_query
    "user_question",    # alias → standalone_query
    "original_question",# alias → question
    "search_keywords",
    "user_id",
    "session_id",
    "agent_id",
    "llm_model",
    "embedding_model",
    "extracted_entities",
    "doc_results",
    "sql_results",
    "graph_results",
}


@dataclass
class ValidationResult:
    """校验结果"""
    passed: bool = True
    errors: list[str] = field(default_factory=list)     # FATAL — 必须修正
    warnings: list[str] = field(default_factory=list)   # WARNING — 记录但放行
    dependency_graph: dict[int, set[int]] = field(default_factory=dict)
    """依赖图: {step_index: {前置依赖的 step_index 集合}}"""


def extract_var_refs(text: str) -> set[str]:
    """
    从文本中提取所有 {{variables.xxx}} 和 {{xxx}} 引用。
    返回变量名集合（不含 variables. 前缀）。
    """
    refs: set[str] = set()

    # 匹配 {{variables.xxx}}, {{variables.xxx.yyy}}
    for m in re.finditer(r"\{\{\s*variables\.(\w+(?:\.\w+)*)\s*\}\}", text):
        # 取第一段（顶层变量名）
        refs.add(m.group(1).split(".")[0])

    # 匹配 {{xxx}} (单大括号也已兼容)
    for m in re.finditer(r"\{\s*(\w+)\s*\}", text):
        name = m.group(1)
        # 排除 JSON 语法中的大括号（简单启发式：前面有引号的不算）
        if name not in ("step_id", "skill", "task_description", "output_var"):
            refs.add(name)

    return refs


class PlanValidator:
    """
    统一程序化计划校验器。

    用法:
        validator = PlanValidator(skill_registry=skill_names_set)
        result = validator.validate(plan_steps, sop_steps=None, initial_vars=None)
        if not result.passed:
            ...  # 将 result.errors 反馈给 LLM 重新生成
    """

    def __init__(self, skill_registry: set[str] | None = None):
        """
        Args:
            skill_registry: 已注册的技能名集合（从 SkillManager 获取）
        """
        self.skill_registry = skill_registry or set()

    def validate(
        self,
        steps: list[TaskStep],
        sop_steps: list[dict] | None = None,
        initial_vars: dict | None = None,
    ) -> ValidationResult:
        """
        执行所有校验规则，返回 ValidationResult。

        Args:
            steps: 执行计划的步骤列表
            sop_steps: SOP 定义的步骤（仅 SOP 增强模式时有值）
            initial_vars: 初始变量（ctx["variables"] 的快照）
        """
        result = ValidationResult()

        # ── FATAL 校验 ──
        self._check_skill_existence(steps, result)
        self._check_empty_task_description(steps, result)
        self._check_duplicate_output_var(steps, result)

        # 构建依赖图（用于变量校验和后续并行调度）
        deps = self._build_dependency_graph(steps, initial_vars or {})
        result.dependency_graph = deps

        self._check_variable_dependencies(steps, deps, initial_vars or {}, result)

        # ── SOP 专属 FATAL ──
        if sop_steps:
            self._check_sop_coverage(steps, sop_steps, result)
            self._check_sop_order(steps, sop_steps, result)

        # ── WARNING 校验 ──
        self._check_has_output_step(steps, result)
        self._check_step_count(steps, result)

        result.passed = len(result.errors) == 0
        return result

    # ═══════════════════════════════════════════════════════════════
    # FATAL 校验
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_skill_name(name: str) -> str:
        """标准化技能名：去大小写、连字符、下划线、-skill 后缀差异"""
        n = name.lower().replace("-", "").replace("_", "")
        if n.endswith("skill"):
            n = n[:-5]  # 去掉 "skill" 后缀，使 "reasoningskill" → "reasoning"
        return n

    def _skill_matches(self, a: str, b: str) -> bool:
        """两个技能名是否指同一个技能（模糊匹配）"""
        return self._normalize_skill_name(a) == self._normalize_skill_name(b)

    def _check_skill_existence(self, steps: list[TaskStep], result: ValidationResult) -> None:
        """每步的 skill 必须在 SkillManager 中已注册"""
        for s in steps:
            skill = s.get("skill", "")
            if skill not in self.skill_registry:
                # 模糊匹配：处理大小写、连字符、下划线、-skill 后缀差异
                if not any(self._skill_matches(skill, reg) for reg in self.skill_registry):
                    available = ", ".join(sorted(self.skill_registry)[:10])
                    result.errors.append(
                        f"步骤{s.get('step_id', '?')}: Skill '{skill}' 未注册。"
                        f"可用技能: {available}{'...' if len(self.skill_registry) > 10 else ''}"
                    )

    def _check_empty_task_description(self, steps: list[TaskStep], result: ValidationResult) -> None:
        """每步必须有非空的 task_description"""
        for s in steps:
            if not s.get("task_description", "").strip():
                result.errors.append(
                    f"步骤{s.get('step_id', '?')}: task_description 为空，无法确定该步骤的执行内容"
                )

    def _check_duplicate_output_var(self, steps: list[TaskStep], result: ValidationResult) -> None:
        """output_var 不能重复"""
        seen: dict[str, int] = {}
        for s in steps:
            v = s.get("output_var", "")
            if not v:
                continue
            if v in seen:
                result.errors.append(
                    f"步骤{s.get('step_id', '?')}: output_var '{v}' 与步骤{seen[v]}重复"
                )
            seen[v] = s.get("step_id", 0)

    def _check_variable_dependencies(
        self,
        steps: list[TaskStep],
        deps: dict[int, set[int]],
        initial_vars: dict,
        result: ValidationResult,
    ) -> None:
        """{{variables.xxx}} 引用的变量必须已定义"""
        available = SYSTEM_VARS | set(initial_vars.keys())

        for i, s in enumerate(steps):
            refs = extract_var_refs(s.get("task_description", ""))
            for ref in refs:
                # 检查该引用是否在依赖图中（即是否有前序步骤产出）
                ref_is_satisfied = ref in available

                if not ref_is_satisfied:
                    # 检查是否由某个前序步骤产出
                    for j in deps.get(i, set()):
                        if steps[j].get("output_var") == ref:
                            ref_is_satisfied = True
                            break

                if not ref_is_satisfied:
                    result.errors.append(
                        f"步骤{s.get('step_id', '?')}: 引用了未定义的变量 "
                        f"'{{{{variables.{ref}}}}}'，请确认该变量已由前序步骤产出或为系统变量"
                    )

            # 当前步骤的 output_var 变为可用
            if s.get("output_var"):
                available.add(s["output_var"])

    # ═══════════════════════════════════════════════════════════════
    # SOP 专属 FATAL
    # ═══════════════════════════════════════════════════════════════

    def _check_sop_coverage(
        self, steps: list[TaskStep], sop_steps: list[dict], result: ValidationResult
    ) -> None:
        """SOP 核心步骤必须全部出现在 plan 中"""
        plan_skills = [s.get("skill", "") for s in steps]
        for sop_s in sop_steps:
            sop_skill = sop_s.get("skill", "")
            if not any(self._skill_matches(sop_skill, ps) for ps in plan_skills):
                result.errors.append(
                    f"SOP 步骤 '{sop_skill}' ({sop_s.get('instruction', '')}) 在计划中缺失"
                )

    def _check_sop_order(
        self, steps: list[TaskStep], sop_steps: list[dict], result: ValidationResult
    ) -> None:
        """SOP 步骤的相对顺序必须保持"""
        plan_skills = [s.get("skill", "") for s in steps]
        sop_skills = [s.get("skill", "") for s in sop_steps]

        # 模糊子序列匹配
        if not _fuzzy_is_subsequence(sop_skills, plan_skills, self._skill_matches):
            result.errors.append(
                f"SOP 步骤顺序被破坏。期望顺序: {sop_skills}，"
                f"实际顺序: {plan_skills}"
            )

    # ═══════════════════════════════════════════════════════════════
    # WARNING 校验
    # ═══════════════════════════════════════════════════════════════

    def _check_has_output_step(self, steps: list[TaskStep], result: ValidationResult) -> None:
        """是否包含总结/回答/可视化等用户可见输出步骤"""
        output_skills = [
            "reasoning-skill", "chit-chat-skill", "echarts-skill", "db-analysis-skill",
        ]
        if not any(
            any(self._skill_matches(s.get("skill", ""), out_sk) for out_sk in output_skills)
            for s in steps
        ):
            result.warnings.append(
                "计划缺少总结/回答/可视化步骤，LLM 可能未生成完整的输出计划"
            )

    def _check_step_count(self, steps: list[TaskStep], result: ValidationResult) -> None:
        """步骤数合理性检查"""
        if len(steps) == 0:
            result.errors.append("计划为空，没有任何执行步骤")
        elif len(steps) > 10:
            result.warnings.append(f"计划包含 {len(steps)} 个步骤，可能过于复杂")

    # ═══════════════════════════════════════════════════════════════
    # 依赖图构建
    # ═══════════════════════════════════════════════════════════════

    def _build_dependency_graph(
        self, steps: list[TaskStep], initial_vars: dict
    ) -> dict[int, set[int]]:
        """
        分析步骤间的依赖关系。

        依赖来源：
        1. 显式: task_description 中的 {{variables.xxx}} 引用前序步骤的 output_var
        2. 隐式: 通过数据缓冲区的读写关系（如 ReasoningSkill 依赖 AskDocSkill 写入的 doc_results）

        Returns:
            {step_index: {前置依赖的 step_index 集合}}
        """
        deps: dict[int, set[int]] = {i: set() for i in range(len(steps))}
        produced_vars: dict[str, int] = {}

        for i, step in enumerate(steps):
            skill = step.get("skill", "")
            io = get_skill_io(skill)

            # 显式依赖：变量引用
            for ref in extract_var_refs(step.get("task_description", "")):
                if ref in produced_vars:
                    deps[i].add(produced_vars[ref])

            # 隐式依赖：缓冲区读写
            for read_buf in io.get("reads", []):
                for j in range(i):
                    prev_io = get_skill_io(steps[j].get("skill", ""))
                    if read_buf in prev_io.get("writes", []):
                        deps[i].add(j)

            # 注册当前步骤的输出变量
            out_var = step.get("output_var")
            if out_var:
                produced_vars[out_var] = i

        return deps


def format_validation_errors(result: ValidationResult) -> str:
    """将校验结果格式化为 LLM 友好的错误描述文本"""
    lines: list[str] = []

    if result.errors:
        lines.append("## ⚠️ 计划校验失败，请修正以下严重错误：\n")
        for i, err in enumerate(result.errors, 1):
            lines.append(f"{i}. [FATAL] {err}")

    if result.warnings:
        lines.append("\n## 💡 改进建议（非强制）：\n")
        for i, warn in enumerate(result.warnings, 1):
            lines.append(f"{i}. [WARNING] {warn}")

    return "\n".join(lines)


def _is_subsequence(sub: list[str], full: list[str]) -> bool:
    """检查 sub 是否为 full 的子序列（保持相对顺序，精确匹配）"""
    it = iter(full)
    return all(item in it for item in sub)


def _fuzzy_is_subsequence(
    sub: list[str], full: list[str], match_fn: Callable[[str, str], bool]
) -> bool:
    """检查 sub 是否为 full 的模糊子序列（使用 match_fn 比较技能名）"""
    fi = 0
    for si in range(len(sub)):
        while fi < len(full) and not match_fn(sub[si], full[fi]):
            fi += 1
        if fi >= len(full):
            return False
        fi += 1
    return True
