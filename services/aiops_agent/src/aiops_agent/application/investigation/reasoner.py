"""先理解用户材料，再形成任务框架和调查计划。"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from aiops_agent.ports.model import AIOpsModelPort, StructuredModelResult
from platform_core.contracts.aiops import InvestigationPlanningOutput


_PROMPT = """
你是一名资深 Oracle DBA，也是调查规划者。先识别用户实际提供了什么材料，再判断用户
希望你完成什么，最后决定是否需要调用工具。不要把问题强行归入固定意图或固定回答模板。

输入材料可能同时包含问题、Oracle Alert Log、ORA 错误、SQL 输出、命令输出、监控快照、
配置和普通文字。必须逐项识别并抽取关键事实。用户粘贴的日志或查询结果本身就是证据，
即使数据库或监控源离线，也必须基于这些证据继续调查。

task_frame 要列出一个或多个目标，并明确问题、已知事实、未知项、约束和完成标准。plan
采用最小充分调查：只有
确实能区分假设或补齐回答所需事实时才安排工具。工具只能从 available_tools 中选择；
Playbook 只是经验参考，可选，不是能力白名单。没有合适 Playbook 不能阻止分析。
调用工具时必须完整遵循 available_tools 中的 input 和 policy；input 是完整参数契约，
必须遵循其中的类型、必填、默认值、数值范围、长度和枚举约束。对于
db.oracle.readonly_query，policy.allowed_functions 是完整函数白名单，SQL 不得使用
清单外的同义函数、系统函数或自定义函数；应优先显式列出最小必要列，若确有诊断必要，
宽泛投影或敏感系统列会进入用户审批，不能因此删除必要取证动作；固定目录工具能够回答时
优先使用固定工具。

可用工具无法取得某项证据时，应判断现有证据能否部分或完整回答；只有缺失信息会实质
改变结论且系统无法自动获取时，才准备向用户提出具体补证请求。不要虚构工具结果。
""".strip()

_POLICY_REPAIR_PROMPT = """
你是一名资深 Oracle DBA 调查规划者。上一份调查计划中的工具输入或动态只读 SQL 未通过
确定性安全策略。请依据 validation_error 修正计划，再返回完整的
InvestigationPlanningOutput。

必须保持原问题、任务框架和 revision_no 不变，只修正不合规的调查动作。工具只能从
available_tools 中选择，且每项工具的 input 是完整参数契约，必须遵循类型、必填、默认值、
数值范围、长度和枚举约束。对于 db.oracle.readonly_query，policy.allowed_functions 是完整
函数白名单，SQL 不得使用清单外函数；同时必须遵循该工具的全部 input 和 policy 约束。
宽泛投影或敏感系统列属于需要审批的有效调查动作，不要仅因它们需要审批而删除动作。
如果固定目录工具能够取得同类证据，应改用固定工具。不要虚构工具结果或声称工具已执行。
""".strip()

_REPLAN_PROMPT = """
你是一名资深Oracle DBA调查者。首轮调查已经完成，但证据评估表明仍存在会实质影响结论的
缺口。请基于原始材料、原任务框架、上一版计划和真实评估结果形成下一版调查计划。

只选择能够补齐剩余未知项的最小原子工具集合。不要无参数变化地重复已经执行过的动作；
不要把不可重试的权限、配置或授权缺口再次安排给系统。若可用工具已经无法取得关键证据，
应返回空动作并允许系统依据现有证据回答或向用户提出明确补证请求。Playbook仅提供经验，
不是能力白名单。不得虚构工具、证据或执行结果。
""".strip()


class InvestigationPlanValidationError(ValueError):
    """模型计划引用未知或越界能力。"""

    code = "AIOPS_INVESTIGATION_PLAN_INVALID"


class InvestigationReasoner:
    """使用结构化模型完成材料理解、任务建模和首轮调查规划。"""

    def __init__(self, model: AIOpsModelPort) -> None:
        self._model = model
        self._prompt_ref = {
            "prompt_id": "aiops.investigation-planner",
            "prompt_version": "1",
            "prompt_sha256": hashlib.sha256(_PROMPT.encode("utf-8")).hexdigest(),
            "content": _PROMPT,
        }

    async def plan(
        self,
        *,
        content: tuple[dict[str, Any], ...],
        conversation_context: tuple[str, ...],
        available_tools: tuple[dict[str, Any], ...],
        available_playbooks: tuple[dict[str, Any], ...],
        model_snapshot: dict[str, Any],
        deadline: datetime | None,
        idempotency_key: str,
        source_run_evidence: dict[str, Any] | None = None,
    ) -> StructuredModelResult:
        result = await self._model.generate_structured(
            purpose="aiops.investigation-plan",
            output_model=InvestigationPlanningOutput,
            model_snapshot=model_snapshot,
            prompt_ref=self._prompt_ref,
            input_payload={
                "content": list(content),
                "recent_context": list(conversation_context[-8:]),
                "source_run_evidence": source_run_evidence,
                "available_tools": list(available_tools),
                "available_playbooks": list(available_playbooks),
            },
            deadline=deadline,
            idempotency_key=idempotency_key,
        )
        output = InvestigationPlanningOutput.model_validate(result.output)
        known_tools = {str(item["tool_id"]) for item in available_tools}
        unknown = tuple(
            action.tool_id
            for action in output.plan.actions
            if action.tool_id not in known_tools
        )
        if unknown:
            raise InvestigationPlanValidationError(
                f"调查计划引用了未注册工具：{', '.join(unknown)}"
            )
        return StructuredModelResult(output=output, receipt=result.receipt)

    async def repair_policy_invalid_plan(
        self,
        *,
        content: tuple[dict[str, Any], ...],
        conversation_context: tuple[str, ...],
        source_run_evidence: dict[str, Any] | None,
        invalid_output: InvestigationPlanningOutput,
        validation_error: str,
        available_tools: tuple[dict[str, Any], ...],
        available_playbooks: tuple[dict[str, Any], ...],
        model_snapshot: dict[str, Any],
        deadline: datetime | None,
        idempotency_key: str,
    ) -> StructuredModelResult:
        """携带确定性策略反馈，修正尚未执行的调查计划。"""
        prompt_ref = {
            "prompt_id": "aiops.investigation-policy-repair",
            "prompt_version": "1",
            "prompt_sha256": hashlib.sha256(
                _POLICY_REPAIR_PROMPT.encode("utf-8")
            ).hexdigest(),
            "content": _POLICY_REPAIR_PROMPT,
        }
        result = await self._model.generate_structured(
            purpose="aiops.investigation-policy-repair",
            output_model=InvestigationPlanningOutput,
            model_snapshot=model_snapshot,
            prompt_ref=prompt_ref,
            input_payload={
                "content": list(content),
                "recent_context": list(conversation_context[-8:]),
                "source_run_evidence": source_run_evidence,
                "validation_error": validation_error,
                "rejected_output": invalid_output.model_dump(mode="json"),
                "available_tools": list(available_tools),
                "available_playbooks": list(available_playbooks),
            },
            deadline=deadline,
            idempotency_key=idempotency_key,
        )
        output = InvestigationPlanningOutput.model_validate(result.output)
        if output.input_envelope != invalid_output.input_envelope:
            raise InvestigationPlanValidationError(
                "策略修正规划不得改变输入材料理解"
            )
        if output.task_frame != invalid_output.task_frame:
            raise InvestigationPlanValidationError(
                "策略修正规划不得改变任务框架"
            )
        if output.suggested_playbook_ids != invalid_output.suggested_playbook_ids:
            raise InvestigationPlanValidationError(
                "策略修正规划不得改变建议 Playbook"
            )
        output_plan_context = output.plan.model_copy(update={"actions": ()})
        invalid_plan_context = invalid_output.plan.model_copy(
            update={"actions": ()}
        )
        if output_plan_context != invalid_plan_context:
            raise InvestigationPlanValidationError(
                "策略修正规划只能修改调查动作"
            )
        known_tools = {str(item["tool_id"]) for item in available_tools}
        unknown = tuple(
            action.tool_id
            for action in output.plan.actions
            if action.tool_id not in known_tools
        )
        if unknown:
            raise InvestigationPlanValidationError(
                f"策略修正规划引用了未注册工具：{', '.join(unknown)}"
            )
        return StructuredModelResult(output=output, receipt=result.receipt)

    async def replan(
        self,
        *,
        content: tuple[dict[str, Any], ...],
        conversation_context: tuple[str, ...],
        source_run_evidence: dict[str, Any] | None,
        task_frame: dict[str, Any],
        prior_plan: dict[str, Any],
        assessment: dict[str, Any],
        available_tools: tuple[dict[str, Any], ...],
        available_playbooks: tuple[dict[str, Any], ...],
        model_snapshot: dict[str, Any],
        deadline: datetime | None,
        idempotency_key: str,
        revision_no: int,
    ) -> StructuredModelResult:
        """根据真实Evidence Assessment生成下一轮最小调查计划。"""
        prompt_ref = {
            "prompt_id": "aiops.investigation-replanner",
            "prompt_version": "1",
            "prompt_sha256": hashlib.sha256(
                _REPLAN_PROMPT.encode("utf-8")
            ).hexdigest(),
            "content": _REPLAN_PROMPT,
        }
        result = await self._model.generate_structured(
            purpose="aiops.investigation-replan",
            output_model=InvestigationPlanningOutput,
            model_snapshot=model_snapshot,
            prompt_ref=prompt_ref,
            input_payload={
                "content": list(content),
                "recent_context": list(conversation_context[-8:]),
                "source_run_evidence": source_run_evidence,
                "task_frame": task_frame,
                "prior_plan": prior_plan,
                "assessment": assessment,
                "available_tools": list(available_tools),
                "available_playbooks": list(available_playbooks),
                "required_revision_no": revision_no,
            },
            deadline=deadline,
            idempotency_key=idempotency_key,
        )
        output = InvestigationPlanningOutput.model_validate(result.output)
        if output.plan.revision_no != revision_no:
            raise InvestigationPlanValidationError(
                f"重规划版本必须为 {revision_no}"
            )
        known_tools = {str(item["tool_id"]) for item in available_tools}
        unknown = tuple(
            action.tool_id
            for action in output.plan.actions
            if action.tool_id not in known_tools
        )
        if unknown:
            raise InvestigationPlanValidationError(
                f"重规划引用了未注册工具：{', '.join(unknown)}"
            )
        prior_calls = {
            (
                str(action.get("tool_id")),
                canonical_input(dict(action.get("input") or {})),
            )
            for action in prior_plan.get("actions", ())
        }
        repeated = tuple(
            action.action_id
            for action in output.plan.actions
            if (action.tool_id, canonical_input(dict(action.input)))
            in prior_calls
        )
        if repeated:
            raise InvestigationPlanValidationError(
                "重规划不得原样重复已执行动作："
                + ", ".join(repeated)
            )
        return StructuredModelResult(output=output, receipt=result.receipt)


def canonical_input(value: dict[str, Any]) -> str:
    """稳定比较Tool输入，避免无进展重复调用。"""
    import json

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
