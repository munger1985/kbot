"""先理解用户材料，再形成任务框架和调查计划。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from aiops_agent.orchestration.diagnosis import (
    TURN_PROMPT_IDS,
    AIOpsPromptRegistry,
)
from aiops_agent.ports.model import AIOpsModelPort, StructuredModelResult
from platform_core.contracts.aiops import InvestigationPlanningOutput


class InvestigationPlanValidationError(ValueError):
    """模型计划引用未知或越界能力。"""

    code = "AIOPS_INVESTIGATION_PLAN_INVALID"


class InvestigationReasoner:
    """使用结构化模型完成材料理解、任务建模和首轮调查规划。"""

    def __init__(
        self,
        model: AIOpsModelPort,
        prompts: AIOpsPromptRegistry,
    ) -> None:
        self._model = model
        self._prompts = prompts

    async def freeze_prompts(
        self,
        frozen_prompts: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, dict[str, str]]:
        """冻结本 Turn 使用的全部数据库 Prompt 版本。"""
        return await self._prompts.snapshot(
            TURN_PROMPT_IDS,
            frozen_prompts=frozen_prompts,
        )

    async def plan(
        self,
        *,
        content: tuple[dict[str, Any], ...],
        conversation_context: tuple[str, ...],
        target_context: dict[str, Any],
        prompt_snapshot: dict[str, dict[str, str]],
        available_tools: tuple[dict[str, Any], ...],
        available_playbooks: tuple[dict[str, Any], ...],
        model_snapshot: dict[str, Any],
        deadline: datetime | None,
        idempotency_key: str,
        source_run_evidence: dict[str, Any] | None = None,
    ) -> StructuredModelResult:
        prompt = await self._prompts.resolve(
            "investigation_planner",
            frozen_prompts=prompt_snapshot,
        )
        result = await self._model.generate_structured(
            purpose="aiops.investigation-plan",
            output_model=InvestigationPlanningOutput,
            model_snapshot=model_snapshot,
            prompt_ref={**prompt.ref(), "content": prompt.content},
            input_payload={
                "content": list(content),
                "recent_context": list(conversation_context[-8:]),
                "target_context": dict(target_context),
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
        target_context: dict[str, Any],
        prompt_snapshot: dict[str, dict[str, str]],
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
        prompt = await self._prompts.resolve(
            "investigation_policy_repair",
            frozen_prompts=prompt_snapshot,
        )
        result = await self._model.generate_structured(
            purpose="aiops.investigation-policy-repair",
            output_model=InvestigationPlanningOutput,
            model_snapshot=model_snapshot,
            prompt_ref={**prompt.ref(), "content": prompt.content},
            input_payload={
                "content": list(content),
                "recent_context": list(conversation_context[-8:]),
                "target_context": dict(target_context),
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
        target_context: dict[str, Any],
        prompt_snapshot: dict[str, dict[str, str]],
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
        prompt = await self._prompts.resolve(
            "investigation_replanner",
            frozen_prompts=prompt_snapshot,
        )
        result = await self._model.generate_structured(
            purpose="aiops.investigation-replan",
            output_model=InvestigationPlanningOutput,
            model_snapshot=model_snapshot,
            prompt_ref={**prompt.ref(), "content": prompt.content},
            input_payload={
                "content": list(content),
                "recent_context": list(conversation_context[-8:]),
                "target_context": dict(target_context),
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
