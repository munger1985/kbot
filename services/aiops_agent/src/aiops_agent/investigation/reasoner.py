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

task_frame 要明确问题、已知事实、未知项、约束和完成标准。plan 采用最小充分调查：只有
确实能区分假设或补齐回答所需事实时才安排工具。工具只能从 available_tools 中选择；
Playbook 只是经验参考，可选，不是能力白名单。没有合适 Playbook 不能阻止分析。

可用工具无法取得某项证据时，应判断现有证据能否部分或完整回答；只有缺失信息会实质
改变结论且系统无法自动获取时，才准备向用户提出具体补证请求。不要虚构工具结果。
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
    ) -> StructuredModelResult:
        result = await self._model.generate_structured(
            purpose="aiops.investigation-plan",
            output_model=InvestigationPlanningOutput,
            model_snapshot=model_snapshot,
            prompt_ref=self._prompt_ref,
            input_payload={
                "content": list(content),
                "recent_context": list(conversation_context[-8:]),
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
