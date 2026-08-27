"""专业 DBA Intent Router 与确定性输出校验。"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from aiops_agent.ports.model import AIOpsModelPort, StructuredModelResult
from platform_core.contracts.aiops.skills import DbaIntentPlan


_INTENT_ROUTER_PROMPT = """
你是专业 DBA 助手的意图路由器。你的职责仅是理解用户本轮问题并输出结构化意图，
不得回答问题、生成 SQL、选择 Tool 或虚构证据。

一级意图只能是 OBSERVE、DIAGNOSE、EXPLAIN、PLAN、CHANGE、VERIFY、INSPECT。
专业领域只能使用输出 Schema 中的枚举。识别用户关心的对象、时间窗口、数量、排序和
展示偏好；不影响安全的歧义使用 DBA 常用默认值。只有 Target、跨库范围、变更对象或
安全流程不明确且确实影响后续执行时，才填写 clarification_question。

OBSERVE 查询事实；DIAGNOSE 解释异常原因；EXPLAIN 解释错误或机制；PLAN 制定方案；
CHANGE 请求产生或执行变更；VERIFY 验证结果；INSPECT 进行综合检查。

subject 必须使用受控语义：数据库当前运行和资源使用的综合概览使用
DATABASE_OVERVIEW；Top SQL 使用 TOP_SQL；活动会话使用 ACTIVE_SESSION；阻塞链使用
BLOCKING_CHAIN；表空间使用 TABLESPACE。用户要求综合查看数据库当前情况、负载、资源
使用或健康概览时，使用 DATABASE_OVERVIEW，不要自行创造近义 subject。
""".strip()


class IntentPlanValidationError(ValueError):
    code = "AIOPS_INTENT_PLAN_INVALID"


class DbaIntentRouter:
    """调用结构化模型，并拒绝不满足业务不变量的意图计划。"""

    def __init__(self, model: AIOpsModelPort) -> None:
        self._model = model
        self._prompt_ref = {
            "prompt_id": "aiops.dba-intent-router",
            "prompt_version": "1",
            "prompt_sha256": hashlib.sha256(
                _INTENT_ROUTER_PROMPT.encode("utf-8")
            ).hexdigest(),
            "content": _INTENT_ROUTER_PROMPT,
        }

    async def route(
        self,
        *,
        question: str,
        conversation_context: tuple[str, ...],
        model_snapshot: dict[str, Any],
        deadline: datetime | None,
        idempotency_key: str,
    ) -> StructuredModelResult:
        result = await self._model.generate_structured(
            purpose="aiops.dba-intent-route",
            output_model=DbaIntentPlan,
            model_snapshot=model_snapshot,
            prompt_ref=self._prompt_ref,
            input_payload={
                "question": question,
                "recent_context": list(conversation_context[-6:]),
            },
            deadline=deadline,
            idempotency_key=idempotency_key,
        )
        plan = DbaIntentPlan.model_validate(result.output)
        self.validate(plan)
        return StructuredModelResult(output=plan, receipt=result.receipt)

    @staticmethod
    def validate(plan: DbaIntentPlan) -> None:
        primary = next(
            item
            for item in plan.candidates
            if item.intent == plan.primary_intent
        )
        if primary.confidence < max(
            item.confidence for item in plan.candidates
        ):
            raise IntentPlanValidationError(
                "主意图必须是置信度最高的候选意图"
            )
        if primary.confidence < 0.55 and not plan.clarification_question:
            raise IntentPlanValidationError(
                "低置信度意图必须给出最小澄清问题"
            )
