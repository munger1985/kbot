"""Turn 证据充分性判断与自然回答组合 Handler。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from aiops_agent.contracts.skill_execution import DbaSkillResult
from aiops_agent.contracts.turn_answer import (
    AIOpsTurnResult,
    DbaAnswerDraft,
    DbaSufficiencyAssessment,
    TurnAnswerBlock,
    TurnEvidenceFact,
    TurnEvidenceGap,
)
from platform_core.contracts.aiops import (
    AnswerBlockType,
    MeasurementSemantics,
    SufficiencyStatus,
)

from .handlers import TaskExecutionContext


class DbaEvidenceAssessmentHandler:
    """只按本轮 Skill Artifact 判断证据，不读取历史 Turn。"""

    async def execute(
        self, context: TaskExecutionContext
    ) -> DbaSufficiencyAssessment:
        facts: list[TurnEvidenceFact] = []
        gaps: list[TurnEvidenceGap] = []
        reasons: list[str] = []
        for artifact in context.input_artifacts:
            if artifact.get("schema_version") != "DBA_SKILL_RESULT.v1":
                continue
            result = DbaSkillResult.model_validate(artifact["payload"])
            artifact_id = str(artifact["artifact_id"])
            for outcome in result.tool_outcomes:
                if outcome.observation is not None:
                    observation = outcome.observation
                    if outcome.tool_id != "db.instance.identity":
                        facts.append(
                            TurnEvidenceFact(
                                evidence_ref=(
                                    f"artifact:{artifact_id}#"
                                    f"{outcome.step_id}"
                                ),
                                artifact_id=artifact_id,
                                skill_id=result.skill_id,
                                step_id=outcome.step_id,
                                tool_id=outcome.tool_id,
                                measurement_semantics=(
                                    result.measurement_semantics
                                ),
                                presentation_kind=result.presentation_kind,
                                captured_at=observation.captured_at.isoformat(),
                                columns=tuple(
                                    column.model_dump(mode="json")
                                    for column in observation.columns
                                ),
                                rows=observation.rows,
                                row_count=observation.row_count,
                                truncated=observation.truncated,
                                warnings=observation.warnings,
                            )
                        )
                if outcome.gap is not None:
                    gaps.append(
                        TurnEvidenceGap(
                            skill_id=result.skill_id,
                            step_id=outcome.step_id,
                            code=outcome.gap.code,
                            detail=outcome.gap.detail,
                            retryable=outcome.gap.retryable,
                        )
                    )

        answer_context = dict(context.plan_snapshot.get("answer_context", {}))
        intent = dict(answer_context.get("intent", {}))
        clarification = intent.get("clarification_question")
        if clarification:
            return DbaSufficiencyAssessment(
                status=SufficiencyStatus.NEEDS_CLARIFICATION,
                evidence=tuple(facts),
                gaps=tuple(gaps),
                reasons=("当前问题存在影响诊断范围的必要歧义",),
                clarification_question=str(clarification),
            )

        time_window = dict(intent.get("time_window") or {})
        requested_window = time_window.get("mode") in {"RECENT", "ABSOLUTE"}
        cumulative_only = bool(facts) and all(
            fact.measurement_semantics
            == MeasurementSemantics.CUMULATIVE_SINCE_LOAD
            for fact in facts
        )
        if requested_window and cumulative_only:
            reasons.append(
                "请求的是时间窗口数据，但当前证据只有实例启动后的累计口径"
            )
        if gaps:
            reasons.append("部分受控取证步骤未能返回可验证结果")

        if not facts:
            status = SufficiencyStatus.NEEDS_EVIDENCE
            reasons.append("当前没有取得能够回答问题的主题证据")
        elif reasons:
            status = SufficiencyStatus.PARTIAL
        else:
            status = SufficiencyStatus.ANSWERABLE
        return DbaSufficiencyAssessment(
            status=status,
            evidence=tuple(facts),
            gaps=tuple(gaps),
            reasons=tuple(reasons),
        )


class DbaAnswerComposeHandler:
    """模型写自然语言，表格和图表由服务端从已验证证据生成。"""

    def __init__(self, *, model_client, prompts) -> None:
        self._model = model_client
        self._prompts = prompts

    async def execute(self, context: TaskExecutionContext) -> AIOpsTurnResult:
        assessment = self._assessment(context.input_artifacts)
        if assessment.status in {
            SufficiencyStatus.NEEDS_CLARIFICATION,
            SufficiencyStatus.NEEDS_EVIDENCE,
            SufficiencyStatus.CAPABILITY_UNAVAILABLE,
            SufficiencyStatus.UNSAFE,
        }:
            return self._waiting_result(assessment)

        answer_context = dict(context.plan_snapshot.get("answer_context", {}))
        prompt = self._prompts.resolve("answer_compose", "1")
        result = await self._model.generate_structured(
            purpose="aiops.dba-answer-compose",
            output_model=DbaAnswerDraft,
            model_snapshot=dict(answer_context["model"]),
            prompt_ref={**prompt.ref(), "content": prompt.content},
            input_payload={
                "question": str(answer_context.get("question", "")),
                "intent": dict(answer_context.get("intent", {})),
                "sufficiency": assessment.model_dump(mode="json"),
            },
            deadline=self._deadline(context.deadline_at),
            idempotency_key=f"turn:{context.run_id}:answer:{context.attempt}",
        )
        draft = DbaAnswerDraft.model_validate(result.output)
        allowed_refs = {item.evidence_ref for item in assessment.evidence}
        if not set(draft.evidence_refs) <= allowed_refs:
            raise ValueError("回答引用了本轮批准证据之外的内容")

        blocks: list[TurnAnswerBlock] = [
            TurnAnswerBlock(
                block_type=AnswerBlockType.MARKDOWN,
                schema_version="AIOPS_MARKDOWN_BLOCK.v1",
                payload={"markdown": draft.markdown},
                evidence_refs=draft.evidence_refs,
            )
        ]
        blocks.extend(self._data_blocks(assessment.evidence))
        status = (
            "COMPLETED"
            if assessment.status == SufficiencyStatus.ANSWERABLE
            else "PARTIAL"
        )
        return AIOpsTurnResult(
            status=status,
            sufficiency_status=assessment.status,
            blocks=tuple(blocks),
            model_receipt=result.receipt.model_dump(mode="json"),
        )

    @staticmethod
    def _assessment(
        artifacts: tuple[dict[str, Any], ...]
    ) -> DbaSufficiencyAssessment:
        matches = [
            item["payload"]
            for item in artifacts
            if item.get("schema_version") == "DBA_SUFFICIENCY.v1"
        ]
        if len(matches) != 1:
            raise ValueError("回答任务必须且只能接收一个充分性 Artifact")
        return DbaSufficiencyAssessment.model_validate(matches[0])

    @staticmethod
    def _waiting_result(
        assessment: DbaSufficiencyAssessment,
    ) -> AIOpsTurnResult:
        if assessment.clarification_question:
            message = assessment.clarification_question
        else:
            detail = "；".join(assessment.reasons) or "当前证据不足"
            message = (
                f"我暂时还不能可靠回答这个问题：{detail}。"
                "请把相关命令或只读查询结果以文字或截图贴到对话中，我会继续判断。"
            )
        return AIOpsTurnResult(
            status="WAITING_USER",
            sufficiency_status=assessment.status,
            blocks=(
                TurnAnswerBlock(
                    block_type=AnswerBlockType.MARKDOWN,
                    schema_version="AIOPS_MARKDOWN_BLOCK.v1",
                    payload={"markdown": message},
                ),
            ),
        )

    @staticmethod
    def _data_blocks(
        evidence: tuple[TurnEvidenceFact, ...],
    ) -> tuple[TurnAnswerBlock, ...]:
        blocks: list[TurnAnswerBlock] = []
        for fact in evidence:
            columns = list(fact.columns)
            if fact.presentation_kind in {"TABLE", "TABLE_AND_CHART"}:
                blocks.append(
                    TurnAnswerBlock(
                        block_type=AnswerBlockType.TABLE,
                        schema_version="AIOPS_TABLE_BLOCK.v1",
                        payload={
                            "title": fact.tool_id,
                            "columns": columns,
                            "rows": [list(row) for row in fact.rows],
                            "row_count": fact.row_count,
                            "truncated": fact.truncated,
                            "captured_at": fact.captured_at,
                            "measurement_semantics": (
                                fact.measurement_semantics
                            ),
                        },
                        evidence_refs=(fact.evidence_ref,),
                    )
                )
            chart = DbaAnswerComposeHandler._chart_payload(fact)
            if (
                chart is not None
                and fact.presentation_kind in {"CHART", "TABLE_AND_CHART"}
            ):
                blocks.append(
                    TurnAnswerBlock(
                        block_type=AnswerBlockType.CHART,
                        schema_version="AIOPS_CHART_BLOCK.v1",
                        payload=chart,
                        evidence_refs=(fact.evidence_ref,),
                    )
                )
        return tuple(blocks)

    @staticmethod
    def _chart_payload(fact: TurnEvidenceFact) -> dict[str, Any] | None:
        dimension = next(
            (
                index
                for index, column in enumerate(fact.columns)
                if column.get("logical_type") in {"STRING", "DATETIME"}
            ),
            None,
        )
        metric = next(
            (
                index
                for index, column in enumerate(fact.columns)
                if column.get("logical_type") in {"INTEGER", "DECIMAL"}
            ),
            None,
        )
        if dimension is None or metric is None or not fact.rows:
            return None
        return {
            "title": fact.tool_id,
            "chart_type": "BAR",
            "category": fact.columns[dimension]["name"],
            "metric": fact.columns[metric]["name"],
            "categories": [row[dimension] for row in fact.rows[:50]],
            "series": [row[metric] for row in fact.rows[:50]],
            "captured_at": fact.captured_at,
            "measurement_semantics": fact.measurement_semantics,
        }

    @staticmethod
    def _deadline(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None
