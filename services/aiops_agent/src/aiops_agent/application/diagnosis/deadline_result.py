"""截止时间收敛时只呈现已持久化事实，不再调用模型或创建调查动作。"""

from __future__ import annotations

from pydantic import ValidationError

from aiops_agent.application.diagnosis.sql_evidence_answer import build_sql_evidence_answer
from aiops_agent.contracts.tool_execution import DbaToolResult
from aiops_agent.contracts.turn_answer import (
    AIOpsTurnResult,
    DbaSufficiencyAssessment,
    TurnAnswerBlock,
    TurnEvidenceFact,
    TurnEvidenceGap,
)


def deadline_result(*, artifacts, task_frame: dict, database_type: str, projected_ids: set[str]) -> AIOpsTurnResult | None:
    """仅使用已投影、可追溯的受控结果；没有任何事实时不伪造部分成功。"""
    facts: dict[str, TurnEvidenceFact] = {}
    for artifact in artifacts:
        if str(artifact.artifact_id) not in projected_ids:
            continue
        try:
            if artifact.schema_version == "DBA_SUFFICIENCY.v1":
                assessment = DbaSufficiencyAssessment.model_validate(artifact.payload_json)
                for fact in assessment.evidence:
                    if fact.trust_level == "SOURCE_VERIFIED" and str(fact.artifact_id) in projected_ids:
                        facts[fact.evidence_ref] = fact
            elif artifact.schema_version == "DBA_TOOL_RESULT.v1" and artifact.trust_level == "SOURCE_VERIFIED":
                source = DbaToolResult.model_validate(artifact.payload_json)
                for outcome in source.tool_outcomes:
                    observation = outcome.observation
                    if observation is None or outcome.status != "SUCCEEDED":
                        continue
                    reference = f"artifact:{artifact.artifact_id}#{outcome.step_id}"
                    facts[reference] = TurnEvidenceFact(
                        evidence_ref=reference, artifact_id=str(artifact.artifact_id),
                        source_id=source.source_id, step_id=outcome.step_id,
                        tool_id=outcome.tool_id, measurement_semantics=source.measurement_semantics,
                        presentation_kind=source.presentation_kind, captured_at=observation.captured_at.isoformat(),
                        columns=tuple(column.model_dump(mode="json") for column in observation.columns),
                        rows=observation.rows, row_count=observation.row_count,
                        truncated=observation.truncated,
                        warnings=(*observation.warnings, *(
                            (f"本轮按需采样实际完成 {source.sampling.completed_count}/{source.sampling.requested.count} 次；仅代表实际采集时点，不覆盖请求的历史窗口",)
                            if source.sampling is not None else ()
                        )),
                    )
        except (ValidationError, ValueError, TypeError):
            continue
    if not facts:
        return None
    evidence = tuple(facts.values())
    summary = build_sql_evidence_answer(task_frame=task_frame, database_type=database_type, evidence=evidence)
    text = "本轮已达到执行时限，停止继续补证。以下仅保留截止前已核验的结果，尚未完成的原因判断不作推断。"
    refs: tuple[str, ...] = ()
    if summary is not None:
        text += "\n\n" + summary.analysis
        refs = summary.evidence_refs
    else:
        text += f"\n\n已保留 {len(evidence)} 项受控数据库观测；具体采集时点、原始行与统计口径见诊断证据。"
    return AIOpsTurnResult(
        status="PARTIAL", sufficiency_status="PARTIAL", evidence=evidence,
        blocks=(TurnAnswerBlock(
            block_type="ANALYSIS_MARKDOWN", schema_version="aiops.analysis.v1",
            payload={"markdown": text, "budget_exhausted": True}, evidence_refs=refs,
        ),),
        evidence_gaps=(TurnEvidenceGap(
            source_id="execution.budget", step_id="execution.budget", code="RUN_DEADLINE_EXCEEDED",
            detail="全流程预算已用完；已停止后续规划，保留截止前的已核验事实。", retryable=False,
        ),),
    )
