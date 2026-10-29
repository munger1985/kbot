"""截止时间收敛时只呈现已持久化事实，不再调用模型或创建调查动作。"""

from __future__ import annotations

from pydantic import ValidationError

from aiops_agent.application.diagnosis.sql_evidence_answer import build_sql_evidence_answer, rebind_prior_sql_facts
from aiops_agent.application.diagnosis.evidence_projection import database_evidence_fact, top_sql_window_evidence_fact
from aiops_agent.contracts.tool_execution import DbaToolResult
from aiops_agent.contracts.turn_answer import (
    AIOpsTurnResult,
    DbaSufficiencyAssessment,
    TurnAnswerBlock,
    TurnEvidenceFact,
    TurnEvidenceGap,
)


def deadline_result(*, artifacts, task_frame: dict, database_type: str, projected_ids: set[str], validated_prior_ids: frozenset[str] = frozenset(), validated_assessment_ids: frozenset[str] = frozenset()) -> AIOpsTurnResult | None:
    """仅使用已投影、可追溯的受控结果；没有任何事实时不伪造部分成功。"""
    facts: dict[str, TurnEvidenceFact] = {}
    for artifact in artifacts:
        if str(artifact.artifact_id) not in projected_ids:
            continue
        try:
            if artifact.trust_level != "SOURCE_VERIFIED":
                continue
            if artifact.schema_version == "DBA_TOOL_RESULT.v1":
                source = DbaToolResult.model_validate(artifact.payload_json)
                for outcome in source.tool_outcomes:
                    observation = outcome.observation
                    if observation is None or outcome.status != "SUCCEEDED":
                        continue
                    reference = f"artifact:{artifact.artifact_id}#{outcome.step_id}"
                    facts[reference] = database_evidence_fact(
                        result=source, outcome=outcome, artifact_id=artifact.artifact_id,
                        provenance=getattr(artifact, "provenance_json", None),
                    )
            elif artifact.schema_version == "TOP_SQL_WINDOW_RESULT.v1":
                fact = top_sql_window_evidence_fact(result=artifact.payload_json["result"], artifact_id=artifact.artifact_id)
                if fact is not None:
                    facts[fact.evidence_ref] = fact
            elif artifact.schema_version == "PRIOR_TURN_SQL_EVIDENCE.v1" and str(artifact.artifact_id) in validated_prior_ids:
                for fact in rebind_prior_sql_facts(
                    payload=artifact.payload_json, artifact_id=str(artifact.artifact_id),
                    task_frame=task_frame, database_type=database_type,
                ):
                    facts[fact.evidence_ref] = fact
            elif artifact.schema_version == "DBA_SUFFICIENCY.v1" and str(artifact.artifact_id) in validated_assessment_ids:
                # 监控、计划及未来观察的副本须经正式来源复核，不能只相信副本自述。
                assessment = DbaSufficiencyAssessment.model_validate(artifact.payload_json)
                for fact in assessment.evidence:
                    if fact.trust_level == "SOURCE_VERIFIED" and str(fact.artifact_id) in projected_ids:
                        facts.setdefault(fact.evidence_ref, fact)
        except (ValidationError, ValueError, TypeError, KeyError):
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
