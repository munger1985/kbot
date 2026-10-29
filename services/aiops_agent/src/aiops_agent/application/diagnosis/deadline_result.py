"""截止时间收敛时只呈现已持久化事实，不再调用模型或创建调查动作。"""

from __future__ import annotations

from types import SimpleNamespace
from pydantic import ValidationError

from aiops_agent.application.diagnosis.sql_evidence_answer import build_sql_evidence_answer, rebind_prior_sql_facts
from aiops_agent.application.diagnosis.evidence_projection import database_evidence_fact, top_sql_window_evidence_fact
from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle
from aiops_agent.application.diagnosis.sql_review import reusable_checkpoint, evidence_hash
from aiops_agent.application.diagnosis.investigation_record import sql_investigation_draft, render_investigation, validate_diagnosis
from aiops_agent.application.diagnosis.sql_plan_presentation import matching_plan_refs, plan_availability_block, plan_table_block
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
    checkpoints = []
    gaps: dict[tuple[str, str | None, str], TurnEvidenceGap] = {}
    for artifact in artifacts:
        if str(artifact.artifact_id) not in projected_ids:
            continue
        try:
            if artifact.trust_level != "SOURCE_VERIFIED":
                continue
            if artifact.schema_version == "DBA_TOOL_RESULT.v1":
                source = DbaToolResult.model_validate(artifact.payload_json)
                for outcome in source.tool_outcomes:
                    if outcome.gap is not None:
                        gap = TurnEvidenceGap(
                            source_id=source.source_id, step_id=outcome.step_id,
                            code=outcome.gap.code, detail=outcome.gap.detail,
                            retryable=outcome.gap.retryable,
                            evidence_ref=f"artifact:{artifact.artifact_id}#{outcome.step_id}",
                            object_access=outcome.gap.object_access,
                        )
                        gaps[(gap.source_id, gap.step_id, gap.code)] = gap
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
                if assessment.sql_checkpoint is not None and assessment.sql_checkpoint.evidence_hash == evidence_hash(assessment.evidence):
                    checkpoints.append(assessment.sql_checkpoint)
                for gap in assessment.gaps:
                    gaps.setdefault((gap.source_id, gap.step_id, gap.code), gap)
                for fact in assessment.evidence:
                    if fact.trust_level == "SOURCE_VERIFIED" and str(fact.artifact_id) in projected_ids:
                        facts.setdefault(fact.evidence_ref, fact)
        except (ValidationError, ValueError, TypeError, KeyError):
            continue
    if not facts:
        return None
    evidence = tuple(facts.values())
    bundle = build_sql_evidence_bundle(task_frame=task_frame, database_type=database_type, evidence=evidence)
    matched_plans = matching_plan_refs(bundle) if bundle is not None else set()
    summary = build_sql_evidence_answer(task_frame=task_frame, database_type=database_type, evidence=evidence,
        gaps=tuple(gaps.values()), matched_plan_refs=tuple(matched_plans))
    text = "本轮已达到执行时限，停止继续补证。以下仅保留截止前已核验的结果，尚未完成的原因判断不作推断。"
    refs: tuple[str, ...] = ()
    if summary is not None:
        text += "\n\n" + summary.analysis
        refs = summary.evidence_refs
    else:
        text += f"\n\n已保留 {len(evidence)} 项受控数据库观测；具体采集时点、原始行与统计口径见诊断证据。"
    solution = summary.solution if summary is not None else None
    checkpoint_used = False
    for checkpoint in reversed(checkpoints):
        draft = reusable_checkpoint(checkpoint, task_frame=task_frame, database_type=database_type, evidence=evidence, current_only=False)
        if draft is None:
            continue
        try:
            converted = sql_investigation_draft(draft, "当前选定 SQL", bundle["subjects"])
            validate_diagnosis(converted, evidence)
            analysis, solution, refs = render_investigation(SimpleNamespace(
                claims=converted.claims, recommendations=converted.recommendations, unresolved_questions=converted.unresolved_questions,
            ))
        except (ValueError, TypeError, KeyError):
            continue
        text = "本轮已达到执行时限，停止继续补证；保留下列阶段已核验判断和原适用时点，尚未分析的新增证据不视为已完成诊断。\n\n阶段结论：" + draft.summary + "\n\n" + analysis
        checkpoint_used = True
        break
    blocks = [plan_table_block(fact) for fact in evidence if fact.evidence_ref in matched_plans and fact.sql_plan is not None]
    blocks.append(TurnAnswerBlock(
            block_type="ANALYSIS_MARKDOWN", schema_version="aiops.analysis.v1",
            payload={"markdown": text, "budget_exhausted": True, "diagnosis_status": "PARTIAL" if checkpoint_used else "INCOMPLETE"}, evidence_refs=refs,
        ))
    if solution is not None:
        blocks.append(TurnAnswerBlock(block_type="SOLUTION_MARKDOWN", schema_version="AIOPS_SOLUTION_BLOCK.v1",
            payload={"markdown": solution}, evidence_refs=refs))
    if bundle is not None:
        blocks.append(plan_availability_block(database_type=database_type, bundle=bundle))
    return AIOpsTurnResult(
        status="PARTIAL", sufficiency_status="PARTIAL", evidence=evidence,
        blocks=tuple(blocks),
        evidence_gaps=(*gaps.values(), TurnEvidenceGap(
            source_id="execution.budget", step_id="execution.budget", code="RUN_DEADLINE_EXCEEDED",
            detail="全流程预算已用完；已停止后续规划，保留截止前的已核验事实。", retryable=False,
        ),),
    )
