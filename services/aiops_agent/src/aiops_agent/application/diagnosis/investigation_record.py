"""调查记录的证据校验和确定性投影，引用有效不等于自动证明因果。"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from aiops_agent.contracts.investigation import (
    DiagnosisAnswerDraft, InvestigationClaim, InvestigationGoal,
    InvestigationObservation, InvestigationRecord,
)


def validate_diagnosis(draft, evidence):
    facts = {fact.evidence_ref: fact for fact in evidence}
    for claim in draft.claims:
        supporting = []
        for reference in (*claim.evidence_fields, *claim.counterevidence_fields):
            fact = facts.get(reference.evidence_ref)
            if fact is None:
                raise ValueError("诊断引用不在本轮授权证据集合")
            columns = {str(item.get("name")) for item in fact.columns}
            if reference.field not in columns or reference.row_index >= len(fact.rows):
                raise ValueError("诊断证据行或字段不存在")
            if len(fact.rows[reference.row_index]) != len(fact.columns):
                raise ValueError("证据行与列不一致")
            if reference in claim.evidence_fields:
                supporting.append(fact)
        if claim.level in {"OBSERVED", "SUPPORTED"} and not any(
            fact.trust_level == "SOURCE_VERIFIED" for fact in supporting
        ):
            raise ValueError("人工材料、知识或模型推断不能自动升级为现场观测")
        if claim.level == "OBSERVED" and any(fact.truncated for fact in supporting):
            if not claim.limitations:
                raise ValueError("截断观测必须说明事实适用范围")
        if claim.level == "OBSERVED" and any(fact.trust_level != "SOURCE_VERIFIED" for fact in supporting):
            raise ValueError("直接事实不能混合人工材料并升级其可信级别")


def freeze_investigation(*, draft, evidence, context, gaps=()):
    validate_diagnosis(draft, evidence)
    answer = dict(context.plan_snapshot.get("answer_context") or {})
    frame = dict(answer.get("task_frame") or {})
    return InvestigationRecord(
        record_id=str(uuid5(NAMESPACE_URL, f"aiops:{context.run_id}:{context.task_id}")),
        target_id=str(context.target_id), run_id=str(context.run_id), task_id=str(context.task_id),
        original_request=str(answer.get("question") or ""),
        task_kind=str(frame.get("task_kind") or frame.get("diagnostic_profile") or "DIAGNOSIS"),
        generated_at=datetime.now(timezone.utc).isoformat(),
        observations=tuple(InvestigationObservation(
            evidence_ref=fact.evidence_ref, artifact_id=str(fact.artifact_id),
            tool_id=fact.tool_id, source_id=fact.source_id, captured_at=fact.captured_at,
            measurement_semantics=str(fact.measurement_semantics), trust_level=fact.trust_level,
            row_count=fact.row_count, truncated=fact.truncated,
        ) for fact in evidence),
        claims=draft.claims, recommendations=draft.recommendations,
        goal_assessments=(InvestigationGoal(
            goal_id="answer",
            status="SATISFIED" if draft.goal_satisfied else "PARTIAL",
            claim_refs=tuple(item.claim_id for item in draft.claims if item.revision_state == "ACTIVE"),
            blocking_gaps=tuple(dict.fromkeys(str(gap.code) for gap in gaps)) if not draft.goal_satisfied else (),
        ),),
        unresolved_questions=draft.unresolved_questions,
        stop_reason="GOAL_ANSWERED" if draft.goal_satisfied else "PARTIAL_ANSWER",
    )


def sql_investigation_draft(draft, subject):
    """保留 SQL 专项校验的输出，但不从自由正文猜测原因类型。"""
    return DiagnosisAnswerDraft(
        claims=tuple(InvestigationClaim(
            claim_id=item.conclusion_id,
            kind="FACT" if item.level == "OBSERVED" else "INTERPRETATION",
            subject=subject if item.scope == "SUBJECT" else "数据库背景（不可直接归因到指定 SQL）",
            statement=item.statement, level=item.level, applicable_time=item.applicable_time,
            evidence_fields=item.evidence_fields, reasoning=item.reasoning,
            alternatives=item.alternatives,
            limitations=item.limitations or (("未确认原因，须继续核验。",) if item.level != "OBSERVED" else ()),
        ) for item in draft.conclusions),
        recommendations=draft.recommendations,
        unresolved_questions=draft.unresolved_questions, goal_satisfied=draft.goal_satisfied,
    )


def render_investigation(record):
    labels = {"OBSERVED": "已确认", "SUPPORTED": "有证据支持", "HYPOTHESIS": "待验证", "UNKNOWN": "无法判断"}
    analysis, solutions, refs = [], [], []
    for claim in record.claims:
        if claim.revision_state != "ACTIVE":
            continue
        analysis.append(f"{labels[claim.level]}：{claim.statement}\n\n{claim.reasoning}\n\n适用对象与时间：{claim.subject}；{claim.applicable_time}")
        if claim.alternatives:
            analysis.append("其他解释：" + "；".join(claim.alternatives))
        if claim.limitations:
            analysis.append("判断边界：" + "；".join(claim.limitations))
        refs.extend(item.evidence_ref for item in (*claim.evidence_fields, *claim.counterevidence_fields))
    for action in record.recommendations:
        parts = [action.action, "适用条件：" + "；".join(action.preconditions),
                 "风险：" + "；".join(action.risks), "验证：" + "；".join(action.validation),
                 "停止或回退：" + action.rollback_or_stop]
        if action.requires_approval:
            parts.append("待确认与审批建议，本轮未执行。")
        if action.draft_sql:
            parts.append("待验证草案（不构成执行授权）：\n\n```sql\n" + action.draft_sql + "\n```")
        solutions.append("\n\n".join(parts))
    if record.unresolved_questions:
        solutions.append("下一步需核实：" + "；".join(record.unresolved_questions))
    return "\n\n".join(analysis), "\n\n".join(solutions), tuple(dict.fromkeys(refs))


def investigation_report_source(record):
    """报告仅投影已冻结记录，不重新调用模型或数据库。"""
    analysis, solution, refs = render_investigation(record)
    active = [claim for claim in record.claims if claim.revision_state == "ACTIVE"]
    supported_causes = [claim for claim in active if claim.kind == "CAUSE" and claim.level == "SUPPORTED"]
    root_grade = ("POSSIBLE" if any(claim.counterevidence_fields for claim in supported_causes) else "PROBABLE") if supported_causes else "INCONCLUSIVE"
    return {
        "status": "READY" if record.stop_reason == "GOAL_ANSWERED" else "PARTIAL",
        "root_cause": {"effective_level": root_grade},
        "diagnosis_rationale": analysis,
        "facts": tuple({
            "kind": claim.kind, "claim_id": claim.claim_id,
            "record_id": record.record_id, "record_revision": record.revision, "source_run_id": record.run_id,
            "summary": claim.statement, "subject": claim.subject,
            "applicable_time": claim.applicable_time, "level": claim.level,
            "evidence_refs": list(dict.fromkeys(item.evidence_ref for item in claim.evidence_fields)),
        } for claim in active),
        "gaps": tuple({"detail": item} for item in record.unresolved_questions),
        "solution": {"long_term_remediations": (solution,) if solution else ()},
        "evidence_refs": refs,
        "investigation_record": record.model_dump(mode="json"),
    }
