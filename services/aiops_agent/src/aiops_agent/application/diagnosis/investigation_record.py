"""调查记录的证据校验和确定性投影，引用有效不等于自动证明因果。"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from aiops_agent.contracts.investigation import (
    DiagnosisAnswerDraft, InvestigationClaim, InvestigationGoal,
    InvestigationObservation, InvestigationRecord, InvestigationParent, InvestigationAccessGap,
)


def validate_diagnosis(draft, evidence):
    from aiops_agent.domain.evidence.series_calculation import replay_series_computation
    for fact in evidence:
        names = [str(column.get("name")) for column in fact.columns]
        for trace in fact.series_computations:
            if not trace.input_ref.startswith(f"artifact:{fact.artifact_id}#/observations/"):
                raise ValueError("序列计算来源不属于原始监控 Artifact")
            summary = replay_series_computation(trace)
            if trace.row_index >= len(fact.rows):
                raise ValueError("序列计算引用不存在的摘要行")
            row = fact.rows[trace.row_index]
            identity = {"metric_code": trace.metric_code, "unit": trace.unit,
                "value_kind": trace.value_kind, "window_start": trace.window_start.isoformat(),
                "window_end": trace.window_end.isoformat(),
                "dimensions": ", ".join(f"{key}={value}" for key, value in trace.dimensions) or "-"}
            if len(row) != len(names) or any(key not in names or row[names.index(key)] != value for key, value in identity.items()):
                raise ValueError("序列计算身份与摘要行不一致")
            for key, value in summary.items():
                field = "history_elapsed_days" if key == "elapsed_days" else key
                if field in names and row[names.index(field)] != value:
                    raise ValueError("序列摘要与冻结计算不一致")
    facts = {fact.evidence_ref: fact for fact in evidence}
    for action in draft.recommendations:
        for parameter in action.parameters:
            if parameter.source != "OBSERVATION":
                continue
            ref = parameter.evidence_field
            fact = facts.get(ref.evidence_ref)
            if fact is None or fact.trust_level != "SOURCE_VERIFIED":
                raise ValueError("参数来源不是本轮已核验观测")
            names = [str(column.get("name")) for column in fact.columns]
            if ref.field not in names or ref.row_index >= len(fact.rows):
                raise ValueError("参数来源字段不存在")
            if str(fact.rows[ref.row_index][names.index(ref.field)]) != parameter.value:
                raise ValueError("参数值与原始观测不一致")
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
    parent_data = answer.get("parent_investigation")
    parent = InvestigationParent.model_validate(parent_data) if draft.continues_investigation and parent_data else None
    if draft.continues_investigation and parent is None:
        raise ValueError("没有已授权的上一调查记录，不能建立修订")
    expected_goals = {f"g{index}" for index, _ in enumerate(frame.get("success_criteria", ()), start=1)}
    goals = draft.goal_assessments or (InvestigationGoal(
        goal_id=next(iter(expected_goals)) if len(expected_goals) == 1 else "answer",
        status="SATISFIED" if draft.goal_satisfied else "PARTIAL",
        claim_refs=tuple(item.claim_id for item in draft.claims if item.revision_state == "ACTIVE"),
        blocking_gaps=tuple(dict.fromkeys(str(gap.code) for gap in gaps)) if not draft.goal_satisfied else (),
    ),)
    if draft.goal_assessments and {item.goal_id for item in goals} != (expected_goals or {"answer"}):
        raise ValueError("必须逐项评估冻结任务的全部完成标准，不能增删目标")
    if len(expected_goals) > 1 and not draft.goal_assessments:
        # 未逐目标评估时保留局部回答，不能用一个布尔值掩盖未完成目标。
        goals = tuple(InvestigationGoal(goal_id=key, status="PARTIAL", explanation="尚未逐项核验") for key in sorted(expected_goals))
    satisfied = all(goal.status in {"SATISFIED", "NOT_APPLICABLE"} for goal in goals)
    record = InvestigationRecord(
        revision=parent.revision + 1 if parent else 1,
        previous_revision=parent.artifact_id if parent else None,
        parent=parent, claim_revisions=draft.claim_revisions,
        record_id=parent.record_id if parent else str(uuid5(NAMESPACE_URL, f"aiops:{context.run_id}:{context.task_id}")),
        target_id=str(context.target_id), run_id=str(context.run_id), task_id=str(context.task_id),
        original_request=str(answer.get("question") or ""),
        task_kind=str(frame.get("task_kind") or frame.get("diagnostic_profile") or "DIAGNOSIS"),
        generated_at=datetime.now(timezone.utc).isoformat(),
        observations=tuple(InvestigationObservation(
            evidence_ref=fact.evidence_ref, artifact_id=str(fact.artifact_id),
            tool_id=fact.tool_id, source_id=fact.source_id, captured_at=fact.captured_at,
            measurement_semantics=str(fact.measurement_semantics), trust_level=fact.trust_level,
            row_count=fact.row_count, truncated=fact.truncated,
            followup_run_id=str(fact.followup_run_id) if fact.followup_run_id else None,
            provenance=fact.provenance, derivations=fact.derivations,
            sql_plan=fact.sql_plan,
            object_access=fact.object_access,
            pagination=fact.pagination,
            source_pagination=fact.source_pagination,
            calculation_graphs=fact.calculation_graphs,
            series_computations=fact.series_computations,
            source_identity_witnesses=fact.source_identity_witnesses,
        ) for fact in evidence),
        claims=draft.claims, recommendations=draft.recommendations,
        access_gaps=tuple(InvestigationAccessGap(
            evidence_ref=gap.evidence_ref, source_id=gap.source_id, step_id=gap.step_id,
            code=gap.code, object_access=gap.object_access,
        ) for gap in gaps if gap.object_access),
        goal_assessments=goals,
        unresolved_questions=draft.unresolved_questions,
        stop_reason="GOAL_ANSWERED" if satisfied else "PARTIAL_ANSWER",
    )
    from aiops_agent.application.implementation.diagnostic_runbook import compile_diagnostic_runbook
    from aiops_agent.domain.evidence.physical_identity import build_physical_identity_graph
    identity_graph = build_physical_identity_graph(evidence)
    if identity_graph.nodes:
        record = record.model_copy(update={"physical_identity_graph": identity_graph})
    capability = dict(context.plan_snapshot.get("capability_snapshot") or {})
    database = dict(dict(context.plan_snapshot.get("investigation_execution") or {}).get("database") or {})
    if frame.get("diagnostic_profile") in {"SINGLE_SQL_PERFORMANCE", "SQL_COMPARISON"}:
        from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle
        from aiops_agent.contracts.sql_object_graph import SqlObjectGraph
        bundle = build_sql_evidence_bundle(task_frame=frame,
            database_type=str(capability.get("database_type") or database.get("db_type") or ""), evidence=evidence)
        if bundle and bundle.get("object_graph") is not None:
            record = record.model_copy(update={"sql_object_graph": SqlObjectGraph.model_validate(bundle["object_graph"])})
        elif bundle:
            graphs = tuple(SqlObjectGraph.model_validate(item["object_graph"]) for item in bundle["subjects"] if item.get("object_graph") is not None)
            record = record.model_copy(update={"sql_subject_graphs": graphs})
        if bundle:
            from aiops_agent.contracts.sql_plan import SqlPlanAssessment
            record = record.model_copy(update={"sql_plan_assessments": tuple(
                SqlPlanAssessment.model_validate(item) for item in bundle.get("plan_assessments", ())
            )})
    record = type(record).model_validate(record.model_dump(mode="python"))
    record.validate_observations(evidence)
    runbook = compile_diagnostic_runbook(
        record, database_type=str(capability.get("database_type") or database.get("db_type") or ""),
        database_version=capability.get("database_version"),
    )
    return record.model_copy(update={"runbook": runbook})


def sql_investigation_draft(draft, subject, subjects=()):
    """保留 SQL 专项校验的输出，但不从自由正文猜测原因类型。"""
    labels = {item["subject_id"]: "SQL " + str(next((item["subject_ref"][key] for key in ("query_id", "digest", "sql_id") if key in item["subject_ref"]), subject)) + "（" + "，".join(f"{key}={value}" for key, value in item["dimensions"].items()) + "）"
              for item in subjects}
    return DiagnosisAnswerDraft(
        claims=tuple(InvestigationClaim(
            claim_id=item.conclusion_id,
            kind="FACT" if item.level == "OBSERVED" else "INTERPRETATION",
            subject=("；".join(labels[key] for key in item.subject_ids) if item.subject_ids else subject)
                if item.scope in {"SUBJECT", "COMPARISON"} else "数据库背景（不可直接归因到指定 SQL）",
            statement=item.statement, level=item.level, applicable_time=item.applicable_time,
            evidence_fields=item.evidence_fields, reasoning=item.reasoning,
            alternatives=item.alternatives,
            limitations=item.limitations or (("未确认原因，须继续核验。",) if item.level != "OBSERVED" else ()),
        ) for item in draft.conclusions),
        recommendations=draft.recommendations,
        unresolved_questions=draft.unresolved_questions, goal_satisfied=draft.goal_satisfied,
        goal_assessments=draft.goal_assessments,
        continues_investigation=draft.continues_investigation, claim_revisions=draft.claim_revisions,
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
        if action.missing_parameters:
            parts.append("待补齐参数：" + "；".join(action.missing_parameters))
        if action.parameters:
            parts.append("参数及来源：" + "；".join(
                f"{item.name}={item.value if item.value is not None else '未知'}（{item.source}）"
                for item in action.parameters
            ))
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
        "implementation_runbook": record.runbook.model_dump(mode="json") if record.runbook else None,
    }
