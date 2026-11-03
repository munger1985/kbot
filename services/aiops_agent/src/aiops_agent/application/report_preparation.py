"""报告受理时只冻结既有调查与确切模板，不装配产物、不重新取证。"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from aiops_agent.application.errors import resource_not_found, state_conflict, validation_failed
from aiops_agent.application.inspections.period_sources import inspection_period_sources
from aiops_agent.application.inspections.monthly import monthly_business_review, merge_monthly_comparison
from aiops_agent.application.report_lifecycle import report_template_for_source
from aiops_agent.application.reporting import (
    ReportTemplate, closed_period_window, compare_monthly_inspection_summaries, normalize_report_source,
)


@dataclass(frozen=True)
class PreparedReport:
    run: Any
    task: Any
    source_artifact: Any
    source: dict
    template: ReportTemplate
    period_kind: str
    period_start: datetime
    period_end: datetime
    extra_artifacts: tuple = ()


async def prepare_user_report(
    runtime,
    *,
    uow,
    domain_id: UUID,
    actor_id: UUID,
    ops_run_id: UUID,
    template: ReportTemplate,
    period_kind: str,
    trace_id: str,
) -> PreparedReport:
    """从完成的聊天或告警诊断显式创建正式报告。"""
    assert uow.inspections is not None
    run = await uow.runs.get_run_scoped(
        ops_run_id=ops_run_id, domain_id=domain_id, lock=True
    )
    if run is None:
        raise resource_not_found("OpsRun")
    source_kind = (
        "CHAT" if run.trigger_type == "CHAT"
        else "INSPECTION" if run.trigger_type == "SCHEDULE"
        else "ALERT"
    )
    if source_kind not in template.applicable_source_kinds:
        raise validation_failed("所选报告模板不适用于当前诊断入口")
    if source_kind != "INSPECTION":
        period_kind = "AD_HOC"
    elif period_kind not in {"DAILY", "MONTHLY", "QUARTERLY", "ANNUAL"}:
        raise validation_failed("巡检报告周期无效")
    if period_kind not in template.allowed_period_kinds:
        raise validation_failed("所选报告模板不适用于当前报告周期")
    if run.status not in {"COMPLETED", "PARTIAL"}:
        raise state_conflict("诊断尚未结束，暂不能生成正式报告")
    if run.final_artifact_id is None:
        raise state_conflict("诊断未形成可报告的最终结果")
    source_artifact = await uow.runs.get_artifact(
        artifact_id=run.final_artifact_id
    )
    if source_artifact is None:
        raise state_conflict("诊断最终结果引用不完整")
    if source_artifact.schema_version not in {
        "DIAGNOSIS_REPORT_DRAFT.v1", "AIOPS_TURN_RESULT.v1",
        "DB_DIAGNOSTIC_REPORT.v1", "REPORT_CONTENT.v1",
    }:
        raise validation_failed("当前诊断结果不支持生成正式报告")
    from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
    if not await validate_investigation_sources(
        uow=uow, run=run, payload=dict(source_artifact.payload_json or {}),
        plan_capture_reader=getattr(runtime, "_plan_capture_reader", None),
    ):
        raise state_conflict("调查计算或对象关系来源失效，不能生成正式报告")
    task_artifact_id = source_artifact.artifact_id
    if source_artifact.schema_version == "REPORT_CONTENT.v1":
        source_artifact_id = str(
            dict(source_artifact.provenance_json or {}).get(
                "source_artifact_id"
            )
            or ""
        )
        if source_artifact_id:
            try:
                parent_artifact = await uow.runs.get_artifact(
                    artifact_id=UUID(source_artifact_id)
                )
            except ValueError:
                parent_artifact = None
            if parent_artifact is not None:
                task_artifact_id = parent_artifact.artifact_id
    task = next(
        (
            item for item in await uow.runs.list_tasks(
                ops_run_id=run.ops_run_id
            )
            if item.output_artifact_id == task_artifact_id
        ),
        None,
    )
    if task is None:
        raise state_conflict("诊断最终结果缺少生成任务")
    now = await uow.runs.database_now()
    extra_artifacts = []
    source_override = None
    period_start = None
    period_end = None
    if source_kind == "INSPECTION" and period_kind != "DAILY":
        inspection = dict(
            dict(run.plan_snapshot_json or {}).get(
                "client_metadata", {}
            ).get("inspection", {})
        )
        timezone = str(inspection.get("timezone") or "UTC")
        period_start, period_end = closed_period_window(
            period_kind=period_kind, timezone=timezone, now=now
        )
        period_runs, business_scope, missing_targets = await inspection_period_sources(
            uow=uow, run=run, inspection=inspection, domain_id=domain_id,
            period_start=period_start, period_end=period_end,
        )
        if not period_runs and not missing_targets:
            raise state_conflict("当前完整报告周期内没有可汇总的巡检结果")
        source_override = await runtime._aggregate_inspection_sources(
            uow=uow, runs=period_runs, period_kind=period_kind,
            period_start=period_start, period_end=period_end, timezone=timezone,
        )
        if business_scope is not None:
            source_override.update(business_scope)
        source_override["gaps"] = (*source_override["gaps"], *missing_targets)
        if period_kind == "MONTHLY":
            review = monthly_business_review(
                facts=source_override["facts"],
                period_start=period_start, period_end=period_end, timezone=timezone,
                members=(business_scope or {}).get("member_targets", ()),
            )
            if missing_targets:
                review["limitations"].extend(str(item["detail"]) for item in missing_targets)
                review["conclusion"]["completion_display"] = "部分完成"
                review["conclusion"]["text"] += " 存在未生成运行的目标，详见检查限制。"
                if review["conclusion"].get("health_status") == "NORMAL":
                    review["conclusion"].update(health_status="INCOMPLETE", health_status_display="检查未完成")
            metric_limitations = source_override["monthly_summary"].get("metric_limitations", [])
            review["limitations"] = list(dict.fromkeys((*review["limitations"], *metric_limitations)))
            source_override["monthly_summary"].update(review)
            previous_summary = None
            member_ids = [
                UUID(item["target_id"])
                for item in (business_scope or {}).get("member_targets", ())
            ] or [run.target_id]
            for target_id in member_ids:
                candidate = await uow.inspections.get_previous_current_period_report(
                    target_id=target_id, report_type="INSPECTION_MONTHLY",
                    before_period_start=period_start,
                )
                if (candidate is None or not candidate.content_artifact_id
                    or getattr(candidate, "period_end", None) != period_start):
                    continue
                artifact = await uow.runs.get_artifact(artifact_id=candidate.content_artifact_id)
                if artifact is None:
                    continue
                previous_scope = dict(dict(artifact.payload_json or {}).get("scope") or {})
                if business_scope and str(previous_scope.get("business_system_id") or "") != str(business_scope["business_system_id"]):
                    continue
                previous_summary = dict(previous_scope.get("monthly_summary") or {}) or None
                if previous_summary is not None:
                    extra_artifacts.append((candidate.ops_run_id, artifact))
                break
            source_override["previous_month_summary"] = previous_summary
            source_override["previous_month_comparison"] = compare_monthly_inspection_summaries(
                source_override["monthly_summary"], previous_summary,
            )
            finalized = merge_monthly_comparison(
                source_override["monthly_summary"],
                source_override["previous_month_comparison"],
            )
            source_override["monthly_summary"] = finalized
            source_override["timezone"] = timezone
            source_override["health_status"] = finalized["conclusion"]["health_status"]
            source_override["diagnosis_rationale"] = finalized["conclusion"]["text"]
            source_override["status"] = "PARTIAL" if finalized["limitations"] else "READY"
            source_override["solution"] = {
                "long_term_remediations": tuple(finalized["recommendations"]),
            }
    template = report_template_for_source(
        template=template, plan=dict(run.plan_snapshot_json or {}), source_kind=source_kind,
    )
    return PreparedReport(run=run, task=task, source_artifact=source_artifact,
        source=source_override or normalize_report_source(
            schema_version=source_artifact.schema_version,
            payload=dict(source_artifact.payload_json or {}), source_kind=source_kind),
        template=template, period_kind=period_kind, period_start=period_start or run.created_at,
        period_end=period_end or now, extra_artifacts=tuple(extra_artifacts))


async def prepare_conversation_report(
    runtime,
    *,
    uow,
    domain_id: UUID,
    actor_id: UUID,
    conversation_id: UUID,
    template: ReportTemplate,
    trace_id: str,
) -> PreparedReport:
    """冻结一个智能运维 Session 内全部已终态 Turn 的报告上下文。"""
    assert uow.inspections is not None
    conversation = await uow.conversations.get_conversation(
        conversation_id=conversation_id,
        domain_id=domain_id,
        lock=True,
    )
    if conversation is None or conversation.created_by != actor_id:
        raise resource_not_found("Conversation")
    if "CHAT" not in template.applicable_source_kinds:
        raise validation_failed("所选报告模板不适用于智能运维会话")
    if "AD_HOC" not in template.allowed_period_kinds:
        raise validation_failed("所选报告模板不适用于会话诊断报告")
    turns = await uow.turns.list_all_turns(
        conversation_id=conversation_id
    )
    if not turns:
        raise state_conflict("当前会话尚无可生成报告的诊断内容")
    active_turns = [
        turn for turn in turns
        if turn.status not in {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"}
    ]
    if active_turns:
        raise state_conflict("会话仍有进行中的诊断，暂不能生成正式报告")

    source_rows: list[tuple[Any, Any, Any]] = []
    missing_turns: list[dict[str, Any]] = []
    for turn in turns:
        link = await uow.turns.get_run_link(
            turn_id=turn.turn_id, purpose="PRIMARY"
        )
        if link is None:
            missing_turns.append({
                "code": "MISSING_PRIMARY_RUN",
                "turn_id": str(turn.turn_id),
                "turn_no": int(turn.turn_no),
            })
            continue
        run = await uow.runs.get_run_scoped(
            ops_run_id=link.ops_run_id,
            domain_id=domain_id,
            lock=True,
        )
        if (
            run is None
            or run.status not in {"COMPLETED", "PARTIAL"}
            or run.final_artifact_id is None
        ):
            missing_turns.append({
                "code": "MISSING_FINAL_RESULT",
                "turn_id": str(turn.turn_id),
                "turn_no": int(turn.turn_no),
            })
            continue
        artifact = await uow.runs.get_artifact(
            artifact_id=run.final_artifact_id
        )
        if artifact is None or artifact.schema_version not in {
            "DIAGNOSIS_REPORT_DRAFT.v1", "AIOPS_TURN_RESULT.v1",
            "DB_DIAGNOSTIC_REPORT.v1", "REPORT_CONTENT.v1",
        }:
            missing_turns.append({
                "code": "UNREPORTABLE_FINAL_RESULT",
                "turn_id": str(turn.turn_id),
                "turn_no": int(turn.turn_no),
            })
            continue
        from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
        if not await validate_investigation_sources(uow=uow, run=run, payload=dict(artifact.payload_json or {}),
            plan_capture_reader=getattr(runtime, "_plan_capture_reader", None)):
            raise state_conflict("调查计算或对象关系来源失效，不能汇入会话报告")
        source_rows.append((turn, run, artifact))
    if not source_rows:
        raise state_conflict("当前会话尚未形成可报告的诊断结果")

    _, anchor_run, anchor_artifact = source_rows[-1]
    task_artifact_id = anchor_artifact.artifact_id
    if anchor_artifact.schema_version == "REPORT_CONTENT.v1":
        source_artifact_id = str(
            dict(anchor_artifact.provenance_json or {}).get(
                "source_artifact_id"
            ) or ""
        )
        if source_artifact_id:
            try:
                parent_artifact = await uow.runs.get_artifact(
                    artifact_id=UUID(source_artifact_id)
                )
            except ValueError:
                parent_artifact = None
            if parent_artifact is not None:
                task_artifact_id = parent_artifact.artifact_id
    task = next(
        (
            item for item in await uow.runs.list_tasks(
                ops_run_id=anchor_run.ops_run_id
            )
            if item.output_artifact_id == task_artifact_id
        ),
        None,
    )
    if task is None:
        raise state_conflict("会话最终诊断结果缺少生成任务")
    now = await uow.runs.database_now()
    source_override = runtime._aggregate_conversation_sources(
        conversation=conversation,
        source_rows=source_rows,
        missing_turns=missing_turns,
    )
    closure_artifact = None
    report_run = anchor_run
    report_task = task
    closure_candidates = []
    for _, candidate_run, _ in reversed(source_rows):
        candidate = await uow.runs.get_latest_artifact_by_prefix(
            ops_run_id=candidate_run.ops_run_id,
            artifact_key_prefix=f"diagnosis-closure:{conversation_id}:",
        )
        if (
            candidate is None
            or candidate.schema_version != "DIAGNOSIS_CLOSURE.v1"
        ):
            continue
        closure_candidates.append((candidate_run, candidate))
    # 收口先后以不可变原件登记时间为准，不由所选最后一轮决定。
    # 最新收口失效时保留缺口，不能退回更早人工判断而掩盖更正。
    closure_candidates.sort(key=lambda row: (row[1].created_at, str(row[1].artifact_id)), reverse=True)
    for candidate_run, candidate in closure_candidates[:1]:
        from aiops_agent.application.case_closure import validate_closure_sources
        if not await validate_closure_sources(uow=uow, domain_id=domain_id, run=candidate_run,
            artifact=candidate, plan_capture_reader=getattr(runtime, "_plan_capture_reader", None)):
            source_override["gaps"] = (*source_override.get("gaps", ()), {
                "code": "CLOSURE_SOURCE_INVALIDATED",
                "message": "人工收口的来源或授权已变化，未作为报告事实附录。",
            })
            source_override["status"] = "PARTIAL"
            continue
        candidate_tasks = await uow.runs.list_tasks(
            ops_run_id=candidate_run.ops_run_id
        )
        candidate_task = next((
            item for item in candidate_tasks
            if item.ops_task_id == candidate.ops_task_id
        ), None)
        if candidate_task is None:
            source_override["gaps"] = (*source_override.get("gaps", ()), {
                "code": "CLOSURE_TASK_MISSING", "message": "最新人工收口缺少生成任务，未作为报告事实附录。",
            })
            source_override["status"] = "PARTIAL"
            continue
        closure_artifact = candidate
        report_run = candidate_run
        report_task = candidate_task
        break
    if closure_artifact is not None:
        closure_source = normalize_report_source(
            schema_version=closure_artifact.schema_version,
            payload=dict(closure_artifact.payload_json or {}),
            source_kind="CHAT",
        )
        # 人工收口作为附录，不覆盖多轮调查正文和后续反证。
        source_override["facts"] = (*source_override.get("facts", ()), *closure_source.get("facts", ()))
        source_override["closure_artifact_id"] = str(closure_artifact.artifact_id)
        source_override["verification"] = closure_source.get("verification", {})
        source_override["executed_actions"] = closure_source.get("executed_actions", ())
        source_override["action_source"] = closure_source.get("action_source")
        source_override["evidence_refs"] = (*source_override.get("evidence_refs", ()), *closure_source.get("evidence_refs", ()))
        source_override["conversation"] = {
            **dict(source_override.get("conversation") or {}),
            "title": conversation.title,
            "question_summaries": list(dict.fromkeys(
                str(dict(run.plan_snapshot_json or {}).get(
                    "diagnosis", {}
                ).get("question_summary") or "").strip()
                for _, run, _ in source_rows
                if str(dict(run.plan_snapshot_json or {}).get(
                    "diagnosis", {}
                ).get("question_summary") or "").strip()
            )),
        }
    template = report_template_for_source(
        template=template, plan=dict(report_run.plan_snapshot_json or {}), source_kind="CHAT",
    )
    return PreparedReport(run=report_run, task=report_task,
        source_artifact=closure_artifact or anchor_artifact, source=source_override, template=template,
        period_kind="AD_HOC", period_start=(getattr(source_rows[0][1], "started_at", None)
            or source_rows[0][1].created_at), period_end=now,
        extra_artifacts=tuple((row.ops_run_id, artifact) for _, row, artifact in source_rows))
