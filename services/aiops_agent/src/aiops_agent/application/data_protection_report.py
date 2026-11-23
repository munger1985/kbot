"""将冻结的演练结果发布到现有报告中心，不调用恢复步骤。"""

from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from json import dumps
from uuid import UUID

from aiops_agent.application.reporting import resolve_system_template
from aiops_agent.contracts.report.models import ReportContent
from aiops_agent.domain.data_protection_metrics import evaluate_protection_metrics
from aiops_agent.entities.inspection import ReportEntity, ReportSourceEntity
from aiops_agent.entities.runtime import OpsArtifactEntity, OpsRunEntity, OpsTaskEntity
from platform_core.identity import uuid7


_REPORT_TYPE = "RECOVERY_DRILL"
_VERDICTS = {
    "PASS": "达标", "OVER_TARGET": "超标", "INCONCLUSIVE": "尚不能判断是否达标",
    "NOT_MEASURED": "未测得", "NO_TARGET": "未设置期望", "NOT_COMPARABLE": "不可比较",
}
_STATUS = {
    "SUCCEEDED": "恢复验证通过", "FAILED": "恢复验证未通过",
    "REJECTED": "已拒绝写入", "SKIPPED": "已跳过", "UNKNOWN": "现场状态未确认",
}


def _time(value):
    return datetime.fromisoformat(value) if value else None


def _metric_text(name, metric):
    def fmt(lower, upper):
        if lower is None:
            return "—"
        return f"{lower} 秒" if lower == upper else f"{lower} 至 {upper} 秒"

    return (
        f"{name}：期望 {fmt(metric.target_seconds, metric.target_seconds)}；"
        f"实测 {fmt(metric.measured_min_seconds, metric.measured_max_seconds)}；"
        f"实测减期望 {fmt(metric.gap_min_seconds, metric.gap_max_seconds)}；"
        f"{_VERDICTS[metric.verdict]}。"
    )


def _recommendations(status, rto, rpo, stages):
    recommendations = []
    if status != "SUCCEEDED":
        recommendations.append("DBA：根据拒绝或失败项核对独立实例、仓库及权限；进程状态不明时先确认已停止。重新准备空环境后再复演，不覆盖原现场。")
    if rto.verdict == "OVER_TARGET":
        phases = {key: value for key, value in stages.items() if value is not None}
        longest = max(phases, key=phases.get) if phases else None
        actions = {
            "queue_seconds": "核对 Worker 调度与排队延迟",
            "retrieval_seconds": "测量仓库至演练主机吞吐与取回耗时",
            "restore_seconds": "检查基础备份周期、回放量和恢复实例负载",
            "verification_seconds": "核对只读验证项的耗时明细",
        }
        recommendations.append(f"DBA／运维：{actions.get(longest, '补齐各阶段耗时以定位瓶颈')}；在同等预置环境下复演，对比总 RTO 和分段耗时。")
    if rpo.verdict == "OVER_TARGET":
        recommendations.append("DBA：核对 T0 前最后可用日志、归档延迟和链路缺段；用同口径复演，对比可信恢复坐标及时间差。")
    elif rpo.verdict in ("NOT_MEASURED", "INCONCLUSIVE"):
        recommendations.append("DBA：补齐可信恢复坐标与时间映射，验证日志覆盖；在同一 T0 定义下复演。")
    return tuple(recommendations)


def drill_report_input(run):
    """报告仅提取约定的结构化字段；不复制脚本输出、路径或认证材料。"""
    settings, result = run.snapshot_json["settings"], run.result_json or {}
    verified = run.status == "SUCCEEDED" and result.get("database_verified") is True
    t0 = run.scheduled_at.astimezone(UTC)
    rto, rpo = evaluate_protection_metrics(
        simulated_failure_at=t0, verified_at=_time(result.get("verified_at")),
        recovery_coordinate=result.get("recovery_coordinate"),
        recovered_at_earliest=_time(result.get("recovered_at_earliest")),
        recovered_at_latest=_time(result.get("recovered_at_latest")),
        target_rto=settings["rto_seconds"], target_rpo=settings["rpo_seconds"],
        database_verified=verified, latest_recovery_scenario=True,
    )
    claimed = _time(run.snapshot_json["claimed_at"])
    stages = {key: result.get(key) for key in ("retrieval_seconds", "restore_seconds", "verification_seconds")}
    stages["queue_seconds"] = int((claimed - t0).total_seconds())
    artifact = run.snapshot_json.get("backup") or {}
    binding = run.snapshot_json["approval"]["binding"]
    return {
        "schema_version": "RECOVERY_DRILL_EVIDENCE.v1", "protection_run_id": str(run.run_id),
        "resource_id": str(run.resource_id), "tool": settings["tool"], "result": run.status,
        "message": result.get("message"), "database_verified": verified, "application_verified": False,
        "plan_revision": run.snapshot_json["revision"], "template_key": binding["template_key"],
        "template_revision": binding["template_revision"], "scenario": "LATEST_PROVEN",
        "simulated_failure_at": t0.isoformat(), "completed_at": run.completed_at.astimezone(UTC).isoformat(),
        "claimed_at": claimed.isoformat(), "verified_at": result.get("verified_at"),
        "elapsed_seconds": int((run.completed_at - t0).total_seconds()), "stages": stages,
        "backup": {key: artifact.get(key) for key in (
            "run_id", "reference", "available_at", "continuous_logs_verified", "kind", "dependencies",
        )},
        "coordinate_kind": result.get("recovery_coordinate_kind"), "coordinate": result.get("recovery_coordinate"),
        "recovered_at_earliest": result.get("recovered_at_earliest"),
        "recovered_at_latest": result.get("recovered_at_latest"),
        "rto": asdict(rto), "rpo": asdict(rpo),
        "rto_text": _metric_text("RTO", rto), "rpo_text": _metric_text("RPO", rpo),
        "recommendations": _recommendations(run.status, rto, rpo, stages),
    }


def _artifact(*, run_id, task_id, key, artifact_type, payload, at):
    encoded = dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return OpsArtifactEntity(
        artifact_id=uuid7(), ops_run_id=run_id, ops_task_id=task_id, artifact_key=key,
        artifact_type=artifact_type, schema_version=payload["schema_version"], payload_json=payload,
        content_hash=sha256(encoded).hexdigest(), byte_size=len(encoded),
        provenance_json={"producer": "aiops.data-protection", "llm_used": False},
        trust_level="SYSTEM", security_level=1, created_at=at,
    )


async def publish_next_drill_report(*, uow_factory, now):
    async with uow_factory() as uow:
        repo = uow.protection_plans
        drill = await repo.report_pending()
        if drill is None:
            return
        plan = await repo.plan_for_run(drill)
        evidence = drill_report_input(drill)
        approval = drill.snapshot_json["approval"]
        target_id = UUID(approval["binding"]["target_id"])
        run_id, task_id, report_id = uuid7(), uuid7(), uuid7()
        raw = _artifact(run_id=run_id, task_id=task_id, key="recovery-drill:evidence", artifact_type="DRILL_EVIDENCE",
                        payload=evidence, at=now)
        t0, ended = _time(evidence["simulated_failure_at"]), _time(evidence["completed_at"])
        title = f"{evidence['tool']} 空库恢复演练报告"
        status = ("READY" if evidence["database_verified"] and evidence["rpo"]["measured_min_seconds"] is not None
                  else "PARTIAL")
        labels = {"queue_seconds": "调度等待", "retrieval_seconds": "取回", "restore_seconds": "恢复与回放",
                  "verification_seconds": "验证"}
        stages = "；".join(f"{name}：{evidence['stages'][key]} 秒" if evidence["stages"][key] is not None
                           else f"{name}：未测得" for key, name in labels.items())
        summary = _STATUS[drill.status] + "。"
        backup_fact = evidence["backup"]
        backup_type = "增量备份" if backup_fact["kind"] == "INCREMENTAL" else "全量备份"
        backup_description = (f"{backup_type} {backup_fact['reference']}" if backup_fact["reference"] else "未选定")
        if backup_fact["dependencies"]:
            backup_description += "；所需基线及前序备份：" + " → ".join(backup_fact["dependencies"])
        template = resolve_system_template("system:recovery-drill")
        content = ReportContent(
            report_key="recovery-drill", report_type=_REPORT_TYPE, ops_run_id=str(run_id), target_id=str(target_id),
            title=title, status=status, summary=summary, period_start=t0, period_end=ended,
            scope={"raw_report_artifact_id": str(raw.artifact_id), "resource_id": str(drill.resource_id),
                   "protection_run_id": str(drill.run_id), "plan_revision": evidence["plan_revision"]},
            facts=tuple({"summary": item} for item in (
                f"演练编号：{drill.run_id}；配置版本：{evidence['plan_revision']}；模板版本：{evidence['template_revision']}。",
                "仅演练专用空实例，使用 T0 时已存在产物；不验证生产恢复、业务可用或主备切换。",
                f"计划 / T0：{t0.isoformat()}；结束：{ended.isoformat()}；已用 {evidence['elapsed_seconds']} 秒。",
                f"{stages}。报告生成时间不计入 RTO。",
                f"产物：{backup_description}；连续日志："
                f"{'已验证' if evidence['backup']['continuous_logs_verified'] is True else '未证明连续'}。",
                f"验证坐标：{evidence['coordinate_kind'] or '未验证'} {evidence['coordinate'] or ''}。",
                evidence["rto_text"], evidence["rpo_text"],
            )),
            gaps=({"code": "REPORT_CONTRACT", "detail": evidence["message"] or summary},) if status == "PARTIAL" else (),
            evidence_refs=({"artifact_id": str(raw.artifact_id), "content_hash": raw.content_hash,
                            "schema_version": raw.schema_version},),
            recommendations=evidence["recommendations"], provenance={
                "producer": "aiops.data-protection", "llm_used": False,
                "template": {"template_ref": template.template_ref, "version": template.version,
                             "content_hash": template.content_hash, "definition": template.definition},
            },
        )
        display = _artifact(run_id=run_id, task_id=task_id, key="recovery-drill:report", artifact_type="REPORT_CONTENT",
                            payload=content.model_dump(mode="json"), at=now)
        scope = {key: getattr(plan, key) for key in ("tenant_id", "domain_id") if hasattr(plan, key)}
        await uow.runs.add_run(OpsRunEntity(
            ops_run_id=run_id, **scope, target_id=target_id, actor_id=approval["actor_id"],
            trigger_type="SCHEDULE", interaction_mode="ASYNCHRONOUS", workflow_kind="RECOVERY_DRILL",
            idempotency_key=f"recovery-drill:{drill.run_id}", status="COMPLETED", original_request=title,
            final_artifact_id=display.artifact_id, trace_id=str(drill.run_id), created_at=now,
            started_at=now, completed_at=now, updated_at=now,
        ))
        await uow.runs.add_task(OpsTaskEntity(
            ops_task_id=task_id, ops_run_id=run_id, task_key="recovery-drill:report", task_type="REPORT_BUILD",
            handler_id="data-protection.report", handler_version="1", status="COMPLETED",
            input_schema_version=raw.schema_version, output_schema_version=display.schema_version,
            output_artifact_id=display.artifact_id, available_at=now, attempt_count=1, max_attempts=1,
            timeout_seconds=120, started_at=now, completed_at=now, created_at=now, updated_at=now,
        ))
        await uow.runs.add_artifact(raw)
        await uow.runs.add_artifact(display)
        await uow.inspections.add_report(ReportEntity(
            report_id=report_id, ops_run_id=run_id, target_id=target_id, report_key=content.report_key,
            report_version=1, is_current=True, report_type=_REPORT_TYPE, title=title, status=status,
            summary=summary, period_start=t0, period_end=ended, result=drill.status,
            template_id=template.template_ref, template_version=template.version, generated_by_task_id=task_id,
            content_artifact_id=display.artifact_id, content_hash=display.content_hash, security_level=1,
            schema_version=display.schema_version, created_at=now, updated_at=now,
        ))
        await uow.inspections.add_report_sources([ReportSourceEntity(
            report_id=report_id, ops_run_id=run_id, source_artifact_id=raw.artifact_id, source_kind=_REPORT_TYPE,
            content_hash=raw.content_hash, observed_at=ended, created_at=now,
        )])
        drill.report_id = report_id
        await uow.commit()

