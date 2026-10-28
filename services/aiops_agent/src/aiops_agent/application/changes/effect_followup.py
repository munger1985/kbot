"""复用已授权周期采集，在固定窗口结束后自动复算冻结的业务效果。"""

from datetime import datetime, timedelta
from uuid import UUID

from aiops_agent.application.changes.effect import ChangeEffectVerification, _key, effect_source
from aiops_agent.application.configuration.common import sha256_json
from aiops_agent.application.errors import AIOpsApplicationError, state_conflict
from aiops_agent.contracts.report import ComparisonPlan
from aiops_agent.entities import OutboxEntity
from platform_core.contracts.aiops.effect import EffectEvaluationCommand


async def schedule_effect_followup(
    *, uow, run, proposal, verification_artifact, receipt, now, trace_id,
):
    """与动作复核在同一事务排队；只读取已冻结条件，不产生变更授权。"""
    plan_artifact = await uow.runs.get_artifact_by_key(
        ops_run_id=run.ops_run_id, artifact_key=_key(proposal.proposal_id, "plan"),
    )
    if plan_artifact is None:
        return False
    await effect_source(uow=uow, artifact_id=plan_artifact.artifact_id, run=run)
    plan = ComparisonPlan.model_validate(plan_artifact.payload_json)
    frozen = plan.business_effect.model_dump(mode="json") if plan.business_effect else {}
    if not frozen.get("auto_evaluate"):
        return False
    if receipt.payload_json.get("status") not in {"EXECUTED", "SUCCEEDED"}:
        return False
    key = _key(proposal.proposal_id, "automatic")
    if await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key=key):
        return False
    occurred_at = datetime.fromisoformat(str(receipt.payload_json["occurred_at"]).replace("Z", "+00:00"))
    interval = int(frozen["collection_interval_seconds"])
    if frozen.get("controlled_comparison"):
        interval = max(interval, int(frozen["control_collection_interval_seconds"]))
    start = occurred_at + timedelta(seconds=plan.settle_delay_seconds)
    # 各端点最多等待一次采集周期；预留两周期，不把不足的实际区间冒充完整窗口。
    end = start + timedelta(seconds=plan.after_window_seconds + 2 * interval)
    payload = {
        "proposal_id": str(proposal.proposal_id),
        "source_run_id": str(run.ops_run_id),
        "domain_id": str(run.domain_id),
        "actor_id": str(run.actor_id),
        "plan_artifact_id": str(plan_artifact.artifact_id),
        "plan_hash": plan_artifact.content_hash,
        "verification_artifact_id": str(verification_artifact.artifact_id),
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "available_at": max(now, end + timedelta(seconds=60)).isoformat(),
        "status": "SCHEDULED",
    }
    await ChangeEffectVerification._effect_artifact(
        uow, run, proposal, key, "BUSINESS_EFFECT_FOLLOWUP.v1", payload,
        {"plan_hash": plan_artifact.content_hash}, plan_artifact.security_level,
    )
    await uow.outbox.add(OutboxEntity(
        aggregate_type="OPS_CHANGE_PROPOSAL", aggregate_id=proposal.proposal_id,
        event_type="OPS_BUSINESS_EFFECT_REQUESTED", idempotency_key=key,
        payload_json=payload, payload_hash=sha256_json(payload), status="PENDING",
        available_at=datetime.fromisoformat(payload["available_at"]),
        max_attempts=5, trace_id=trace_id,
    ))
    await uow.runs.append_event(
        ops_run_id=run.ops_run_id, ops_task_id=proposal.ops_task_id,
        event_type="comparison.business_effect.scheduled", event_key=key,
        visibility="USER", payload_json={**payload, "trace_id": trace_id},
    )
    return True


class BusinessEffectFollowupService:
    def __init__(self, *, uow_factory, change_service, workload_service):
        self._uow_factory = uow_factory
        self._changes = change_service
        self._workloads = workload_service

    async def _scope(self, uow, payload):
        run, proposal = await self._changes._effect_scope(
            uow, UUID(payload["proposal_id"]), int(payload["domain_id"]),
            payload["actor_id"],
        )
        if str(run.ops_run_id) != payload["source_run_id"]:
            raise state_conflict("自动效果验证的调查来源不一致")
        request = await uow.runs.get_artifact_by_key(
            ops_run_id=run.ops_run_id,
            artifact_key=_key(proposal.proposal_id, "automatic"),
        )
        if request is None:
            raise state_conflict("自动效果验证缺少冻结任务")
        request = await effect_source(
            uow=uow, artifact_id=request.artifact_id,
            run=run, schema="BUSINESS_EFFECT_FOLLOWUP.v1",
        )
        if request.payload_json != payload:
            raise state_conflict("自动效果验证请求与冻结的任务不一致")
        plan = await effect_source(
            uow=uow, artifact_id=UUID(payload["plan_artifact_id"]),
            run=run, schema="COMPARISON_PLAN.v1",
        )
        if plan.content_hash != payload["plan_hash"]:
            raise state_conflict("自动效果验证计划哈希变化")
        return run, proposal, ComparisonPlan.model_validate(plan.payload_json)

    async def execute(self, payload):
        async with self._uow_factory() as uow:
            run, proposal, plan = await self._scope(uow, payload)
            prior = await uow.runs.get_artifact_by_key(
                ops_run_id=run.ops_run_id,
                artifact_key=_key(proposal.proposal_id, "automatic-result"),
            )
            if prior is not None:
                return
            now = await uow.runs.database_now()
            if now < datetime.fromisoformat(payload["available_at"]):
                raise state_conflict("观察窗口尚未结束")
            await effect_source(
                uow=uow, artifact_id=UUID(payload["verification_artifact_id"]),
                run=run, schema="ACTION_VERIFICATION.v1",
            )
            scope = {
                "domain_id": run.domain_id,
                "actor_id": run.actor_id, "target_id": run.target_id,
            }
            row_version = int(proposal.row_version)
            control = plan.business_effect.controlled_comparison if plan.business_effect else None
            if control is not None:
                from aiops_agent.application.changes.effect_controls import load_control_reports
                await load_control_reports(
                    uow=uow, run=run, control=control,
                    frozen=plan.business_effect.model_dump(mode="json"),
                )
        report = await self._workloads.create_workload_report(
            **scope, period_start=datetime.fromisoformat(payload["period_start"]),
            period_end=datetime.fromisoformat(payload["period_end"]),
            idempotency_key=f"effect-after:{payload['proposal_id']}",
            trace_id="business-effect-followup",
        )
        control_report = None
        if control is not None:
            control_report = await self._workloads.create_workload_report(
                **{**scope, "target_id": control.target_id},
                period_start=datetime.fromisoformat(payload["period_start"]),
                period_end=datetime.fromisoformat(payload["period_end"]),
                idempotency_key=f"effect-control-after:{payload['proposal_id']}",
                trace_id="business-effect-followup",
            )
        result = await self._changes.evaluate_effect_plan(
            proposal_id=UUID(payload["proposal_id"]), domain_id=scope["domain_id"],
            actor_id=scope["actor_id"], trace_id="business-effect-followup",
            command=EffectEvaluationCommand(
                expected_row_version=row_version, after_artifact_id=report.artifact_id,
                verification_artifact_id=payload["verification_artifact_id"],
                control_after_artifact_id=control_report.artifact_id if control_report else None,
            ),
        )
        await self._finish(payload, status="COMPLETED", report_id=str(result.report_id))

    async def fail(self, payload, exc):
        # 仅保存安全错误类别；不复制 SQL、模型原文或已撤权正文。
        code = exc.code if isinstance(exc, AIOpsApplicationError) else "EFFECT_AUTOMATIC_EXECUTION_FAILED"
        await self._finish(payload, status="FAILED", error_code=code)

    async def _finish(self, payload, **outcome):
        async with self._uow_factory() as uow:
            # 状态审计不重新授权已撤权正文；只核对原任务归属与哈希。
            run = await uow.runs.get_run(ops_run_id=UUID(payload["source_run_id"]), lock=True)
            if run is None or str(run.domain_id) != payload["domain_id"]:
                raise state_conflict("自动验证的原调查不存在")
            request = await uow.runs.get_artifact_by_key(
                ops_run_id=run.ops_run_id, artifact_key=_key(payload["proposal_id"], "automatic"),
            )
            if request is None or request.payload_json != payload or sha256_json(payload) != request.content_hash:
                raise state_conflict("自动效果验证任务已变化")
            key = _key(payload["proposal_id"], "automatic-result")
            if await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key=key):
                return
            proposal = await uow.changes.get_proposal(proposal_id=UUID(payload["proposal_id"]))
            await self._changes._effect_artifact(
                uow, run, proposal, key, "BUSINESS_EFFECT_FOLLOWUP_RESULT.v1",
                {**payload, **outcome}, {"request_hash": request.content_hash}, request.security_level,
            )
            await uow.runs.append_event(
                ops_run_id=run.ops_run_id, ops_task_id=proposal.ops_task_id,
                event_type="comparison.business_effect.followup_finished", event_key=key,
                visibility="USER", payload_json={"proposal_id": payload["proposal_id"], **outcome},
            )
            await uow.commit()
