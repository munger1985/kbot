"""AIOps 平台通知 Outbox 持久化。"""

from platform_core.notifications import NotificationEnvelope, publish_notification


_MAX_RECIPIENTS_PER_EVENT = 50
_MAX_NOTIFICATION_SUMMARY_LENGTH = 1000


def _notification_summary(value: object, *, fallback: str) -> str:
    """将业务长摘要收敛为平台通知可安全承载的简要说明。"""
    summary = str(value or "").strip()
    if not summary:
        return fallback
    if len(summary) <= _MAX_NOTIFICATION_SUMMARY_LENGTH:
        return summary
    return f"{summary[:_MAX_NOTIFICATION_SUMMARY_LENGTH - 1]}…"


class PlatformNotificationRepository:
    def __init__(self, uow):
        self._uow = uow

    async def emit_output_job_event(self, *, job, event_type, now, result=None):
        """任务身份独立于 Run；只给真实发起人投递安全摘要和受控定位。"""
        if event_type not in {"aiops.output_job.started", "aiops.output_job.failed", "aiops.case.draft_ready"}:
            raise ValueError("输出任务通知类型无效")
        if event_type == "aiops.case.draft_ready":
            if (job.job_type != "CASE_EXTRACTION" or result is None
                or result.report_id is not None or result.asset_id is None or result.asset_version_id is None):
                raise ValueError("草稿就绪通知必须绑定确切案例版本")
        elif result is not None:
            raise ValueError("未成功的任务通知不能携带产物定位")
        summary = ("案例草稿已生成，人工审核发布后才能用于后续诊断。" if event_type == "aiops.case.draft_ready"
            else "任务已排队，完成后会在通知中心告知结果。" if event_type.endswith("started")
            else "输出任务未能完成，请查看该任务的失败原因。")
        return await publish_notification(uow=self._uow, producer_service="aiops",
            event_key=f"output-job:{job.job_id}:{event_type}:{job.attempt_count}",
            envelope=NotificationEnvelope(domain_id=int(job.domain_id), event_type=event_type,
                resource_type="aiops_output_job", resource_id=str(job.job_id),
                resource_name="沉淀案例" if job.job_type == "CASE_EXTRACTION" else "生成报告",
                initiator_actor_id=job.actor_id, recipient_actor_ids=[job.actor_id], summary=summary,
                correlation_id=job.trace_id, operation_id=str(job.job_id), occurred_at=now,
                safe_data={"job_id": str(job.job_id), "job_type": job.job_type,
                    "source_version": int(job.row_version or 0),
                    **({"asset_id": str(result.asset_id), "asset_version_id": str(result.asset_version_id)}
                        if result is not None else {})}))

    async def _publish_for_recipients(
        self,
        *,
        event_key: str,
        recipients,
        envelope_data: dict,
    ):
        """按平台通知信封上限稳定拆分订阅者，避免阻断业务事务。"""
        normalized = tuple(sorted(set(recipients)))
        batches = tuple(
            normalized[index : index + _MAX_RECIPIENTS_PER_EVENT]
            for index in range(0, len(normalized), _MAX_RECIPIENTS_PER_EVENT)
        ) or ((),)
        results = []
        for index, batch in enumerate(batches, start=1):
            effective_key = (
                event_key
                if len(batches) == 1
                else f"{event_key}:part:{index}"
            )
            results.append(
                await publish_notification(
                    uow=self._uow,
                    producer_service="aiops",
                    event_key=effective_key,
                    envelope=NotificationEnvelope(
                        **envelope_data,
                        recipient_actor_ids=list(batch),
                    ),
                )
            )
        return results[0] if len(results) == 1 else tuple(results)

    async def emit_run_event(
        self,
        *,
        run,
        event_type: str,
        summary: str,
        actor_id: str,
        hitl_id=None,
        source_status=None,
        now=None,
    ):
        if event_type.startswith("aiops.diagnostic.input_") and (hitl_id is None or source_status is None):
            raise ValueError("人工交互通知必须绑定确切 HITL 与来源状态")
        return await publish_notification(
            uow=self._uow,
            producer_service="aiops",
            event_key=(f"aiops-hitl:{hitl_id}:{event_type}" if hitl_id is not None
                else f"aiops-run:{run.ops_run_id}:{event_type}"),
            envelope=NotificationEnvelope(
                domain_id=int(run.domain_id),
                event_type=event_type,
                resource_type="aiops_run",
                resource_id=str(run.ops_run_id),
                resource_name="AIOps 诊断",
                initiator_actor_id=run.actor_id,
                recipient_actor_ids=[actor_id],
                summary=summary,
                correlation_id=run.trace_id,
                operation_id=str(run.ops_run_id),
                safe_data=({"hitl_id": str(hitl_id), "source_status": source_status}
                    if hitl_id is not None else {}),
                **({"occurred_at": now} if now is not None else {}),
            ),
        )

    async def emit_case_review_required(self, *, asset, version, run):
        """暂停案例后通知原维护人；不在通知中暴露已撤权正文。"""
        return await publish_notification(
            uow=self._uow, producer_service="aiops",
            event_key=f"aiops-case:{version.asset_version_id}:source-invalidated",
            envelope=NotificationEnvelope(
                domain_id=int(asset.domain_id), event_type="aiops.knowledge.review_required",
                resource_type="aiops_run", resource_id=str(run.ops_run_id),
                resource_name="AIOps 案例", initiator_actor_id=None,
                recipient_actor_ids=[asset.created_by],
                summary="一个已发布案例的来源或授权发生变化，已暂停新检索。请在历史诊断案例中重新核验并审核。",
                correlation_id=run.trace_id,
            ),
        )

    async def emit_situation_event(
        self,
        *,
        target,
        situation,
        event_type: str,
        stage: str,
        summary: str,
        trace_id: str,
    ):
        subscriptions = self._uow.notification_subscriptions
        if subscriptions is None:
            raise RuntimeError("主动分享订阅 Repository 未初始化")
        recipients = await subscriptions.recipient_actor_ids(
            domain_id=int(target.domain_id),
            target_id=target.target_id,
            stage=stage,
            severity=situation.severity,
        )
        return await self._publish_for_recipients(
            event_key=f"aiops-situation:{situation.situation_id}:{event_type}:{situation.severity}",
            recipients=recipients,
            envelope_data=dict(
                domain_id=int(target.domain_id),
                event_type=event_type,
                resource_type="aiops_situation",
                resource_id=str(situation.situation_id),
                resource_name=target.display_name,
                initiator_actor_id="system:signal-intake",
                summary=summary,
                correlation_id=trace_id,
                safe_data={
                    "target_id": str(target.target_id),
                    "severity": situation.severity,
                },
            ),
        )

    async def emit_run_started(self, *, run, target_name: str):
        subscriptions = self._uow.notification_subscriptions
        if subscriptions is None:
            raise RuntimeError("主动分享订阅 Repository 未初始化")
        severity = "INFO"
        if run.situation_id is not None:
            situation = await self._uow.situations.get_situation(
                situation_id=run.situation_id
            )
            if situation is not None:
                severity = situation.severity
        recipients = await subscriptions.recipient_actor_ids(
            domain_id=int(run.domain_id),
            target_id=run.target_id,
            stage="DIAGNOSIS_STARTED",
            severity=severity,
        )
        return await self._publish_for_recipients(
            event_key=f"aiops-run:{run.ops_run_id}:aiops.diagnosis.started",
            recipients=recipients,
            envelope_data=dict(
                domain_id=int(run.domain_id),
                event_type="aiops.diagnosis.started",
                resource_type="aiops_run",
                resource_id=str(run.ops_run_id),
                resource_name=target_name,
                initiator_actor_id=run.actor_id,
                summary="已根据严重故障信号启动主动根因诊断",
                correlation_id=run.trace_id,
                operation_id=str(run.ops_run_id),
                safe_data={"target_id": str(run.target_id), "severity": severity},
            ),
        )

    async def emit_proposal_event(
        self,
        *,
        run,
        proposal,
        event_type: str,
        summary: str,
        actor_id: str,
        execution=None,
        now=None,
    ):
        return await publish_notification(
            uow=self._uow,
            producer_service="aiops",
            event_key=(f"aiops-execution:{execution.execution_id}:{int(execution.status_version)}:{event_type}"
                if execution is not None else f"aiops-proposal:{proposal.proposal_id}:{event_type}"),
            envelope=NotificationEnvelope(
                domain_id=int(run.domain_id),
                event_type=event_type,
                resource_type="aiops_proposal",
                resource_id=str(proposal.proposal_id),
                resource_name="AIOps 变更方案",
                initiator_actor_id=run.actor_id,
                recipient_actor_ids=([] if event_type == "aiops.proposal.review_required"
                    else sorted({value for value in (actor_id, run.actor_id) if value and not str(value).startswith("system:")})),
                summary=summary,
                correlation_id=run.trace_id,
                safe_data={"run_id": str(run.ops_run_id),
                    **({"execution_id": str(execution.execution_id), "source_status": execution.status,
                        "aggregate_version": int(execution.status_version)} if execution is not None else {})},
                **({"occurred_at": now} if now is not None else {}),
            ),
        )

    async def emit_report_ready(self, *, run, report, actor_id: str, output_job=None, now=None):
        if output_job is not None and (
            output_job.job_type != "REPORT_GENERATION" or output_job.actor_id != actor_id
            or output_job.domain_id != run.domain_id
            or report.ops_run_id != run.ops_run_id or output_job.status != "SUCCEEDED"
            or report.status not in {"READY", "PARTIAL"} or now is None
        ):
            raise ValueError("报告任务通知必须绑定当前作用域、发起人和确切产物")
        recipients = {actor_id} if not actor_id.startswith("system:") else set()
        subscriptions = self._uow.notification_subscriptions
        if subscriptions is not None:
            severity = "INFO"
            if run.situation_id is not None:
                situation = await self._uow.situations.get_situation(
                    situation_id=run.situation_id
                )
                if situation is not None:
                    severity = situation.severity
            recipients.update(
                await subscriptions.recipient_actor_ids(
                    domain_id=int(run.domain_id),
                    target_id=run.target_id,
                    stage="REPORT_READY",
                    severity=severity,
                )
            )
        return await self._publish_for_recipients(
            event_key=(f"output-job:{output_job.job_id}:aiops.report.ready"
                if output_job is not None else f"aiops-report:{report.report_id}"),
            recipients=recipients,
            envelope_data=dict(
                domain_id=int(run.domain_id),
                event_type="aiops.report.ready",
                resource_type="aiops_report",
                resource_id=str(report.report_id),
                resource_name="AIOps 报告",
                initiator_actor_id=actor_id if output_job is not None else run.actor_id,
                summary=("报告已生成，存在证据或内容缺口，请查看报告说明。" if report.status == "PARTIAL"
                    else "报告已生成，可查看正式报告。") if output_job is not None else _notification_summary(
                    report.summary,
                    fallback="报告已生成",
                ),
                correlation_id=output_job.trace_id if output_job is not None else run.trace_id,
                operation_id=str(output_job.job_id if output_job is not None else run.ops_run_id),
                safe_data={"target_id": str(run.target_id),
                    **({"job_id": str(output_job.job_id), "job_type": output_job.job_type,
                        "report_id": str(report.report_id), "report_version": str(report.report_version),
                        "source_version": int(output_job.row_version or 0)} if output_job is not None else {})},
                **({"occurred_at": now} if output_job is not None else {}),
            ),
        )
