"""AIOps 平台通知 Outbox 持久化。"""

from platform_core.notifications import NotificationEnvelope, publish_notification


_MAX_RECIPIENTS_PER_EVENT = 50


class PlatformNotificationRepository:
    def __init__(self, uow):
        self._uow = uow

    async def _publish_for_recipients(
        self,
        *,
        event_key: str,
        recipients,
        envelope_data: dict,
    ):
        """按平台通知信封上限稳定拆分订阅者，避免阻断业务事务。"""
        normalized = tuple(sorted(set(recipients)))
        if not normalized:
            return None
        batches = tuple(
            normalized[index : index + _MAX_RECIPIENTS_PER_EVENT]
            for index in range(0, len(normalized), _MAX_RECIPIENTS_PER_EVENT)
        )
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
    ):
        return await publish_notification(
            uow=self._uow,
            producer_service="aiops",
            event_key=f"aiops-run:{run.ops_run_id}:{event_type}",
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
        if not recipients:
            return None
        return await self._publish_for_recipients(
            event_key=f"aiops-situation:{situation.situation_id}:{event_type}",
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
        if not recipients:
            return None
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
                safe_data={"target_id": str(run.target_id)},
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
    ):
        return await publish_notification(
            uow=self._uow,
            producer_service="aiops",
            event_key=f"aiops-proposal:{proposal.proposal_id}:{event_type}",
            envelope=NotificationEnvelope(
                domain_id=int(run.domain_id),
                event_type=event_type,
                resource_type="aiops_proposal",
                resource_id=str(proposal.proposal_id),
                resource_name="AIOps 变更方案",
                initiator_actor_id=run.actor_id,
                recipient_actor_ids=[actor_id],
                summary=summary,
                correlation_id=run.trace_id,
                safe_data={"run_id": str(run.ops_run_id)},
            ),
        )

    async def emit_report_ready(self, *, run, report, actor_id: str):
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
        if not recipients:
            return None
        return await self._publish_for_recipients(
            event_key=f"aiops-report:{report.report_id}",
            recipients=recipients,
            envelope_data=dict(
                domain_id=int(run.domain_id),
                event_type="aiops.report.ready",
                resource_type="aiops_report",
                resource_id=str(report.report_id),
                resource_name="AIOps 报告",
                initiator_actor_id=run.actor_id,
                summary=report.summary or "报告已生成",
                correlation_id=run.trace_id,
                operation_id=str(run.ops_run_id),
                safe_data={"target_id": str(run.target_id)},
            ),
        )
