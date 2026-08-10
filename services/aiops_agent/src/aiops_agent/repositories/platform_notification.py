"""AIOps 平台通知 Outbox 持久化。"""

from platform_core.notifications import NotificationEnvelope, publish_notification


class PlatformNotificationRepository:
    def __init__(self, uow):
        self._uow = uow

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
        return await publish_notification(
            uow=self._uow,
            producer_service="aiops",
            event_key=f"aiops-report:{report.report_id}",
            envelope=NotificationEnvelope(
                domain_id=int(run.domain_id),
                event_type="aiops.report.ready",
                resource_type="aiops_report",
                resource_id=str(report.report_id),
                resource_name="AIOps 报告",
                initiator_actor_id=run.actor_id,
                recipient_actor_ids=[actor_id],
                summary=report.summary or "报告已生成",
                correlation_id=run.trace_id,
                operation_id=str(run.ops_run_id),
            ),
        )
