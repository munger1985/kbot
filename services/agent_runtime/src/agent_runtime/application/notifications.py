"""Agent Runtime 事务内通知 Outbox 发布器。"""

from typing import Protocol

from platform_core.notifications import NotificationEnvelope, publish_notification


class AgentRunNotificationPublisher(Protocol):
    async def publish(
        self, *, uow, run, event_type: str, actor_id: str,
        payload: dict[str, object],
    ) -> None: ...


class AgentRunOutboxPublisher:
    """仅写安全摘要，Run 与 Outbox 由同一事务提交。"""

    async def publish(
        self, *, uow, run, event_type: str, actor_id: str,
        payload: dict[str, object],
    ) -> None:
        del actor_id
        initiator = str(run.actor_id)
        error_code = str(run.error_code or payload.get("error_code") or "")
        summary = {
            "agent.run.completed": "Agent 运行已完成。",
            "agent.run.failed": "Agent 运行失败，请查看运行详情。",
            "agent.run.input_required": "Agent 运行等待补充输入。",
        }[event_type]
        await publish_notification(
            uow=uow,
            producer_service="agent-runtime",
            event_key=f"{run.run_id}:{event_type}",
            envelope=NotificationEnvelope(
                domain_id=int(run.domain_id),
                event_type=event_type,
                resource_type="agent_run",
                resource_id=str(run.run_id),
                resource_name=str(run.run_id),
                initiator_actor_id=initiator,
                recipient_actor_ids=[initiator],
                summary=summary,
                correlation_id=str(run.trace_id or run.request_id),
                operation_id=str(run.run_id),
                safe_data={
                    "status": str(run.status),
                    "error_code": error_code or None,
                },
            ),
        )
