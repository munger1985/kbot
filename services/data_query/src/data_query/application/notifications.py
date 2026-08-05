"""Data Query 事务内通知 Outbox 发布辅助。"""

from platform_core.notifications import NotificationEnvelope, publish_notification


async def publish_data_query_notification(
    *, uow, event_type: str, event_key: str,
    domain_id: int, actor_id: str | None,
    resource_type: str, resource_id: str,
    resource_name: str | None, correlation_id: str,
    operation_id: str, summary: str,
    safe_data: dict[str, object] | None = None,
) -> None:
    actor = str(actor_id or "").strip() or None
    await publish_notification(
        uow=uow,
        producer_service="data-query",
        event_key=event_key,
        envelope=NotificationEnvelope(
            domain_id=domain_id,
            event_type=event_type,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            initiator_actor_id=actor,
            recipient_actor_ids=[actor] if actor else [],
            summary=summary,
            correlation_id=correlation_id,
            operation_id=operation_id,
            safe_data=safe_data or {},
        ),
    )
