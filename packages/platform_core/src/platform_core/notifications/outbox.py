"""业务事务内写入通知 Outbox 的共享 Repository 与 Publisher。"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .catalog import event_definition
from .entities import NotificationOutboxEntity
from .models import NotificationEnvelope


class NotificationOutboxRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(
        self,
        *,
        producer_service: str,
        event_key: str,
        envelope: NotificationEnvelope,
    ) -> NotificationOutboxEntity:
        definition = event_definition(envelope.event_type)
        if definition.producer_service != producer_service:
            raise ValueError("NOTIFICATION_EVENT_PRODUCER_INVALID")
        existing = (
            await self._session.execute(
                select(NotificationOutboxEntity).where(
                    NotificationOutboxEntity.producer_service == producer_service,
                    NotificationOutboxEntity.event_key == event_key,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            previous = dict(existing.payload_json or {})
            if (
                existing.event_type != envelope.event_type
                or int(existing.domain_id) != envelope.domain_id
                or previous.get("resource_type") != envelope.resource_type
                or previous.get("resource_id") != envelope.resource_id
            ):
                raise ValueError("NOTIFICATION_EVENT_KEY_REUSED")
            return existing
        entity = NotificationOutboxEntity(
            producer_service=producer_service,
            event_key=event_key,
            event_type=envelope.event_type,
            event_version=envelope.event_version,
            domain_id=envelope.domain_id,
            payload_json=envelope.model_dump(mode="json"),
        )
        self._session.add(entity)
        await self._session.flush()
        return entity


async def publish_notification(
    *,
    uow,
    producer_service: str,
    event_key: str,
    envelope: NotificationEnvelope,
) -> NotificationOutboxEntity:
    repository = getattr(uow, "notification_outbox", None)
    if repository is None:
        raise RuntimeError("业务 UoW 未初始化 Notification Outbox")
    return await repository.add(
        producer_service=producer_service,
        event_key=event_key,
        envelope=envelope,
    )
