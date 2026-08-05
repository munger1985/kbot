"""通知 Outbox 投影和个人 Inbox Repository。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from main_api.entities import (
    BackgroundOperationEntity,
    NotificationInboxEntity,
    NotificationPreferenceEntity,
    OperationWatchEntity,
    WorkItemEntity,
)
from platform_core.identity import uuid7
from platform_core.notifications import NotificationOutboxEntity


def _now() -> datetime:
    return datetime.now(timezone.utc)


class NotificationRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def recover_expired(self, *, now: datetime) -> int:
        result = await self._session.execute(
            update(NotificationOutboxEntity)
            .where(
                NotificationOutboxEntity.status == "PROCESSING",
                NotificationOutboxEntity.lease_expires_at < now,
            )
            .values(
                status="PENDING", lease_token=None, lease_expires_at=None,
                available_at=now, updated_at=now,
            )
        )
        return int(result.rowcount or 0)

    async def claim_batch(
        self, *, limit: int, lease_seconds: int,
    ) -> list[tuple[UUID, UUID]]:
        now = _now()
        # Oracle 不支持 FETCH FIRST 与 FOR UPDATE 同时作用于同一查询块。
        # 内层只负责挑选候选主键，外层仍然直接锁定基表并跳过其他 Worker 已锁行。
        candidate_ids = (
            select(NotificationOutboxEntity.outbox_id)
            .where(
                NotificationOutboxEntity.status == "PENDING",
                NotificationOutboxEntity.available_at <= now,
            )
            .order_by(
                NotificationOutboxEntity.available_at,
                NotificationOutboxEntity.created_at,
            )
            .limit(limit)
        )
        rows = (
            await self._session.execute(
                select(NotificationOutboxEntity)
                .where(
                    NotificationOutboxEntity.status == "PENDING",
                    NotificationOutboxEntity.available_at <= now,
                    NotificationOutboxEntity.outbox_id.in_(candidate_ids),
                )
                .order_by(
                    NotificationOutboxEntity.available_at,
                    NotificationOutboxEntity.created_at,
                )
                .with_for_update(skip_locked=True)
            )
        ).scalars().all()
        claimed: list[tuple[UUID, UUID]] = []
        for row in rows:
            token = uuid7()
            row.status = "PROCESSING"
            row.lease_token = token
            row.lease_expires_at = now + timedelta(seconds=lease_seconds)
            row.attempt_count = int(row.attempt_count) + 1
            row.updated_at = now
            claimed.append((row.outbox_id, token))
        await self._session.flush()
        return claimed

    async def claimed(
        self, *, outbox_id: UUID, lease_token: UUID,
    ) -> NotificationOutboxEntity | None:
        return (
            await self._session.execute(
                select(NotificationOutboxEntity)
                .where(
                    NotificationOutboxEntity.outbox_id == outbox_id,
                    NotificationOutboxEntity.status == "PROCESSING",
                    NotificationOutboxEntity.lease_token == lease_token,
                )
                .with_for_update()
            )
        ).scalar_one_or_none()

    async def mark_published(self, row: NotificationOutboxEntity) -> None:
        now = _now()
        row.status = "PUBLISHED"
        row.published_at = now
        row.lease_token = None
        row.lease_expires_at = None
        row.last_error_code = None
        row.updated_at = now
        await self._session.flush()

    async def release_failed(
        self, *, outbox_id: UUID, lease_token: UUID,
        error_code: str, max_attempts: int,
    ) -> str | None:
        row = await self.claimed(outbox_id=outbox_id, lease_token=lease_token)
        if row is None:
            return None
        now = _now()
        quarantined = int(row.attempt_count) >= max_attempts
        row.status = "QUARANTINED" if quarantined else "PENDING"
        row.available_at = now + timedelta(
            seconds=min(2 ** int(row.attempt_count), 300)
        )
        row.lease_token = None
        row.lease_expires_at = None
        row.last_error_code = error_code[:128]
        row.updated_at = now
        await self._session.flush()
        return row.status

    async def operation_for_update(
        self, *, producer: str, source_operation_id: str,
    ) -> BackgroundOperationEntity | None:
        return (
            await self._session.execute(
                select(BackgroundOperationEntity)
                .where(
                    BackgroundOperationEntity.producer_service == producer,
                    BackgroundOperationEntity.source_operation_id == source_operation_id,
                )
                .with_for_update()
            )
        ).scalar_one_or_none()

    async def add_operation(self, entity: BackgroundOperationEntity) -> None:
        self._session.add(entity)
        await self._session.flush()

    async def watcher_actor_ids(
        self, *, operation_id: UUID, domain_id: int,
    ) -> set[str]:
        return set((
            await self._session.execute(
                select(OperationWatchEntity.actor_id).where(
                    OperationWatchEntity.operation_id == operation_id,
                    OperationWatchEntity.domain_id == domain_id,
                    OperationWatchEntity.notify_terminal == 1,
                )
            )
        ).scalars().all())

    async def preference_enabled(
        self, *, domain_id: int, actor_id: str, event_type: str,
    ) -> bool:
        enabled = (
            await self._session.execute(
                select(NotificationPreferenceEntity.enabled).where(
                    NotificationPreferenceEntity.domain_id == domain_id,
                    NotificationPreferenceEntity.actor_id == actor_id,
                    NotificationPreferenceEntity.event_type == event_type,
                )
            )
        ).scalar_one_or_none()
        return enabled is None or int(enabled) == 1

    async def inbox_exists(self, *, outbox_id: UUID, actor_id: str) -> bool:
        return (
            await self._session.execute(
                select(NotificationInboxEntity.inbox_id).where(
                    NotificationInboxEntity.outbox_id == outbox_id,
                    NotificationInboxEntity.recipient_actor_id == actor_id,
                )
            )
        ).scalar_one_or_none() is not None

    async def add_inbox(self, entity: NotificationInboxEntity) -> None:
        self._session.add(entity)
        await self._session.flush()

    async def work_item_for_update(
        self, *, domain_id: int, actor_id: str, resource_type: str,
        resource_id: str, action_type: str,
    ) -> WorkItemEntity | None:
        # 单个资源的待办历史很短；不使用 LIMIT，避免 Oracle 生成
        # FETCH FIRST ... FOR UPDATE 这种不可更新查询块。
        return (
            await self._session.execute(
                select(WorkItemEntity)
                .where(
                    WorkItemEntity.domain_id == domain_id,
                    WorkItemEntity.actor_id == actor_id,
                    WorkItemEntity.resource_type == resource_type,
                    WorkItemEntity.resource_id == resource_id,
                    WorkItemEntity.action_type == action_type,
                )
                .order_by(WorkItemEntity.last_occurred_at.desc())
                .with_for_update()
            )
        ).scalars().first()

    async def add_work_item(self, entity: WorkItemEntity) -> None:
        self._session.add(entity)
        await self._session.flush()

    async def summary(self, *, domain_id: int, actor_id: str) -> dict[str, int]:
        unread = await self._session.scalar(
            select(func.count()).select_from(NotificationInboxEntity).where(
                NotificationInboxEntity.domain_id == domain_id,
                NotificationInboxEntity.recipient_actor_id == actor_id,
                NotificationInboxEntity.read_at.is_(None),
                NotificationInboxEntity.expires_at > _now(),
            )
        )
        work_items = await self._session.scalar(
            select(func.count()).select_from(WorkItemEntity).where(
                WorkItemEntity.domain_id == domain_id,
                WorkItemEntity.actor_id == actor_id,
                WorkItemEntity.status == "OPEN",
            )
        )
        active_operations = await self._session.scalar(
            select(func.count()).select_from(BackgroundOperationEntity).where(
                BackgroundOperationEntity.domain_id == domain_id,
                BackgroundOperationEntity.status.in_({"RUNNING", "WAITING_USER", "PARTIAL"}),
                or_(
                    BackgroundOperationEntity.initiator_actor_id == actor_id,
                    BackgroundOperationEntity.initiator_actor_id.is_(None),
                    BackgroundOperationEntity.operation_id.in_(
                        select(OperationWatchEntity.operation_id).where(
                            OperationWatchEntity.domain_id == domain_id,
                            OperationWatchEntity.actor_id == actor_id,
                        )
                    ),
                ),
            )
        )
        return {
            "unread": int(unread or 0),
            "open_work_items": int(work_items or 0),
            "active_operations": int(active_operations or 0),
        }

    async def list_inbox(
        self, *, domain_id: int, actor_id: str, limit: int,
        before_sequence: int | None = None,
    ) -> list[NotificationInboxEntity]:
        statement = select(NotificationInboxEntity).where(
            NotificationInboxEntity.domain_id == domain_id,
            NotificationInboxEntity.recipient_actor_id == actor_id,
            NotificationInboxEntity.expires_at > _now(),
        )
        if before_sequence is not None:
            statement = statement.where(
                NotificationInboxEntity.event_sequence < before_sequence
            )
        return list((
            await self._session.execute(
                statement.order_by(NotificationInboxEntity.event_sequence.desc()).limit(limit)
            )
        ).scalars().all())

    async def stream_inbox(
        self, *, domain_id: int, actor_id: str,
        after_sequence: int, limit: int,
    ) -> list[NotificationInboxEntity]:
        return list((
            await self._session.execute(
                select(NotificationInboxEntity)
                .where(
                    NotificationInboxEntity.domain_id == domain_id,
                    NotificationInboxEntity.recipient_actor_id == actor_id,
                    NotificationInboxEntity.event_sequence > after_sequence,
                    NotificationInboxEntity.expires_at > _now(),
                )
                .order_by(NotificationInboxEntity.event_sequence)
                .limit(limit)
            )
        ).scalars().all())

    async def list_resource_inbox(
        self, *, domain_id: int, actor_id: str, resource_type: str,
        resource_id: str, limit: int,
    ) -> list[NotificationInboxEntity]:
        """按资源读取当前 Actor 可见且尚未过期的通知。"""
        return list((
            await self._session.execute(
                select(NotificationInboxEntity)
                .where(
                    NotificationInboxEntity.domain_id == domain_id,
                    NotificationInboxEntity.recipient_actor_id == actor_id,
                    NotificationInboxEntity.resource_type == resource_type,
                    NotificationInboxEntity.resource_id == resource_id,
                    NotificationInboxEntity.expires_at > _now(),
                )
                .order_by(NotificationInboxEntity.event_sequence.desc())
                .limit(limit)
            )
        ).scalars().all())

    async def get_inbox(
        self, *, inbox_id: UUID, domain_id: int, actor_id: str,
        lock: bool = False,
    ) -> NotificationInboxEntity | None:
        statement = select(NotificationInboxEntity).where(
            NotificationInboxEntity.inbox_id == inbox_id,
            NotificationInboxEntity.domain_id == domain_id,
            NotificationInboxEntity.recipient_actor_id == actor_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def mark_many_read(
        self, *, inbox_ids: list[UUID], domain_id: int, actor_id: str,
    ) -> int:
        result = await self._session.execute(
            update(NotificationInboxEntity)
            .where(
                NotificationInboxEntity.inbox_id.in_(inbox_ids),
                NotificationInboxEntity.domain_id == domain_id,
                NotificationInboxEntity.recipient_actor_id == actor_id,
                NotificationInboxEntity.read_at.is_(None),
            )
            .values(
                read_at=_now(),
                row_version=NotificationInboxEntity.row_version + 1,
            )
        )
        return int(result.rowcount or 0)

    async def list_work_items(
        self, *, domain_id: int, actor_id: str, status: str, limit: int,
    ) -> list[WorkItemEntity]:
        return list((
            await self._session.execute(
                select(WorkItemEntity)
                .where(
                    WorkItemEntity.domain_id == domain_id,
                    WorkItemEntity.actor_id == actor_id,
                    WorkItemEntity.status == status,
                )
                .order_by(WorkItemEntity.updated_at.desc())
                .limit(limit)
            )
        ).scalars().all())

    async def list_operations(
        self, *, domain_id: int, actor_id: str, limit: int,
    ) -> list[BackgroundOperationEntity]:
        return list((
            await self._session.execute(
                select(BackgroundOperationEntity)
                .where(
                    BackgroundOperationEntity.domain_id == domain_id,
                    or_(
                        BackgroundOperationEntity.initiator_actor_id == actor_id,
                        BackgroundOperationEntity.initiator_actor_id.is_(None),
                        BackgroundOperationEntity.operation_id.in_(
                            select(OperationWatchEntity.operation_id).where(
                                OperationWatchEntity.domain_id == domain_id,
                                OperationWatchEntity.actor_id == actor_id,
                            )
                        ),
                    ),
                )
                .order_by(BackgroundOperationEntity.updated_at.desc())
                .limit(limit)
            )
        ).scalars().all())

    async def get_operation(
        self, *, operation_id: UUID, domain_id: int,
    ) -> BackgroundOperationEntity | None:
        return (
            await self._session.execute(
                select(BackgroundOperationEntity).where(
                    BackgroundOperationEntity.operation_id == operation_id,
                    BackgroundOperationEntity.domain_id == domain_id,
                )
            )
        ).scalar_one_or_none()

    async def watch(
        self, *, operation_id: UUID, domain_id: int,
        actor_id: str, notify_terminal: bool,
    ) -> None:
        existing = await self._session.get(
            OperationWatchEntity,
            {"operation_id": operation_id, "domain_id": domain_id, "actor_id": actor_id},
        )
        if existing is None:
            self._session.add(OperationWatchEntity(
                operation_id=operation_id, domain_id=domain_id,
                actor_id=actor_id, notify_terminal=int(notify_terminal),
            ))
        else:
            existing.notify_terminal = int(notify_terminal)
        await self._session.flush()

    async def unwatch(
        self, *, operation_id: UUID, domain_id: int, actor_id: str,
    ) -> int:
        result = await self._session.execute(
            delete(OperationWatchEntity).where(
                OperationWatchEntity.operation_id == operation_id,
                OperationWatchEntity.domain_id == domain_id,
                OperationWatchEntity.actor_id == actor_id,
            )
        )
        return int(result.rowcount or 0)

    async def preferences(
        self, *, domain_id: int, actor_id: str,
    ) -> list[NotificationPreferenceEntity]:
        return list((await self._session.execute(
            select(NotificationPreferenceEntity).where(
                NotificationPreferenceEntity.domain_id == domain_id,
                NotificationPreferenceEntity.actor_id == actor_id,
            ).order_by(NotificationPreferenceEntity.event_type)
        )).scalars().all())

    async def set_preference(
        self, *, domain_id: int, actor_id: str,
        event_type: str, enabled: bool,
    ) -> None:
        entity = await self._session.get(
            NotificationPreferenceEntity,
            {"domain_id": domain_id, "actor_id": actor_id, "event_type": event_type},
        )
        if entity is None:
            self._session.add(NotificationPreferenceEntity(
                domain_id=domain_id, actor_id=actor_id,
                event_type=event_type, enabled=int(enabled),
            ))
        else:
            entity.enabled = int(enabled)
            entity.row_version = int(entity.row_version) + 1
        await self._session.flush()

    async def forget_actor(self, *, domain_id: int, actor_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name, entity, field in (
            ("preferences", NotificationPreferenceEntity, NotificationPreferenceEntity.actor_id),
            ("inbox", NotificationInboxEntity, NotificationInboxEntity.recipient_actor_id),
            ("work_items", WorkItemEntity, WorkItemEntity.actor_id),
            ("watches", OperationWatchEntity, OperationWatchEntity.actor_id),
        ):
            result = await self._session.execute(
                delete(entity).where(entity.domain_id == domain_id, field == actor_id)
            )
            counts[name] = int(result.rowcount or 0)
        return counts

    async def quarantine(self, *, domain_id: int, limit: int):
        return list((await self._session.execute(
            select(NotificationOutboxEntity).where(
                NotificationOutboxEntity.domain_id == domain_id,
                NotificationOutboxEntity.status == "QUARANTINED",
            ).order_by(NotificationOutboxEntity.updated_at.desc()).limit(limit)
        )).scalars().all())

    async def retry_quarantined(
        self, *, domain_id: int, outbox_id: UUID,
    ) -> bool:
        result = await self._session.execute(
            update(NotificationOutboxEntity).where(
                NotificationOutboxEntity.outbox_id == outbox_id,
                NotificationOutboxEntity.domain_id == domain_id,
                NotificationOutboxEntity.status == "QUARANTINED",
            ).values(
                status="PENDING", attempt_count=0, available_at=_now(),
                last_error_code=None, updated_at=_now(),
            )
        )
        return bool(result.rowcount)
