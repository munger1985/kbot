"""Slack Inbox、会话映射和出站投递 Repository。"""

from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from km_asset_app.entities import (
    SlackDeliveryEntity,
    SlackInboxEntity,
    SlackThreadEntity,
)


_CLAIM_SCAN_LIMIT = 32


class SlackIntegrationRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_inbox_by_event_id(
        self, event_id: str
    ) -> SlackInboxEntity | None:
        result = await self._session.execute(
            select(SlackInboxEntity).where(
                SlackInboxEntity.event_id == event_id
            )
        )
        return result.scalar_one_or_none()

    async def get_inbox_by_message_key(
        self, message_key: str
    ) -> SlackInboxEntity | None:
        result = await self._session.execute(
            select(SlackInboxEntity).where(
                SlackInboxEntity.message_key == message_key
            )
        )
        return result.scalar_one_or_none()

    async def get_inbox(self, inbox_id: UUID) -> SlackInboxEntity | None:
        return await self._session.get(SlackInboxEntity, inbox_id)

    async def add_inbox(self, entity: SlackInboxEntity) -> None:
        self._session.add(entity)
        await self._session.flush()

    async def claim_inbox(
        self, *, worker_id: str, lease_seconds: int
    ) -> SlackInboxEntity | None:
        now = datetime.now(UTC)
        eligibility = (
            SlackInboxEntity.status.in_(("RECEIVED", "RUNNING")),
            or_(
                SlackInboxEntity.lease_until.is_(None),
                SlackInboxEntity.lease_until < now,
            ),
        )
        # Oracle 禁止 FETCH FIRST 与 FOR UPDATE 作用于同一查询块。
        # 先读取少量候选主键，再逐个锁定基表行并跳过其他 Worker 已锁行。
        candidate_statement = (
            select(SlackInboxEntity.inbox_id)
            .where(*eligibility)
            .order_by(
                SlackInboxEntity.created_at,
                SlackInboxEntity.inbox_id,
            )
            .limit(_CLAIM_SCAN_LIMIT)
        )
        candidate_ids = list(
            (await self._session.execute(candidate_statement)).scalars()
        )
        for inbox_id in candidate_ids:
            lock_statement = (
                select(SlackInboxEntity)
                .where(
                    SlackInboxEntity.inbox_id == inbox_id,
                    *eligibility,
                )
                .with_for_update(skip_locked=True)
            )
            entity = (
                await self._session.execute(lock_statement)
            ).scalar_one_or_none()
            if entity is None:
                continue
            entity.lease_owner = worker_id
            entity.lease_until = now + timedelta(seconds=lease_seconds)
            entity.updated_at = now
            await self._session.flush()
            return entity
        return None

    async def get_thread(
        self,
        *,
        workspace_id: str,
        channel_id: str,
        root_thread_ts: str,
        slack_user_id: str,
    ) -> SlackThreadEntity | None:
        result = await self._session.execute(
            select(SlackThreadEntity).where(
                SlackThreadEntity.workspace_id == workspace_id,
                SlackThreadEntity.channel_id == channel_id,
                SlackThreadEntity.root_thread_ts == root_thread_ts,
                SlackThreadEntity.slack_user_id == slack_user_id,
            )
        )
        return result.scalar_one_or_none()

    async def add_thread(self, entity: SlackThreadEntity) -> None:
        self._session.add(entity)
        await self._session.flush()

    async def add_delivery(self, entity: SlackDeliveryEntity) -> None:
        self._session.add(entity)
        await self._session.flush()

    async def get_delivery(
        self, *, inbox_id: UUID, delivery_type: str
    ) -> SlackDeliveryEntity | None:
        result = await self._session.execute(
            select(SlackDeliveryEntity).where(
                SlackDeliveryEntity.inbox_id == inbox_id,
                SlackDeliveryEntity.delivery_type == delivery_type,
            )
        )
        return result.scalar_one_or_none()

    async def claim_delivery(
        self, *, worker_id: str, lease_seconds: int
    ) -> SlackDeliveryEntity | None:
        now = datetime.now(UTC)
        eligibility = (
            SlackDeliveryEntity.status == "PENDING",
            or_(
                SlackDeliveryEntity.next_attempt_at.is_(None),
                SlackDeliveryEntity.next_attempt_at <= now,
            ),
            or_(
                SlackDeliveryEntity.lease_until.is_(None),
                SlackDeliveryEntity.lease_until < now,
            ),
        )
        candidate_statement = (
            select(SlackDeliveryEntity.delivery_id)
            .where(*eligibility)
            .order_by(
                SlackDeliveryEntity.created_at,
                SlackDeliveryEntity.delivery_id,
            )
            .limit(_CLAIM_SCAN_LIMIT)
        )
        candidate_ids = list(
            (await self._session.execute(candidate_statement)).scalars()
        )
        for delivery_id in candidate_ids:
            lock_statement = (
                select(SlackDeliveryEntity)
                .where(
                    SlackDeliveryEntity.delivery_id == delivery_id,
                    *eligibility,
                )
                .with_for_update(skip_locked=True)
            )
            entity = (
                await self._session.execute(lock_statement)
            ).scalar_one_or_none()
            if entity is None:
                continue
            entity.lease_owner = worker_id
            entity.lease_until = now + timedelta(seconds=lease_seconds)
            entity.attempt_count += 1
            await self._session.flush()
            return entity
        return None
