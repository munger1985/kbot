"""Inbox 去重与 Outbox 至少一次投递 Repository。"""

from collections.abc import Callable, Collection
from datetime import datetime
from uuid import UUID

from sqlalchemy import Select, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.application.errors import StateConflictError
from aiops_agent.entities import InboxEntity, OutboxEntity
from aiops_agent.repositories._base import AIOpsRepository


class InboxRepository(AIOpsRepository):
    def __init__(
        self,
        session: AsyncSession,
        assert_active: Callable[[], None] | None = None,
    ):
        super().__init__(session, assert_active)

    async def add(self, entity: InboxEntity) -> InboxEntity:
        return await self._add(entity)

    async def get_by_message(
        self,
        *,
        source_system: str,
        message_key: str,
        lock: bool = False,
    ) -> InboxEntity | None:
        self._check_active()
        statement: Select = select(InboxEntity).where(
            InboxEntity.source_system == source_system,
            InboxEntity.message_key == message_key,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def transition(
        self,
        *,
        inbox_id: UUID,
        expected_version: int,
        allowed_statuses: Collection[str],
        new_status: str,
        processed_at: datetime | None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        self._check_active()
        statement = (
            update(InboxEntity)
            .where(
                InboxEntity.inbox_id == inbox_id,
                InboxEntity.row_version == expected_version,
                InboxEntity.status.in_(allowed_statuses),
            )
            .values(
                status=new_status,
                processed_at=processed_at,
                error_code=error_code,
                error_message=error_message,
                row_version=InboxEntity.row_version + 1,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1


class OutboxRepository(AIOpsRepository):
    def __init__(
        self,
        session: AsyncSession,
        assert_active: Callable[[], None] | None = None,
    ):
        super().__init__(session, assert_active)

    async def add(self, entity: OutboxEntity) -> OutboxEntity:
        return await self._add(entity)

    async def get_by_idempotency(
        self,
        *,
        idempotency_key: str,
    ) -> OutboxEntity | None:
        self._check_active()
        statement = select(OutboxEntity).where(
            OutboxEntity.idempotency_key == idempotency_key
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def claim(
        self,
        *,
        now: datetime,
        lease_owner: str,
        lease_token: UUID,
        lease_until: datetime,
    ) -> OutboxEntity | None:
        claimed_id = await self._claim_oracle_uuid(
            plsql="""
                DECLARE
                    CURSOR c_claim IS
                        SELECT OUTBOX_ID
                        FROM KBOT_OPS_OUTBOX
                        WHERE STATUS IN ('PENDING', 'RETRY_WAIT')
                          AND AVAILABLE_AT <= SYSTIMESTAMP
                          AND ATTEMPT_COUNT < MAX_ATTEMPTS
                        ORDER BY AVAILABLE_AT, OUTBOX_ID
                        FOR UPDATE OF OUTBOX_ID SKIP LOCKED;
                BEGIN
                    :claimed_id := NULL;
                    OPEN c_claim;
                    FETCH c_claim INTO :claimed_id;
                    CLOSE c_claim;
                END;
            """,
            parameters={},
        )
        if claimed_id is None:
            return None
        entity = (
            await self._session.execute(
                select(OutboxEntity).where(
                    OutboxEntity.outbox_id == claimed_id
                )
            )
        ).scalar_one_or_none()
        if entity is None:
            raise StateConflictError(
                f"领取后的 Outbox 不存在：{claimed_id}"
            )
        entity.status = "PUBLISHING"
        entity.lease_owner = lease_owner
        entity.lease_token = lease_token
        entity.lease_until = lease_until
        entity.attempt_count = int(entity.attempt_count) + 1
        await self._session.flush()
        return entity

    async def mark_published(
        self,
        *,
        outbox_id: UUID,
        lease_owner: str,
        lease_token: UUID,
        now: datetime,
    ) -> bool:
        self._check_active()
        statement = (
            update(OutboxEntity)
            .where(
                OutboxEntity.outbox_id == outbox_id,
                OutboxEntity.status == "PUBLISHING",
                OutboxEntity.lease_owner == lease_owner,
                OutboxEntity.lease_token == lease_token,
                OutboxEntity.lease_until > now,
            )
            .values(
                status="PUBLISHED",
                published_at=now,
                lease_owner=None,
                lease_token=None,
                lease_until=None,
                error_code=None,
                error_message=None,
                updated_at=now,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def release_failed(
        self,
        *,
        outbox_id: UUID,
        lease_owner: str,
        lease_token: UUID,
        now: datetime,
        new_status: str,
        available_at: datetime,
        error_code: str,
        error_message: str,
    ) -> bool:
        self._check_active()
        statement = (
            update(OutboxEntity)
            .where(
                OutboxEntity.outbox_id == outbox_id,
                OutboxEntity.status == "PUBLISHING",
                OutboxEntity.lease_owner == lease_owner,
                OutboxEntity.lease_token == lease_token,
                OutboxEntity.lease_until > now,
            )
            .values(
                status=new_status,
                available_at=available_at,
                lease_owner=None,
                lease_token=None,
                lease_until=None,
                error_code=error_code,
                error_message=error_message,
                updated_at=now,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def recover_expired(
        self,
        *,
        now: datetime,
        available_at: datetime,
    ) -> bool:
        """回收崩溃 Dispatcher 遗留的过期发布租约。"""
        self._check_active()
        claimed_id = await self._claim_oracle_uuid(
            plsql="""
                DECLARE
                    CURSOR c_claim IS
                        SELECT OUTBOX_ID
                        FROM KBOT_OPS_OUTBOX
                        WHERE STATUS = 'PUBLISHING'
                          AND LEASE_UNTIL <= SYSTIMESTAMP
                        ORDER BY LEASE_UNTIL, OUTBOX_ID
                        FOR UPDATE OF OUTBOX_ID SKIP LOCKED;
                BEGIN
                    :claimed_id := NULL;
                    OPEN c_claim;
                    FETCH c_claim INTO :claimed_id;
                    CLOSE c_claim;
                END;
            """,
            parameters={},
        )
        if claimed_id is None:
            return False
        entity = (
            await self._session.execute(
                select(OutboxEntity).where(
                    OutboxEntity.outbox_id == claimed_id
                )
            )
        ).scalar_one()
        entity.status = (
            "RETRY_WAIT"
            if int(entity.attempt_count) < int(entity.max_attempts)
            else "FAILED"
        )
        entity.available_at = available_at
        entity.lease_owner = None
        entity.lease_token = None
        entity.lease_until = None
        entity.error_code = "OUTBOX_LEASE_EXPIRED"
        entity.error_message = "Dispatcher 发布租约过期"
        entity.updated_at = now
        await self._session.flush()
        return True
