"""Conversation 与 Memory 的持久化 Repository。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Select, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_runtime.entities import (
    AgentConversationEntity,
    AgentConversationItemEntity,
    AgentConversationTurnEntity,
    AgentMemoryItemEntity,
    AgentMemoryIndexProfileEntity,
    AgentMemoryJobEntity,
    AgentMemorySnapshotEntity,
    AgentMemorySourceEntity,
)


class AgentConversationRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(
        self, entity: AgentConversationEntity
    ) -> AgentConversationEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def get_scoped(
        self,
        *,
        conversation_id: UUID,
        app_id: int,
        domain_id: int,
        actor_id: str,
        lock: bool = False,
    ) -> AgentConversationEntity | None:
        statement: Select = select(AgentConversationEntity).where(
            AgentConversationEntity.conversation_id == conversation_id,
            AgentConversationEntity.app_id == app_id,
            AgentConversationEntity.domain_id == domain_id,
            AgentConversationEntity.actor_id == actor_id,
            AgentConversationEntity.status != "DELETED",
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get(
        self, *, conversation_id: UUID, lock: bool = False
    ) -> AgentConversationEntity | None:
        statement: Select = select(AgentConversationEntity).where(
            AgentConversationEntity.conversation_id == conversation_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_scoped(
        self,
        *,
        app_id: int,
        domain_id: int,
        actor_id: str,
        limit: int,
    ) -> list[AgentConversationEntity]:
        statement = (
            select(AgentConversationEntity)
            .where(
                AgentConversationEntity.app_id == app_id,
                AgentConversationEntity.domain_id == domain_id,
                AgentConversationEntity.actor_id == actor_id,
                AgentConversationEntity.status != "DELETED",
            )
            .order_by(
                AgentConversationEntity.last_active_at.desc(),
                AgentConversationEntity.conversation_id.desc(),
            )
            .limit(limit)
        )
        return list((await self._session.execute(statement)).scalars())

    async def remove(self, entity: AgentConversationEntity) -> None:
        await self._session.delete(entity)
        await self._session.flush()

    async def claim_due_purge(
        self, *, now: datetime
    ) -> AgentConversationEntity | None:
        eligibility = (
            AgentConversationEntity.status == "ARCHIVED",
            AgentConversationEntity.purge_after.is_not(None),
            AgentConversationEntity.purge_after <= now,
        )
        candidate_ids = list(
            (
                await self._session.execute(
                    select(AgentConversationEntity.conversation_id)
                    .where(*eligibility)
                    .order_by(
                        AgentConversationEntity.purge_after,
                        AgentConversationEntity.conversation_id,
                    )
                    .limit(20)
                )
            ).scalars()
        )
        for conversation_id in candidate_ids:
            statement = (
                select(AgentConversationEntity)
                .where(
                    AgentConversationEntity.conversation_id
                    == conversation_id,
                    *eligibility,
                )
                .with_for_update(skip_locked=True)
            )
            row = (
                await self._session.execute(statement)
            ).scalar_one_or_none()
            if row is not None:
                return row
        return None


class AgentConversationTurnRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(
        self, entity: AgentConversationTurnEntity
    ) -> AgentConversationTurnEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def get(
        self, *, turn_id: UUID, lock: bool = False
    ) -> AgentConversationTurnEntity | None:
        statement: Select = select(AgentConversationTurnEntity).where(
            AgentConversationTurnEntity.turn_id == turn_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_by_run(
        self, *, run_id: UUID, lock: bool = False
    ) -> AgentConversationTurnEntity | None:
        statement: Select = select(AgentConversationTurnEntity).where(
            AgentConversationTurnEntity.root_run_id == run_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_by_idempotency(
        self, *, conversation_id: UUID, idempotency_key: str
    ) -> AgentConversationTurnEntity | None:
        statement = select(AgentConversationTurnEntity).where(
            AgentConversationTurnEntity.conversation_id == conversation_id,
            AgentConversationTurnEntity.idempotency_key == idempotency_key,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def find_active(
        self, *, conversation_id: UUID
    ) -> AgentConversationTurnEntity | None:
        statement = select(AgentConversationTurnEntity).where(
            AgentConversationTurnEntity.conversation_id == conversation_id,
            AgentConversationTurnEntity.status.in_(
                ("ACCEPTED", "RUNNING", "WAITING")
            ),
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_by_conversation(
        self,
        *,
        conversation_id: UUID,
        after_sequence: int,
        limit: int,
    ) -> list[AgentConversationTurnEntity]:
        statement = (
            select(AgentConversationTurnEntity)
            .where(
                AgentConversationTurnEntity.conversation_id
                == conversation_id,
                AgentConversationTurnEntity.turn_sequence > after_sequence,
            )
            .order_by(AgentConversationTurnEntity.turn_sequence)
            .limit(limit)
        )
        return list((await self._session.execute(statement)).scalars())

    async def delete_by_conversation(
        self, *, conversation_id: UUID
    ) -> None:
        await self._session.execute(
            delete(AgentConversationTurnEntity).where(
                AgentConversationTurnEntity.conversation_id
                == conversation_id
            )
        )


class AgentConversationItemRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(
        self, entity: AgentConversationItemEntity
    ) -> AgentConversationItemEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def list_by_turn_ids(
        self, *, turn_ids: list[UUID]
    ) -> list[AgentConversationItemEntity]:
        if not turn_ids:
            return []
        statement = (
            select(AgentConversationItemEntity)
            .where(AgentConversationItemEntity.turn_id.in_(turn_ids))
            .order_by(
                AgentConversationItemEntity.item_sequence,
                AgentConversationItemEntity.item_id,
            )
        )
        return list((await self._session.execute(statement)).scalars())

    async def list_recent(
        self, *, conversation_id: UUID, limit: int
    ) -> list[AgentConversationItemEntity]:
        statement = (
            select(AgentConversationItemEntity)
            .where(
                AgentConversationItemEntity.conversation_id
                == conversation_id,
                AgentConversationItemEntity.visibility == "USER",
                AgentConversationItemEntity.item_type == "MESSAGE",
            )
            .order_by(AgentConversationItemEntity.item_sequence.desc())
            .limit(limit)
        )
        rows = list((await self._session.execute(statement)).scalars())
        return list(reversed(rows))

    async def delete_by_conversation(
        self, *, conversation_id: UUID
    ) -> None:
        await self._session.execute(
            delete(AgentConversationItemEntity).where(
                AgentConversationItemEntity.conversation_id
                == conversation_id
            )
        )


class AgentMemorySnapshotRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(
        self, entity: AgentMemorySnapshotEntity
    ) -> AgentMemorySnapshotEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def get_active(
        self, *, conversation_id: UUID
    ) -> AgentMemorySnapshotEntity | None:
        statement = select(AgentMemorySnapshotEntity).where(
            AgentMemorySnapshotEntity.conversation_id == conversation_id,
            AgentMemorySnapshotEntity.status == "ACTIVE",
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def supersede_active(
        self, *, conversation_id: UUID
    ) -> None:
        row = await self.get_active(conversation_id=conversation_id)
        if row is not None:
            row.status = "SUPERSEDED"
            await self._session.flush()

    async def delete_by_conversation(
        self, *, conversation_id: UUID
    ) -> None:
        await self._session.execute(
            delete(AgentMemorySnapshotEntity).where(
                AgentMemorySnapshotEntity.conversation_id
                == conversation_id
            )
        )


class AgentMemoryItemRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(
        self, entity: AgentMemoryItemEntity
    ) -> AgentMemoryItemEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def get_scoped(
        self,
        *,
        memory_id: UUID,
        app_id: int,
        domain_id: int,
        actor_id: str,
        lock: bool = False,
    ) -> AgentMemoryItemEntity | None:
        statement: Select = select(AgentMemoryItemEntity).where(
            AgentMemoryItemEntity.memory_id == memory_id,
            AgentMemoryItemEntity.app_id == app_id,
            AgentMemoryItemEntity.domain_id == domain_id,
            AgentMemoryItemEntity.actor_id == actor_id,
            AgentMemoryItemEntity.status != "DELETED",
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_active(
        self,
        *,
        app_id: int,
        domain_id: int,
        actor_id: str,
        agent_id: UUID,
        now: datetime,
        limit: int,
    ) -> list[AgentMemoryItemEntity]:
        statement = (
            select(AgentMemoryItemEntity)
            .where(
                AgentMemoryItemEntity.app_id == app_id,
                AgentMemoryItemEntity.domain_id == domain_id,
                AgentMemoryItemEntity.actor_id == actor_id,
                AgentMemoryItemEntity.status == "ACTIVE",
                or_(
                    AgentMemoryItemEntity.expires_at.is_(None),
                    AgentMemoryItemEntity.expires_at > now,
                ),
                or_(
                    AgentMemoryItemEntity.agent_id == agent_id,
                    AgentMemoryItemEntity.agent_id.is_(None),
                ),
            )
            .order_by(
                AgentMemoryItemEntity.salience.desc(),
                AgentMemoryItemEntity.updated_at.desc(),
            )
            .limit(limit)
        )
        return list((await self._session.execute(statement)).scalars())

    async def get_active_by_key(
        self,
        *,
        app_id: int,
        domain_id: int,
        actor_id: str,
        agent_id: UUID | None,
        memory_type: str,
        canonical_key: str,
        lock: bool = False,
    ) -> AgentMemoryItemEntity | None:
        agent_clause = (
            AgentMemoryItemEntity.agent_id == agent_id
            if agent_id is not None
            else AgentMemoryItemEntity.agent_id.is_(None)
        )
        statement: Select = select(AgentMemoryItemEntity).where(
            AgentMemoryItemEntity.app_id == app_id,
            AgentMemoryItemEntity.domain_id == domain_id,
            AgentMemoryItemEntity.actor_id == actor_id,
            agent_clause,
            AgentMemoryItemEntity.memory_type == memory_type,
            AgentMemoryItemEntity.canonical_key == canonical_key,
            AgentMemoryItemEntity.status == "ACTIVE",
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_active_by_canonical_key(
        self,
        *,
        app_id: int,
        domain_id: int,
        actor_id: str,
        agent_id: UUID | None,
        canonical_key: str,
        lock: bool = False,
    ) -> list[AgentMemoryItemEntity]:
        agent_clause = (
            AgentMemoryItemEntity.agent_id == agent_id
            if agent_id is not None
            else AgentMemoryItemEntity.agent_id.is_(None)
        )
        statement: Select = select(AgentMemoryItemEntity).where(
            AgentMemoryItemEntity.app_id == app_id,
            AgentMemoryItemEntity.domain_id == domain_id,
            AgentMemoryItemEntity.actor_id == actor_id,
            agent_clause,
            AgentMemoryItemEntity.canonical_key == canonical_key,
            AgentMemoryItemEntity.status == "ACTIVE",
        )
        if lock:
            statement = statement.with_for_update()
        return list((await self._session.execute(statement)).scalars())


class AgentMemoryIndexProfileRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(
        self, entity: AgentMemoryIndexProfileEntity
    ) -> AgentMemoryIndexProfileEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def get(
        self,
        *,
        app_id: int,
        domain_id: int,
        agent_id: UUID,
        lock: bool = False,
    ) -> AgentMemoryIndexProfileEntity | None:
        statement: Select = select(
            AgentMemoryIndexProfileEntity
        ).where(
            AgentMemoryIndexProfileEntity.app_id == app_id,
            AgentMemoryIndexProfileEntity.domain_id == domain_id,
            AgentMemoryIndexProfileEntity.agent_id == agent_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

class AgentMemorySourceRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(
        self, entity: AgentMemorySourceEntity
    ) -> AgentMemorySourceEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def memory_ids_for_conversation(
        self, *, conversation_id: UUID
    ) -> list[UUID]:
        statement = (
            select(AgentMemorySourceEntity.memory_id)
            .where(
                AgentMemorySourceEntity.conversation_id
                == conversation_id
            )
            .distinct()
        )
        return list((await self._session.execute(statement)).scalars())

    async def delete_by_conversation(
        self, *, conversation_id: UUID
    ) -> None:
        await self._session.execute(
            delete(AgentMemorySourceEntity).where(
                AgentMemorySourceEntity.conversation_id
                == conversation_id
            )
        )

    async def delete_by_memory(self, *, memory_id: UUID) -> None:
        await self._session.execute(
            delete(AgentMemorySourceEntity).where(
                AgentMemorySourceEntity.memory_id == memory_id
            )
        )

    async def count_by_memory(self, *, memory_id: UUID) -> int:
        statement = select(func.count()).select_from(
            AgentMemorySourceEntity
        ).where(AgentMemorySourceEntity.memory_id == memory_id)
        return int((await self._session.execute(statement)).scalar_one())


class AgentMemoryJobRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(
        self, entity: AgentMemoryJobEntity
    ) -> AgentMemoryJobEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def claim(
        self,
        *,
        worker_id: str,
        lease_token: UUID,
        now: datetime,
        lease_until: datetime,
    ) -> AgentMemoryJobEntity | None:
        eligibility = (
            AgentMemoryJobEntity.status.in_(
                ("PENDING", "RETRY_WAIT", "PROCESSING")
            ),
            AgentMemoryJobEntity.next_attempt_at <= now,
            or_(
                AgentMemoryJobEntity.status != "PROCESSING",
                AgentMemoryJobEntity.lease_until < now,
            ),
            AgentMemoryJobEntity.attempt_count
            < AgentMemoryJobEntity.max_attempts,
        )
        candidate_ids = list(
            (
                await self._session.execute(
                    select(AgentMemoryJobEntity.memory_job_id)
                    .where(*eligibility)
                    .order_by(
                        AgentMemoryJobEntity.next_attempt_at,
                        AgentMemoryJobEntity.created_at,
                    )
                    .limit(20)
                )
            ).scalars()
        )
        row = None
        for memory_job_id in candidate_ids:
            row = (
                await self._session.execute(
                    select(AgentMemoryJobEntity)
                    .where(
                        AgentMemoryJobEntity.memory_job_id
                        == memory_job_id,
                        *eligibility,
                    )
                    .with_for_update(skip_locked=True)
                )
            ).scalar_one_or_none()
            if row is not None:
                break
        if row is None:
            return None
        row.status = "PROCESSING"
        row.attempt_count = int(row.attempt_count) + 1
        row.lease_owner = worker_id
        row.lease_token = lease_token
        row.lease_until = lease_until
        row.error_code = None
        row.error_message = None
        await self._session.flush()
        return row

    async def get(
        self, *, memory_job_id: UUID, lock: bool = False
    ) -> AgentMemoryJobEntity | None:
        statement: Select = select(AgentMemoryJobEntity).where(
            AgentMemoryJobEntity.memory_job_id == memory_job_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_by_turn_ids(
        self, *, turn_ids: list[UUID]
    ) -> list[AgentMemoryJobEntity]:
        if not turn_ids:
            return []
        statement = select(AgentMemoryJobEntity).where(
            AgentMemoryJobEntity.turn_id.in_(turn_ids)
        )
        return list((await self._session.execute(statement)).scalars())

    async def has_processing(
        self, *, conversation_id: UUID
    ) -> bool:
        statement = select(func.count()).select_from(
            AgentMemoryJobEntity
        ).where(
            AgentMemoryJobEntity.conversation_id == conversation_id,
            AgentMemoryJobEntity.status == "PROCESSING",
        )
        return bool((await self._session.execute(statement)).scalar_one())

    async def delete_by_conversation(
        self, *, conversation_id: UUID
    ) -> None:
        await self._session.execute(
            delete(AgentMemoryJobEntity).where(
                AgentMemoryJobEntity.conversation_id == conversation_id
            )
        )
