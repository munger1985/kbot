"""仅访问 KBOT_AGENT_* 表的持久化 Repository。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Select, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from agent_runtime.entities import (
    AgentArtifactEntity,
    AgentDelegationEntity,
    AgentRunEntity,
    AgentRunEventEntity,
    AgentTaskEntity,
)


class AgentRunRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, entity: AgentRunEntity) -> AgentRunEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def get_by_idempotency(
        self,
        *,
        app_id: int,
        domain_id: int,
        actor_id: str,
        idempotency_key: str,
        lock: bool = False,
    ) -> AgentRunEntity | None:
        statement: Select = select(AgentRunEntity).where(
            AgentRunEntity.app_id == app_id,
            AgentRunEntity.domain_id == domain_id,
            AgentRunEntity.actor_id == actor_id,
            AgentRunEntity.idempotency_key == idempotency_key,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_scoped(
        self,
        *,
        run_id: UUID,
        app_id: int,
        domain_id: int,
        lock: bool = False,
    ) -> AgentRunEntity | None:
        statement: Select = select(AgentRunEntity).where(
            AgentRunEntity.run_id == run_id,
            AgentRunEntity.app_id == app_id,
            AgentRunEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get(
        self, *, run_id: UUID, lock: bool = False
    ) -> AgentRunEntity | None:
        statement: Select = select(AgentRunEntity).where(
            AgentRunEntity.run_id == run_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()


class AgentTaskRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, entity: AgentTaskEntity) -> AgentTaskEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def add_all(
        self, entities: list[AgentTaskEntity]
    ) -> list[AgentTaskEntity]:
        self._session.add_all(entities)
        await self._session.flush()
        return entities

    async def get(
        self, *, task_id: UUID, lock: bool = False
    ) -> AgentTaskEntity | None:
        statement: Select = select(AgentTaskEntity).where(
            AgentTaskEntity.task_id == task_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def claim_candidate(
        self, *, now: datetime, max_parallel_tasks: int
    ) -> AgentTaskEntity | None:
        """领取一个 READY Task；SKIP LOCKED 支持多 Worker 并发。"""
        running_task = aliased(AgentTaskEntity)
        running_count = (
            select(func.count())
            .select_from(running_task)
            .where(
                running_task.run_id == AgentTaskEntity.run_id,
                running_task.status == "RUNNING",
            )
            .scalar_subquery()
        )
        statement: Select = (
            select(AgentTaskEntity)
            .join(
                AgentRunEntity,
                AgentRunEntity.run_id == AgentTaskEntity.run_id,
            )
            .where(AgentTaskEntity.status == "READY")
            .where(AgentRunEntity.status == "RUNNING")
            .where(
                or_(
                    AgentRunEntity.deadline_at.is_(None),
                    AgentRunEntity.deadline_at > now,
                )
            )
            .where(running_count < max_parallel_tasks)
            .order_by(AgentTaskEntity.created_at, AgentTaskEntity.task_id)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_by_run(
        self, *, run_id: UUID, lock: bool = False
    ) -> list[AgentTaskEntity]:
        statement = (
            select(AgentTaskEntity)
            .where(AgentTaskEntity.run_id == run_id)
            .order_by(AgentTaskEntity.created_at, AgentTaskEntity.task_id)
        )
        if lock:
            statement = statement.with_for_update()
        return list((await self._session.execute(statement)).scalars())

    async def list_pending_dependents(
        self, *, run_id: UUID, completed_task_key: str
    ) -> list[AgentTaskEntity]:
        """锁定同一 Run 的 PENDING Task，由应用层判定全部依赖。"""
        statement = (
            select(AgentTaskEntity)
            .where(
                AgentTaskEntity.run_id == run_id,
                AgentTaskEntity.status == "PENDING",
            )
            .order_by(AgentTaskEntity.created_at, AgentTaskEntity.task_id)
            .with_for_update()
        )
        rows = list((await self._session.execute(statement)).scalars())
        return [
            task
            for task in rows
            if completed_task_key in (task.depends_on_json or [])
        ]


class AgentArtifactRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(
        self, entity: AgentArtifactEntity
    ) -> AgentArtifactEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def get(
        self, *, artifact_id: UUID
    ) -> AgentArtifactEntity | None:
        statement = select(AgentArtifactEntity).where(
            AgentArtifactEntity.artifact_id == artifact_id
        )
        return (await self._session.execute(statement)).scalar_one_or_none()


class AgentRunEventRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(
        self, entity: AgentRunEventEntity
    ) -> AgentRunEventEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def get_by_key(
        self, *, run_id: UUID, event_key: str
    ) -> AgentRunEventEntity | None:
        statement = select(AgentRunEventEntity).where(
            AgentRunEventEntity.run_id == run_id,
            AgentRunEventEntity.event_key == event_key,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def next_sequence(self, *, run_id: UUID) -> int:
        statement = select(
            func.coalesce(func.max(AgentRunEventEntity.sequence_no), 0)
        ).where(AgentRunEventEntity.run_id == run_id)
        current = (await self._session.execute(statement)).scalar_one()
        return int(current) + 1

    async def latest_sequence(self, *, run_id: UUID) -> int:
        statement = select(
            func.coalesce(func.max(AgentRunEventEntity.sequence_no), 0)
        ).where(AgentRunEventEntity.run_id == run_id)
        return int((await self._session.execute(statement)).scalar_one())

    async def list_after(
        self, *, run_id: UUID, after_sequence: int, limit: int = 200
    ) -> list[AgentRunEventEntity]:
        statement = (
            select(AgentRunEventEntity)
            .where(
                AgentRunEventEntity.run_id == run_id,
                AgentRunEventEntity.sequence_no > after_sequence,
            )
            .order_by(AgentRunEventEntity.sequence_no)
            .limit(limit)
        )
        return list((await self._session.execute(statement)).scalars())


class AgentDelegationRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(
        self, entity: AgentDelegationEntity
    ) -> AgentDelegationEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def get_by_task(
        self, *, parent_task_id: UUID, lock: bool = False
    ) -> AgentDelegationEntity | None:
        statement: Select = select(AgentDelegationEntity).where(
            AgentDelegationEntity.parent_task_id == parent_task_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def claim_poll_candidate(
        self, *, now: datetime
    ) -> AgentDelegationEntity | None:
        statement = (
            select(AgentDelegationEntity)
            .where(
                AgentDelegationEntity.status.in_(
                    (
                        "CREATED",
                        "SUBMITTING",
                        "RUNNING",
                        "WAITING_INPUT",
                        "WAITING_APPROVAL",
                        "CANCEL_REQUESTED",
                    )
                ),
                or_(
                    AgentDelegationEntity.next_poll_at.is_(None),
                    AgentDelegationEntity.next_poll_at <= now,
                ),
            )
            .order_by(
                AgentDelegationEntity.next_poll_at,
                AgentDelegationEntity.created_at,
            )
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        return (await self._session.execute(statement)).scalar_one_or_none()
