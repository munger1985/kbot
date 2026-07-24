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
    _CLAIM_SCAN_LIMIT = 16

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
        """先选候选 ID，再按主键加锁，规避 Oracle ORA-02014。"""
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
        candidate_statement: Select = (
            select(AgentTaskEntity.task_id)
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
            .limit(self._CLAIM_SCAN_LIMIT)
        )
        task_ids = list(
            (await self._session.execute(candidate_statement)).scalars()
        )
        for task_id in task_ids:
            task = await self._lock_task(task_id)
            if task is None or task.status != "READY":
                continue
            run_state = (
                await self._session.execute(
                    select(
                        AgentRunEntity.status,
                        AgentRunEntity.deadline_at,
                    ).where(AgentRunEntity.run_id == task.run_id)
                )
            ).one_or_none()
            if run_state is None or run_state.status != "RUNNING":
                continue
            if (
                run_state.deadline_at is not None
                and run_state.deadline_at <= now
            ):
                continue
            active_count = int(
                (
                    await self._session.execute(
                        select(func.count())
                        .select_from(AgentTaskEntity)
                        .where(
                            AgentTaskEntity.run_id == task.run_id,
                            AgentTaskEntity.status == "RUNNING",
                        )
                    )
                ).scalar_one()
            )
            if active_count >= max_parallel_tasks:
                continue
            return task
        return None

    async def claim_due_retry(
        self, *, now: datetime
    ) -> AgentTaskEntity | None:
        candidate_statement: Select = (
            select(AgentTaskEntity.task_id)
            .join(
                AgentRunEntity,
                AgentRunEntity.run_id == AgentTaskEntity.run_id,
            )
            .where(
                AgentTaskEntity.status == "RETRY_WAIT",
                AgentTaskEntity.next_retry_at <= now,
                AgentRunEntity.status == "RUNNING",
            )
            .order_by(
                AgentTaskEntity.next_retry_at,
                AgentTaskEntity.task_id,
            )
            .limit(self._CLAIM_SCAN_LIMIT)
        )
        task_ids = list(
            (await self._session.execute(candidate_statement)).scalars()
        )
        for task_id in task_ids:
            task = await self._lock_task(task_id)
            if (
                task is not None
                and task.status == "RETRY_WAIT"
                and task.next_retry_at is not None
                and task.next_retry_at <= now
                and await self._run_is_active(task.run_id, now)
            ):
                return task
        return None

    async def claim_expired_lease(
        self, *, now: datetime
    ) -> AgentTaskEntity | None:
        candidate_statement: Select = (
            select(AgentTaskEntity.task_id)
            .join(
                AgentRunEntity,
                AgentRunEntity.run_id == AgentTaskEntity.run_id,
            )
            .where(
                AgentTaskEntity.status == "RUNNING",
                AgentTaskEntity.lease_until.is_not(None),
                AgentTaskEntity.lease_until <= now,
                AgentRunEntity.status == "RUNNING",
            )
            .order_by(
                AgentTaskEntity.lease_until,
                AgentTaskEntity.task_id,
            )
            .limit(self._CLAIM_SCAN_LIMIT)
        )
        task_ids = list(
            (await self._session.execute(candidate_statement)).scalars()
        )
        for task_id in task_ids:
            task = await self._lock_task(task_id)
            if (
                task is not None
                and task.status == "RUNNING"
                and task.lease_until is not None
                and task.lease_until <= now
                and await self._run_is_active(task.run_id, now)
            ):
                return task
        return None

    async def _lock_task(
        self, task_id: UUID
    ) -> AgentTaskEntity | None:
        """只对单表主键查询使用 SKIP LOCKED，保持 Oracle 可更新。"""
        statement = (
            select(AgentTaskEntity)
            .where(AgentTaskEntity.task_id == task_id)
            .with_for_update(skip_locked=True)
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def _run_is_active(
        self, run_id: UUID, now: datetime
    ) -> bool:
        row = (
            await self._session.execute(
                select(
                    AgentRunEntity.status,
                    AgentRunEntity.deadline_at,
                ).where(AgentRunEntity.run_id == run_id)
            )
        ).one_or_none()
        return bool(
            row is not None
            and row.status == "RUNNING"
            and (row.deadline_at is None or row.deadline_at > now)
        )

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

    async def list_by_task_ids(
        self, *, task_ids: list[UUID]
    ) -> list[AgentArtifactEntity]:
        if not task_ids:
            return []
        statement = (
            select(AgentArtifactEntity)
            .where(AgentArtifactEntity.task_id.in_(task_ids))
            .order_by(
                AgentArtifactEntity.created_at,
                AgentArtifactEntity.artifact_id,
            )
        )
        return list((await self._session.execute(statement)).scalars())


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
    _POLL_SCAN_LIMIT = 16

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

    async def get(
        self, *, delegation_id: UUID, lock: bool = False
    ) -> AgentDelegationEntity | None:
        statement: Select = select(AgentDelegationEntity).where(
            AgentDelegationEntity.delegation_id == delegation_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_open_by_run(
        self, *, parent_run_id: UUID, lock: bool = False
    ) -> list[AgentDelegationEntity]:
        statement: Select = select(AgentDelegationEntity).where(
            AgentDelegationEntity.parent_run_id == parent_run_id,
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
        )
        if lock:
            statement = statement.with_for_update()
        return list((await self._session.execute(statement)).scalars())

    async def claim_poll_candidate(
        self, *, now: datetime
    ) -> AgentDelegationEntity | None:
        """先筛选候选主键，再逐行加锁，规避 Oracle ORA-02014。"""
        eligibility = (
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
            or_(
                AgentDelegationEntity.lease_until.is_(None),
                AgentDelegationEntity.lease_until <= now,
            ),
        )
        candidate_statement = (
            select(AgentDelegationEntity.delegation_id)
            .where(*eligibility)
            .order_by(
                AgentDelegationEntity.next_poll_at,
                AgentDelegationEntity.created_at,
                AgentDelegationEntity.delegation_id,
            )
            .limit(self._POLL_SCAN_LIMIT)
        )
        candidate_ids = list(
            (await self._session.execute(candidate_statement)).scalars()
        )
        for delegation_id in candidate_ids:
            lock_statement = (
                select(AgentDelegationEntity)
                .where(
                    AgentDelegationEntity.delegation_id == delegation_id,
                    *eligibility,
                )
                .with_for_update(skip_locked=True)
            )
            delegation = (
                await self._session.execute(lock_statement)
            ).scalar_one_or_none()
            if delegation is not None:
                return delegation
        return None
