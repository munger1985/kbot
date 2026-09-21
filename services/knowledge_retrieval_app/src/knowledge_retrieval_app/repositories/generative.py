"""知识检索应用 X Search Repository。"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from platform_core.identity import uuid7

from knowledge_retrieval_app.entities import (
    KnowledgeRetrievalResearchEventEntity,
    KnowledgeRetrievalResearchRunEntity,
    KnowledgeRetrievalXSourceEntity,
)


IN_PROGRESS_STATUSES = (
    "SEARCHING",
    "ORGANIZING_SOURCES",
    "COMPOSING",
)


class KnowledgeRetrievalResearchRunRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row: KnowledgeRetrievalResearchRunEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def get(
        self, *, domain_id: int, run_id: UUID, lock: bool = False,
    ) -> KnowledgeRetrievalResearchRunEntity | None:
        statement = select(KnowledgeRetrievalResearchRunEntity).where(
            KnowledgeRetrievalResearchRunEntity.domain_id == domain_id,
            KnowledgeRetrievalResearchRunEntity.run_id == run_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_by_idempotency(
        self, *, domain_id: int, actor_id: str, idempotency_key: str,
    ) -> KnowledgeRetrievalResearchRunEntity | None:
        return (
            await self._session.execute(
                select(KnowledgeRetrievalResearchRunEntity).where(
                    KnowledgeRetrievalResearchRunEntity.domain_id == domain_id,
                    KnowledgeRetrievalResearchRunEntity.actor_id == actor_id,
                    KnowledgeRetrievalResearchRunEntity.idempotency_key == idempotency_key,
                )
            )
        ).scalar_one_or_none()

    async def list(
        self,
        *,
        domain_id: int,
        kind: str | None = None,
        status: str | None = None,
        actor_id: str | None = None,
        limit: int = 50,
    ) -> list[KnowledgeRetrievalResearchRunEntity]:
        conditions = [KnowledgeRetrievalResearchRunEntity.domain_id == domain_id]
        if kind:
            conditions.append(KnowledgeRetrievalResearchRunEntity.kind == kind)
        if status:
            conditions.append(KnowledgeRetrievalResearchRunEntity.status == status)
        if actor_id:
            conditions.append(KnowledgeRetrievalResearchRunEntity.actor_id == actor_id)
        rows = await self._session.scalars(
            select(KnowledgeRetrievalResearchRunEntity)
            .where(*conditions)
            .order_by(KnowledgeRetrievalResearchRunEntity.created_at.desc(), KnowledgeRetrievalResearchRunEntity.run_id)
            .limit(limit)
        )
        return list(rows)

    async def claim(self, *, worker_id: str, lease_until: datetime) -> KnowledgeRetrievalResearchRunEntity | None:
        now = datetime.now(timezone.utc)
        eligibility = (
            or_(
                (
                    (KnowledgeRetrievalResearchRunEntity.status == "ACCEPTED")
                    & or_(
                        KnowledgeRetrievalResearchRunEntity.lease_until.is_(None),
                        KnowledgeRetrievalResearchRunEntity.lease_until < now,
                    )
                ),
                (
                    KnowledgeRetrievalResearchRunEntity.status.in_(IN_PROGRESS_STATUSES)
                    & KnowledgeRetrievalResearchRunEntity.lease_until.is_not(None)
                    & (KnowledgeRetrievalResearchRunEntity.lease_until < now)
                ),
            ),
        )
        # Oracle 不允许 FETCH FIRST 与 FOR UPDATE 作用于同一查询块。
        candidate_ids = list(
            (
                await self._session.execute(
                    select(KnowledgeRetrievalResearchRunEntity.run_id)
                    .where(*eligibility)
                    .order_by(KnowledgeRetrievalResearchRunEntity.created_at, KnowledgeRetrievalResearchRunEntity.run_id)
                    .limit(32)
                )
            ).scalars()
        )
        for run_id in candidate_ids:
            row = (
                await self._session.execute(
                    select(KnowledgeRetrievalResearchRunEntity)
                    .where(KnowledgeRetrievalResearchRunEntity.run_id == run_id, *eligibility)
                    .with_for_update(skip_locked=True)
                )
            ).scalar_one_or_none()
            if row is None:
                continue
            row.lease_owner = worker_id
            row.lease_token = uuid7()
            row.lease_until = lease_until
            row.row_version = int(row.row_version) + 1
            if row.started_at is None:
                row.started_at = now
            await self._session.flush()
            return row
        return None

    async def delete(self, row: KnowledgeRetrievalResearchRunEntity) -> None:
        await self._session.delete(row)
        await self._session.flush()

class KnowledgeRetrievalResearchEventRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row: KnowledgeRetrievalResearchEventEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def list(self, *, domain_id: int, run_id: UUID) -> list[KnowledgeRetrievalResearchEventEntity]:
        rows = await self._session.scalars(
            select(KnowledgeRetrievalResearchEventEntity)
            .where(
                KnowledgeRetrievalResearchEventEntity.domain_id == domain_id,
                KnowledgeRetrievalResearchEventEntity.run_id == run_id,
            )
            .order_by(KnowledgeRetrievalResearchEventEntity.sequence_no)
        )
        return list(rows)

    async def next_sequence(self, *, run_id: UUID) -> int:
        value = await self._session.scalar(
            select(func.max(KnowledgeRetrievalResearchEventEntity.sequence_no)).where(
                KnowledgeRetrievalResearchEventEntity.run_id == run_id
            )
        )
        return int(value or 0) + 1

    async def delete_by_run(self, *, domain_id: int, run_id: UUID) -> None:
        await self._session.execute(
            delete(KnowledgeRetrievalResearchEventEntity).where(
                KnowledgeRetrievalResearchEventEntity.domain_id == domain_id,
                KnowledgeRetrievalResearchEventEntity.run_id == run_id,
            )
        )

class KnowledgeRetrievalXSourceRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row: KnowledgeRetrievalXSourceEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def list(self, *, domain_id: int, run_id: UUID) -> list[KnowledgeRetrievalXSourceEntity]:
        rows = await self._session.scalars(
            select(KnowledgeRetrievalXSourceEntity)
            .where(
                KnowledgeRetrievalXSourceEntity.domain_id == domain_id,
                KnowledgeRetrievalXSourceEntity.run_id == run_id,
            )
            .order_by(KnowledgeRetrievalXSourceEntity.display_order)
        )
        return list(rows)

    async def delete_by_run(self, *, domain_id: int, run_id: UUID) -> None:
        await self._session.execute(
            delete(KnowledgeRetrievalXSourceEntity).where(
                KnowledgeRetrievalXSourceEntity.domain_id == domain_id,
                KnowledgeRetrievalXSourceEntity.run_id == run_id,
            )
        )
