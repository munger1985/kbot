"""知识检索 Agent Repository。"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_retrieval_app.entities import (
    KnowledgeRetrievalAgentEntity,
    KnowledgeRetrievalAgentGrantEntity,
    KnowledgeRetrievalAgentVersionEntity,
)


class KnowledgeRetrievalAgentRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add_agent(self, row: KnowledgeRetrievalAgentEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def add_version(self, row: KnowledgeRetrievalAgentVersionEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def get(
        self, *, domain_id: int, agent_id: UUID, lock: bool = False
    ) -> KnowledgeRetrievalAgentEntity | None:
        statement = select(KnowledgeRetrievalAgentEntity).where(
            KnowledgeRetrievalAgentEntity.domain_id == domain_id,
            KnowledgeRetrievalAgentEntity.agent_id == agent_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list(self, *, domain_id: int) -> list[KnowledgeRetrievalAgentEntity]:
        rows = await self._session.scalars(
            select(KnowledgeRetrievalAgentEntity)
            .where(KnowledgeRetrievalAgentEntity.domain_id == domain_id)
            .order_by(
                KnowledgeRetrievalAgentEntity.updated_at.desc(),
                KnowledgeRetrievalAgentEntity.agent_id,
            )
        )
        return list(rows)

    async def current_version(
        self, *, agent_id: UUID, agent_version_id: UUID
    ) -> KnowledgeRetrievalAgentVersionEntity | None:
        statement = select(KnowledgeRetrievalAgentVersionEntity).where(
            KnowledgeRetrievalAgentVersionEntity.agent_id == agent_id,
            KnowledgeRetrievalAgentVersionEntity.agent_version_id == agent_version_id,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def next_version_no(self, *, agent_id: UUID) -> int:
        value = await self._session.scalar(
            select(func.max(KnowledgeRetrievalAgentVersionEntity.version_no)).where(
                KnowledgeRetrievalAgentVersionEntity.agent_id == agent_id
            )
        )
        return int(value or 0) + 1

    async def model_references(self, *, model_id: UUID):
        rows = await self._session.execute(
            select(KnowledgeRetrievalAgentEntity, KnowledgeRetrievalAgentVersionEntity)
            .join(
                KnowledgeRetrievalAgentVersionEntity,
                KnowledgeRetrievalAgentVersionEntity.agent_version_id
                == KnowledgeRetrievalAgentEntity.current_version_id,
            )
        )
        expected = str(model_id)
        return [
            (agent, role)
            for agent, version in rows
            for role, value in dict(version.models_json or {}).items()
            if str(value) == expected
        ]

    async def add_grant(self, row: KnowledgeRetrievalAgentGrantEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def list_grants(
        self, *, domain_id: int
    ) -> list[KnowledgeRetrievalAgentGrantEntity]:
        rows = await self._session.scalars(
            select(KnowledgeRetrievalAgentGrantEntity)
            .where(KnowledgeRetrievalAgentGrantEntity.domain_id == domain_id)
            .order_by(
                KnowledgeRetrievalAgentGrantEntity.updated_at.desc(),
                KnowledgeRetrievalAgentGrantEntity.agent_grant_id,
            )
        )
        return list(rows)

    async def get_grant(
        self, *, domain_id: int, grant_id: UUID, lock: bool = False
    ) -> KnowledgeRetrievalAgentGrantEntity | None:
        statement = select(KnowledgeRetrievalAgentGrantEntity).where(
            KnowledgeRetrievalAgentGrantEntity.domain_id == domain_id,
            KnowledgeRetrievalAgentGrantEntity.agent_grant_id == grant_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def find_grant(
        self,
        *,
        domain_id: int,
        agent_id: UUID,
        subject_type: str,
        subject_id: str,
        lock: bool = False,
    ) -> KnowledgeRetrievalAgentGrantEntity | None:
        statement = select(KnowledgeRetrievalAgentGrantEntity).where(
            KnowledgeRetrievalAgentGrantEntity.domain_id == domain_id,
            KnowledgeRetrievalAgentGrantEntity.agent_id == agent_id,
            KnowledgeRetrievalAgentGrantEntity.subject_type == subject_type,
            KnowledgeRetrievalAgentGrantEntity.subject_id == subject_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def has_active_grant(
        self,
        *,
        domain_id: int,
        agent_id: UUID,
        user_id: str,
        role_codes: tuple[str, ...],
    ) -> bool:
        conditions = [
            (KnowledgeRetrievalAgentGrantEntity.subject_type == "USER")
            & (KnowledgeRetrievalAgentGrantEntity.subject_id == user_id)
        ]
        if role_codes:
            conditions.append(
                (KnowledgeRetrievalAgentGrantEntity.subject_type == "ROLE")
                & KnowledgeRetrievalAgentGrantEntity.subject_id.in_(role_codes)
            )
        from sqlalchemy import or_

        value = await self._session.scalar(
            select(KnowledgeRetrievalAgentGrantEntity.agent_grant_id).where(
                KnowledgeRetrievalAgentGrantEntity.domain_id == domain_id,
                KnowledgeRetrievalAgentGrantEntity.agent_id == agent_id,
                KnowledgeRetrievalAgentGrantEntity.status == "ACTIVE",
                or_(*conditions),
            )
        )
        return value is not None
