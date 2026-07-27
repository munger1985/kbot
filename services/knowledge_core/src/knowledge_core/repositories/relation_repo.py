"""Persistence operations for revision-scoped KC relations."""
from uuid import UUID
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_core.entities import KcRelationEntity


class RelationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add(self, relation: KcRelationEntity) -> KcRelationEntity:
        self.session.add(relation)
        await self.session.flush()
        return relation

    async def list_active(self, *, bundle_revision_id: UUID, predicate: str | None = None) -> list[KcRelationEntity]:
        statement: Select = select(KcRelationEntity).where(
            KcRelationEntity.bundle_revision_id == bundle_revision_id,
            KcRelationEntity.relation_status == "ACTIVE",
        )
        if predicate:
            statement = statement.where(KcRelationEntity.predicate == predicate)
        return list((await self.session.execute(statement)).scalars())

    async def list_for_object(self, *, object_id: UUID, bundle_revision_id: UUID) -> list[KcRelationEntity]:
        statement = select(KcRelationEntity).where(
            KcRelationEntity.bundle_revision_id == bundle_revision_id,
            KcRelationEntity.relation_status == "ACTIVE",
            (KcRelationEntity.subject_id == object_id) | (KcRelationEntity.object_id == object_id),
        )
        return list((await self.session.execute(statement)).scalars())
