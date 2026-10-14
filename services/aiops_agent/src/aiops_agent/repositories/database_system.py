"""数据库系统及拓扑 Repository。"""

from collections.abc import Callable
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.entities import DatabaseSystemEntity, DatabaseSystemMemberEntity, DatabaseSystemRelationEntity, DatabaseSystemSiteEntity
from aiops_agent.repositories._base import AIOpsRepository


class DatabaseSystemRepository(AIOpsRepository):
    def __init__(self, session: AsyncSession, assert_active: Callable[[], None] | None = None) -> None:
        super().__init__(session, assert_active)

    async def add_system(self, entity: DatabaseSystemEntity) -> DatabaseSystemEntity:
        return await self._add(entity)

    async def add_sites(self, entities: list[DatabaseSystemSiteEntity]) -> None:
        self._check_active()
        self._session.add_all(entities)

    async def add_members(self, entities: list[DatabaseSystemMemberEntity]) -> None:
        self._check_active()
        self._session.add_all(entities)

    async def add_relations(self, entities: list[DatabaseSystemRelationEntity]) -> None:
        self._check_active()
        self._session.add_all(entities)

    async def get_scoped(self, *, database_system_id: UUID, domain_id: int, lock: bool = False) -> DatabaseSystemEntity | None:
        self._check_active()
        statement = select(DatabaseSystemEntity).where(DatabaseSystemEntity.database_system_id == database_system_id, DatabaseSystemEntity.domain_id == domain_id)
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_scoped(self, *, domain_id: int, status: str | None, limit: int) -> list[DatabaseSystemEntity]:
        self._check_active()
        statement = select(DatabaseSystemEntity).where(DatabaseSystemEntity.domain_id == domain_id)
        if status is not None:
            statement = statement.where(DatabaseSystemEntity.status == status)
        return list((await self._session.execute(statement.order_by(DatabaseSystemEntity.updated_at.desc(), DatabaseSystemEntity.database_system_id.desc()).limit(limit))).scalars())

    async def get_by_name(self, *, domain_id: int, display_name: str) -> DatabaseSystemEntity | None:
        self._check_active()
        return (await self._session.execute(select(DatabaseSystemEntity).where(DatabaseSystemEntity.domain_id == domain_id, func.lower(DatabaseSystemEntity.display_name) == display_name.casefold()).limit(1))).scalar_one_or_none()

    async def list_sites(self, database_system_id: UUID) -> list[DatabaseSystemSiteEntity]:
        self._check_active()
        return list((await self._session.execute(select(DatabaseSystemSiteEntity).where(DatabaseSystemSiteEntity.database_system_id == database_system_id).order_by(DatabaseSystemSiteEntity.sort_order, DatabaseSystemSiteEntity.site_id))).scalars())

    async def list_members(self, database_system_id: UUID) -> list[DatabaseSystemMemberEntity]:
        self._check_active()
        return list((await self._session.execute(select(DatabaseSystemMemberEntity).where(DatabaseSystemMemberEntity.database_system_id == database_system_id).order_by(DatabaseSystemMemberEntity.sort_order, DatabaseSystemMemberEntity.member_id))).scalars())

    async def list_relations(self, database_system_id: UUID) -> list[DatabaseSystemRelationEntity]:
        self._check_active()
        return list((await self._session.execute(select(DatabaseSystemRelationEntity).where(DatabaseSystemRelationEntity.database_system_id == database_system_id).order_by(DatabaseSystemRelationEntity.sort_order, DatabaseSystemRelationEntity.relation_id))).scalars())

    async def member_count(self, database_system_id: UUID) -> int:
        self._check_active()
        return int((await self._session.execute(select(func.count()).select_from(DatabaseSystemMemberEntity).where(DatabaseSystemMemberEntity.database_system_id == database_system_id))).scalar_one())

    async def find_system_for_target(self, *, target_id: UUID, domain_id: int) -> DatabaseSystemEntity | None:
        self._check_active()
        return (await self._session.execute(select(DatabaseSystemEntity).join(DatabaseSystemMemberEntity, DatabaseSystemMemberEntity.database_system_id == DatabaseSystemEntity.database_system_id).where(DatabaseSystemMemberEntity.target_id == target_id, DatabaseSystemEntity.domain_id == domain_id).limit(1))).scalar_one_or_none()

    async def target_is_assigned_elsewhere(self, *, target_id: UUID, database_system_id: UUID) -> bool:
        self._check_active()
        count = (await self._session.execute(select(func.count()).select_from(DatabaseSystemMemberEntity).where(DatabaseSystemMemberEntity.target_id == target_id, DatabaseSystemMemberEntity.database_system_id != database_system_id))).scalar_one()
        return bool(count)

    async def replace_topology(self, database_system_id: UUID) -> None:
        self._check_active()
        await self._session.execute(delete(DatabaseSystemRelationEntity).where(DatabaseSystemRelationEntity.database_system_id == database_system_id))
        await self._session.execute(delete(DatabaseSystemMemberEntity).where(DatabaseSystemMemberEntity.database_system_id == database_system_id))
        await self._session.execute(delete(DatabaseSystemSiteEntity).where(DatabaseSystemSiteEntity.database_system_id == database_system_id))
