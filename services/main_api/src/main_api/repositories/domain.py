"""平台 Domain 查询 Repository。"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from main_api.entities import PlatformDomainEntity


class PlatformDomainRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def exists_active(self, *, domain_id: int) -> bool:
        statement = select(PlatformDomainEntity.domain_id).where(
            PlatformDomainEntity.domain_id == domain_id,
            PlatformDomainEntity.status == "ACTIVE",
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none() is not None

    async def get_by_name(
        self,
        *,
        name: str,
    ) -> PlatformDomainEntity | None:
        statement = select(PlatformDomainEntity).where(
            PlatformDomainEntity.name == name,
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def add(
        self,
        entity: PlatformDomainEntity,
    ) -> PlatformDomainEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity
