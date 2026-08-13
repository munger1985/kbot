"""KM Asset Agent Repository。"""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from km_asset_app.entities import KmAgentEntity, KmAgentVersionEntity


class KmAgentRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row) -> None:
        self._session.add(row)
        await self._session.flush()

    async def list(self, *, domain_id: int):
        return list(await self._session.scalars(select(KmAgentEntity).where(KmAgentEntity.domain_id == domain_id).order_by(KmAgentEntity.updated_at.desc())))

    async def get(
        self, *, domain_id: int, agent_id: UUID, lock: bool = False
    ):
        statement = select(KmAgentEntity).where(
            KmAgentEntity.domain_id == domain_id,
            KmAgentEntity.agent_id == agent_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def version(self, *, agent_id: UUID, version_id: UUID):
        return (await self._session.execute(select(KmAgentVersionEntity).where(KmAgentVersionEntity.agent_id == agent_id, KmAgentVersionEntity.agent_version_id == version_id))).scalar_one_or_none()

    async def next_version_no(self, *, agent_id: UUID) -> int:
        value = await self._session.scalar(
            select(func.max(KmAgentVersionEntity.version_no)).where(
                KmAgentVersionEntity.agent_id == agent_id
            )
        )
        return int(value or 0) + 1
