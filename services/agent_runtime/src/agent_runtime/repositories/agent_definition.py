"""Root Agent 定义查询 Repository。"""

from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_runtime.entities import AgentDefinitionEntity


class AgentDefinitionRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(
        self, entity: AgentDefinitionEntity
    ) -> AgentDefinitionEntity:
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def get_scoped(
        self,
        *,
        agent_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> AgentDefinitionEntity | None:
        statement: Select = select(AgentDefinitionEntity).where(
            AgentDefinitionEntity.agent_id == agent_id,
            AgentDefinitionEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_active(
        self,
        *,
        agent_id: UUID,
        domain_id: int,
    ) -> AgentDefinitionEntity | None:
        statement = select(AgentDefinitionEntity).where(
            AgentDefinitionEntity.agent_id == agent_id,
            AgentDefinitionEntity.domain_id == domain_id,
            AgentDefinitionEntity.status == "ACTIVE",
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_by_key(
        self,
        *,
        domain_id: int,
        agent_key: str,
    ) -> AgentDefinitionEntity | None:
        statement = select(AgentDefinitionEntity).where(
            AgentDefinitionEntity.domain_id == domain_id,
            AgentDefinitionEntity.agent_key == agent_key,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_scoped(
        self,
        *,
        domain_id: int,
    ) -> list[AgentDefinitionEntity]:
        statement = (
            select(AgentDefinitionEntity)
            .where(
                AgentDefinitionEntity.domain_id == domain_id,
            )
            .order_by(
                AgentDefinitionEntity.display_name,
                AgentDefinitionEntity.agent_id,
            )
        )
        return list((await self._session.execute(statement)).scalars())
