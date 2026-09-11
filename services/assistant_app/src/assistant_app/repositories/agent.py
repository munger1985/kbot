"""智能工作台 Agent Repository。"""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from assistant_app.entities import AssistantAgentEntity, AssistantAgentVersionEntity


class AssistantAgentRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add_agent(self, row: AssistantAgentEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def add_version(self, row: AssistantAgentVersionEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def get(self, *, domain_id: int, agent_id: UUID, lock: bool = False) -> AssistantAgentEntity | None:
        statement = select(AssistantAgentEntity).where(
            AssistantAgentEntity.domain_id == domain_id,
            AssistantAgentEntity.agent_id == agent_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list(self, *, domain_id: int) -> list[AssistantAgentEntity]:
        rows = await self._session.scalars(
            select(AssistantAgentEntity)
            .where(AssistantAgentEntity.domain_id == domain_id)
            .order_by(AssistantAgentEntity.updated_at.desc(), AssistantAgentEntity.agent_id)
        )
        return list(rows)

    async def current_version(self, *, agent_id: UUID, agent_version_id: UUID) -> AssistantAgentVersionEntity | None:
        return (await self._session.execute(
            select(AssistantAgentVersionEntity).where(
                AssistantAgentVersionEntity.agent_id == agent_id,
                AssistantAgentVersionEntity.agent_version_id == agent_version_id,
            )
        )).scalar_one_or_none()

    async def next_version_no(self, *, agent_id: UUID) -> int:
        value = await self._session.scalar(
            select(func.max(AssistantAgentVersionEntity.version_no)).where(
                AssistantAgentVersionEntity.agent_id == agent_id
            )
        )
        return int(value or 0) + 1
