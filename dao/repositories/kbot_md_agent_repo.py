from typing import Optional, Sequence
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from dao.entities.kbot_md_agent import KbotMdAgent


class KbotMdAgentRepository:
    """Agent metadata repository"""
    
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, agent: KbotMdAgent) -> KbotMdAgent:
        """create agent metadata"""
        self.session.add(agent)
        await self.session.flush()
        await self.session.refresh(agent)
        return agent

    async def get_by_id(self, agent_id: int) -> Optional[KbotMdAgent]:
        """get agent metadata by id"""
        stmt = select(KbotMdAgent).where(KbotMdAgent.agent_id == agent_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(self) -> Sequence[KbotMdAgent]:
        """get all agent metadata"""
        stmt = select(KbotMdAgent)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update(self, agent_id: int, **kwargs) -> Optional[KbotMdAgent]:
        """update agent metadata by id"""
        stmt = (
            update(KbotMdAgent)
            .where(KbotMdAgent.agent_id == agent_id)
            .values(**kwargs)
            .returning(KbotMdAgent)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.scalar_one_or_none()

    async def delete(self, agent_id: int) -> bool:
        """delete agent metadata by id"""
        stmt = delete(KbotMdAgent).where(KbotMdAgent.agent_id == agent_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0