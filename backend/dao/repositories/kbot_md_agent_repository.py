from typing import Optional, Sequence
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from dao.entities.kbot_md_agent import KBotMdAgent


class KBotMdAgentRepository:
    """Agent metadata repository"""
    
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, agent: KBotMdAgent) -> KBotMdAgent:
        """create agent metadata"""
        self.session.add(agent)
        await self.session.flush()
        await self.session.refresh(agent)
        return agent

    async def get_by_id(self, agent_id: int) -> Optional[KBotMdAgent]:
        """get agent metadata by id"""
        stmt = select(KBotMdAgent).where(KBotMdAgent.AGENT_ID == agent_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(self) -> Sequence[KBotMdAgent]:
        """get all agent metadata"""
        stmt = select(KBotMdAgent)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update(self, agent_id: int, **kwargs) -> Optional[KBotMdAgent]:
        """update agent metadata by id"""
        stmt = (
            update(KBotMdAgent)
            .where(KBotMdAgent.AGENT_ID == agent_id)
            .values(**kwargs)
            .returning(KBotMdAgent)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.scalar_one_or_none()

    async def delete(self, agent_id: int) -> bool:
        """delete agent metadata by id"""
        stmt = delete(KBotMdAgent).where(KBotMdAgent.AGENT_ID == agent_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0