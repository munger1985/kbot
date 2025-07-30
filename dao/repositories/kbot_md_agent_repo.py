from typing import Sequence
from sqlalchemy import select, update, delete
from dao.entities.kbot_md_agent import KbotMdAgent
from core.database.meta_oracle import get_session

class KbotMdAgentRepository:
    """Agent metadata repository"""
    
    async def create(self, agent: KbotMdAgent) -> KbotMdAgent:
        """create agent metadata"""
        async with get_session() as session:
            session.add(agent)
            await session.flush()
            await session.refresh(agent)
            return agent

    async def get_by_id(self, agent_id: int) -> KbotMdAgent | None:
        """get agent metadata by id"""
        async with get_session() as session:
            stmt = select(KbotMdAgent).where(KbotMdAgent.agent_id == agent_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_all(self) -> Sequence[KbotMdAgent]:
        """get all agent metadata"""
        async with get_session() as session:
            stmt = select(KbotMdAgent)
            result = await session.execute(stmt)
            return result.scalars().all()

    async def update(self, agent_id: int, **kwargs) -> KbotMdAgent | None:
        """update agent metadata by id"""
        async with get_session() as session:
            stmt = (
                update(KbotMdAgent)
                .where(KbotMdAgent.agent_id == agent_id)
                .values(**kwargs)
                .returning(KbotMdAgent)
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.scalar_one_or_none()

    async def delete(self, agent_id: int) -> bool:
        """delete agent metadata by id"""
        async with get_session() as session:
            stmt = delete(KbotMdAgent).where(KbotMdAgent.agent_id == agent_id)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0