from typing import Optional, Sequence
from datetime import datetime
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from dao.entities.kbot_md_agent_feedback import KbotMdAgentFeedback


class KbotMdAgentFeedbackRepository:
    """Repository for agent feedback operations"""
    
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, feedback: KbotMdAgentFeedback) -> KbotMdAgentFeedback:
        """Create new feedback record"""
        self.session.add(feedback)
        await self.session.flush()
        await self.session.refresh(feedback)
        return feedback

    async def get_by_id(self, fb_id: int) -> Optional[KbotMdAgentFeedback]:
        """Get feedback by primary key"""
        stmt = select(KbotMdAgentFeedback).where(KbotMdAgentFeedback.fb_id == fb_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_agent_id(self, agent_id: int) -> Sequence[KbotMdAgentFeedback]:
        """Get all feedbacks for specific agent"""
        stmt = select(KbotMdAgentFeedback).where(KbotMdAgentFeedback.agent_id == agent_id)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update(self, fb_id: int, **kwargs) -> Optional[KbotMdAgentFeedback]:
        """Update feedback record"""
        if 'updated_time' not in kwargs:
            kwargs['updated_time'] = datetime.now()
            
        stmt = (
            update(KbotMdAgentFeedback)
            .where(KbotMdAgentFeedback.fb_id == fb_id)
            .values(**kwargs)
            .returning(KbotMdAgentFeedback)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.scalar_one_or_none()

    async def delete(self, fb_id: int) -> bool:
        """Delete feedback record"""
        stmt = delete(KbotMdAgentFeedback).where(KbotMdAgentFeedback.fb_id == fb_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0

    async def get_all(self) -> Sequence[KbotMdAgentFeedback]:
        """Get all feedback records"""
        stmt = select(KbotMdAgentFeedback)
        result = await self.session.execute(stmt)
        return result.scalars().all()