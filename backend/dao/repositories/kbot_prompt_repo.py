from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.dao.entities.kbot_prompt import KBotPrompt
from backend.core.database.oracle import async_session

class KBotPromptRepository:
    """Repository for KBOT_PROMPT table operations."""
    
    async def create(self, prompt: KBotPrompt) -> KBotPrompt:
        """Create a new prompt record."""
        async with async_session() as session:
            session.add(prompt)
            await session.commit()
            await session.refresh(prompt)
            return prompt
    
    async def get_by_id(self, id: int) -> Optional[KBotPrompt]:
        """Get prompt by ID."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotPrompt).where(KBotPrompt.id == id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> List[KBotPrompt]:
        """Get all prompt records."""
        async with async_session() as session:
            result = await session.execute(select(KBotPrompt))
            return result.scalars().all()
    
    async def update(self, prompt: KBotPrompt) -> KBotPrompt:
        """Update a prompt record."""
        async with async_session() as session:
            session.add(prompt)
            await session.commit()
            await session.refresh(prompt)
            return prompt
    
    async def delete(self, id: int) -> bool:
        """Delete a prompt record by ID."""
        async with async_session() as session:
            prompt = await self.get_by_id(id)
            if prompt:
                await session.delete(prompt)
                await session.commit()
                return True
            return False
    
    async def get_by_app_id(self, app_id: int) -> List[KBotPrompt]:
        """Get prompts by APP ID."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotPrompt).where(KBotPrompt.app_id == app_id)
            )
            return result.scalars().all()
    
    async def get_by_category(self, category: str) -> List[KBotPrompt]:
        """Get prompts by category."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotPrompt).where(KBotPrompt.prompt_category == category)
            )
            return result.scalars().all()
    
    async def get_by_status(self, status: str) -> List[KBotPrompt]:
        """Get prompts by status."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotPrompt).where(KBotPrompt.status == status)
            )
            return result.scalars().all()