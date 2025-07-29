from typing import Sequence, Optional
from sqlalchemy import select, delete, and_, or_
from dao.entities.kbot_md_prompt import KbotMdPrompt
from core.database.meta_oracle import get_session


class KbotMdPromptRepository:
    """Repository for KBOT_MD_PROMPT table operations."""
    
    async def create(self, prompt: KbotMdPrompt) -> KbotMdPrompt:
        """Create a new prompt record."""
        async with get_session() as session:
            session.add(prompt)
            await session.commit()
            await session.refresh(prompt)
            return prompt
    
    async def get_by_id(self, prompt_id: int) -> Optional[KbotMdPrompt]:
        """Get prompt by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdPrompt).where(KbotMdPrompt.prompt_id == prompt_id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> Sequence[KbotMdPrompt]:
        """Get all prompt records."""
        async with get_session() as session:
            result = await session.execute(select(KbotMdPrompt))
            return result.scalars().all()
    
    async def update(self, prompt: KbotMdPrompt) -> KbotMdPrompt:
        """Update a prompt record."""
        async with get_session() as session:
            session.add(prompt)
            await session.commit()
            await session.refresh(prompt)
            return prompt
    
    async def delete(self, prompt_id: int) -> bool:
        """Delete a prompt record by ID."""
        async with get_session() as session:
            prompt = await self.get_by_id(prompt_id)
            if not prompt:
                return False
            await session.delete(prompt)
            await session.commit()
            return True
    