from typing import Sequence, Optional
from sqlalchemy import select, delete, and_, or_
from backend.dao.entities.kbot_md_prompt import KbotMdPrompt
from backend.core.database.oracle import get_session

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
    
    async def get_by_app_id(self, app_id: int) -> Sequence[KbotMdPrompt]:
        """Get prompts by application ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdPrompt).where(KbotMdPrompt.app_id == app_id)
            )
            return result.scalars().all()
    
    async def get_by_domain_id(self, domain_id: int) -> Sequence[KbotMdPrompt]:
        """Get prompts by domain ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdPrompt).where(KbotMdPrompt.domain_id == domain_id)
            )
            return result.scalars().all()
    
    async def get_by_name(self, name: str) -> Optional[KbotMdPrompt]:
        """Get prompt by name."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdPrompt).where(KbotMdPrompt.name == name)
            )
            return result.scalars().first()
    
    async def search_by_template(self, keyword: str) -> Sequence[KbotMdPrompt]:
        """Search prompts by template content (case-insensitive)."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdPrompt).where(
                    KbotMdPrompt.template.ilike(f"%{keyword}%")
                )
            )
            return result.scalars().all()
    
    async def get_by_name_and_app(self, name: str, app_id: int) -> Optional[KbotMdPrompt]:
        """Get prompt by name and application ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdPrompt)
                .where(and_(
                    KbotMdPrompt.name == name,
                    KbotMdPrompt.app_id == app_id
                ))
            )
            return result.scalars().first()