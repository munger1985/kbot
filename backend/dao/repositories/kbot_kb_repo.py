from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.dao.entities.kbot_kb import KBotKB
from backend.core.database.oracle import async_session

class KBotKBRepository:
    """Repository for KBOT_KB table operations."""
    
    async def create(self, kb_info: KBotKB) -> KBotKB:
        """Create a new KB record."""
        async with async_session() as session:
            session.add(kb_info)
            await session.commit()
            await session.refresh(kb_info)
            return kb_info
    
    async def get_by_id(self, id: int) -> Optional[KBotKB]:
        """Get KB info by ID."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotKB).where(KBotKB.id == id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> List[KBotKB]:
        """Get all KB records."""
        async with async_session() as session:
            result = await session.execute(select(KBotKB))
            return result.scalars().all()
    
    async def update(self, kb_info: KBotKB) -> KBotKB:
        """Update a KB record."""
        async with async_session() as session:
            session.add(kb_info)
            await session.commit()
            await session.refresh(kb_info)
            return kb_info
    
    async def delete(self, id: int) -> bool:
        """Delete a KB record by ID."""
        async with async_session() as session:
            kb_info = await self.get_by_id(id)
            if kb_info:
                await session.delete(kb_info)
                await session.commit()
                return True
            return False