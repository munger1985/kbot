from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.dao.entities.kbot_kb_chunks import KBotKBChunks
from backend.core.database.oracle import async_session

class KBotKBChunksRepository:
    """Repository for KBOT_KB_CHUNKS table operations."""
    
    async def create(self, chunk: KBotKBChunks) -> KBotKBChunks:
        """Create a new KB chunk record."""
        async with async_session() as session:
            session.add(chunk)
            await session.commit()
            await session.refresh(chunk)
            return chunk
    
    async def get_by_id(self, id: str) -> Optional[KBotKBChunks]:
        """Get KB chunk by ID."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotKBChunks).where(KBotKBChunks.id == id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> List[KBotKBChunks]:
        """Get all KB chunk records."""
        async with async_session() as session:
            result = await session.execute(select(KBotKBChunks))
            return result.scalars().all()
    
    async def update(self, chunk: KBotKBChunks) -> KBotKBChunks:
        """Update a KB chunk record."""
        async with async_session() as session:
            session.add(chunk)
            await session.commit()
            await session.refresh(chunk)
            return chunk
    
    async def delete(self, id: str) -> bool:
        """Delete a KB chunk record by ID."""
        async with async_session() as session:
            chunk = await self.get_by_id(id)
            if chunk:
                await session.delete(chunk)
                await session.commit()
                return True
            return False

    async def get_by_kb_id(self, kb_id: int) -> List[KBotKBChunks]:
        """Get chunks by KB ID."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotKBChunks).where(KBotKBChunks.kb_id == kb_id)
            )
            return result.scalars().all()