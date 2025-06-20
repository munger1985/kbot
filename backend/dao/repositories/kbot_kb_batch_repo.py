from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.dao.entities.kbot_kb_batch import KBotKBBatch
from backend.core.database.oracle import async_session

class KBotKBBatchRepository:
    """Repository for KBOT_KB_BATCH table operations."""
    
    async def create(self, batch_info: KBotKBBatch) -> KBotKBBatch:
        """Create a new KB batch record."""
        async with async_session() as session:
            session.add(batch_info)
            await session.commit()
            await session.refresh(batch_info)
            return batch_info
    
    async def get_by_id(self, id: int) -> Optional[KBotKBBatch]:
        """Get KB batch info by ID."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotKBBatch).where(KBotKBBatch.id == id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> List[KBotKBBatch]:
        """Get all KB batch records."""
        async with async_session() as session:
            result = await session.execute(select(KBotKBBatch))
            return result.scalars().all()
    
    async def update(self, batch_info: KBotKBBatch) -> KBotKBBatch:
        """Update a KB batch record."""
        async with async_session() as session:
            session.add(batch_info)
            await session.commit()
            await session.refresh(batch_info)
            return batch_info
    
    async def delete(self, id: int) -> bool:
        """Delete a KB batch record by ID."""
        async with async_session() as session:
            batch_info = await self.get_by_id(id)
            if batch_info:
                await session.delete(batch_info)
                await session.commit()
                return True
            return False