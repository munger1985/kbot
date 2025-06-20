from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.dao.entities.kbot_kb_files import KBotKBFiles
from backend.core.database.oracle import async_session

class KBotKBFilesRepository:
    """Repository for KBOT_KB_FILES table operations."""
    
    async def create(self, file: KBotKBFiles) -> KBotKBFiles:
        """Create a new KB file record."""
        async with async_session() as session:
            session.add(file)
            await session.commit()
            await session.refresh(file)
            return file
    
    async def get_by_id(self, id: int) -> Optional[KBotKBFiles]:
        """Get KB file by ID."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotKBFiles).where(KBotKBFiles.id == id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> List[KBotKBFiles]:
        """Get all KB file records."""
        async with async_session() as session:
            result = await session.execute(select(KBotKBFiles))
            return result.scalars().all()
    
    async def update(self, file: KBotKBFiles) -> KBotKBFiles:
        """Update a KB file record."""
        async with async_session() as session:
            session.add(file)
            await session.commit()
            await session.refresh(file)
            return file
    
    async def delete(self, id: int) -> bool:
        """Delete a KB file record by ID."""
        async with async_session() as session:
            file = await self.get_by_id(id)
            if file:
                await session.delete(file)
                await session.commit()
                return True
            return False
    
    async def get_by_kb_id(self, kb_id: int) -> List[KBotKBFiles]:
        """Get files by KB ID."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotKBFiles).where(KBotKBFiles.kb_id == kb_id)
            )
            return result.scalars().all()
    
    async def get_by_batch_id(self, batch_id: int) -> List[KBotKBFiles]:
        """Get files by batch ID."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotKBFiles).where(KBotKBFiles.batch_id == batch_id)
            )
            return result.scalars().all()
    
    async def get_by_status(self, status: str) -> List[KBotKBFiles]:
        """Get files by status."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotKBFiles).where(KBotKBFiles.status == status)
            )
            return result.scalars().all()