from typing import Sequence, Optional
from sqlalchemy import select, delete
from backend.dao.entities.kbot_md_kb_batch import KbotMdKbBatch
from backend.core.database.meta_oracle import get_session

class KbotMdKbBatchRepository:
    """Repository for KBOT_MD_KB_BATCH table operations."""
    
    async def create(self, batch: KbotMdKbBatch) -> KbotMdKbBatch:
        """Create a new knowledge base batch record."""
        async with get_session() as session:
            session.add(batch)
            await session.commit()
            await session.refresh(batch)
            return batch
    
    async def get_by_id(self, batch_id: int) -> Optional[KbotMdKbBatch]:
        """Get knowledge base batch by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbBatch).where(KbotMdKbBatch.batch_id == batch_id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> Sequence[KbotMdKbBatch]:
        """Get all knowledge base batch records."""
        async with get_session() as session:
            result = await session.execute(select(KbotMdKbBatch))
            return result.scalars().all()
    
    async def update(self, batch: KbotMdKbBatch) -> KbotMdKbBatch:
        """Update a knowledge base batch record."""
        async with get_session() as session:
            session.add(batch)
            await session.commit()
            await session.refresh(batch)
            return batch
    
    async def delete(self, batch_id: int) -> bool:
        """Delete a knowledge base batch record by ID."""
        async with get_session() as session:
            batch = await self.get_by_id(batch_id)
            if not batch:
                return False
            await session.delete(batch)
            await session.commit()
            return True
    
    async def get_by_app_id(self, app_id: int) -> Sequence[KbotMdKbBatch]:
        """Get knowledge base batches by application ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbBatch).where(KbotMdKbBatch.app_id == app_id)
            )
            return result.scalars().all()
    
    async def get_by_kb_id(self, kb_id: int) -> Sequence[KbotMdKbBatch]:
        """Get knowledge base batches by knowledge base ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbBatch).where(KbotMdKbBatch.kb_id == kb_id)
            )
            return result.scalars().all()
    
    async def get_by_batch_name(self, batch_name: str) -> Sequence[KbotMdKbBatch]:
        """Get knowledge base batches by batch name."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbBatch).where(KbotMdKbBatch.batch_name == batch_name)
            )
            return result.scalars().all()
    
    async def get_by_name_and_kb(self, batch_name: str, kb_id: int) -> Optional[KbotMdKbBatch]:
        """Get knowledge base batch by batch name and KB ID (unique constraint)."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbBatch)
                .where(KbotMdKbBatch.batch_name == batch_name)
                .where(KbotMdKbBatch.kb_id == kb_id)
            )
            return result.scalars().first()