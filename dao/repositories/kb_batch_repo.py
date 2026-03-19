from typing import Sequence
from sqlalchemy import select
from sqlalchemy.orm import load_only
from core.exceptions import DatabaseException, DataNotFoundException
from dao.entities import BatchEntity
from .base_repo import BaseRepository

class BatchRepository(BaseRepository[BatchEntity]):
    """Repository for KBOT_MD_KB_BATCH table operations."""
    async def create(self, batch: BatchEntity) -> int:
        """Create a new knowledge base batch."""
        try:
            self.session.add(batch)
            await self.session.flush()
            return batch.batch_id
        except Exception as e:
            raise DatabaseException("Failed to create knowledge base batch", original_error=e)

    async def get_by_kb_id(self, kb_id: int) -> Sequence[BatchEntity]:
        """Get knowledge base batches by knowledge base ID."""
        try:
            stmt = select(BatchEntity).where(BatchEntity.kb_id == kb_id)
            result = await self.session.execute(stmt)
            batches = result.scalars().all()
            
            return batches
        except Exception as e:
            raise DatabaseException("Failed to get knowledge base batches by KB ID", original_error=e)
    
    async def get_id_by_name(self, batch_name: str, kb_id: int, app_id: int) -> BatchEntity:
        """Get knowledge base batch ID by batch name, KB ID and app ID."""
        try:
            stmt = select(BatchEntity).options(
                load_only(BatchEntity.batch_id)
            ).where(
                BatchEntity.batch_name == batch_name,
                BatchEntity.kb_id == kb_id,
                BatchEntity.app_id == app_id
            )
            
            result = await self.session.execute(stmt)
            batch = result.scalars().first()
            
            if not batch:
                raise DataNotFoundException(
                    f"Batch not found - name: {batch_name}, kb_id: {kb_id}, app_id: {app_id}"
                )
            
            return batch
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get batch ID by name", original_error=e)
    
    async def get_by_name_and_kb(self, batch_name: str, kb_id: int) -> BatchEntity:
        """Get knowledge base batch by batch name and KB ID (unique constraint)."""
        try:
            stmt = select(BatchEntity).where(
                BatchEntity.batch_name == batch_name,
                BatchEntity.kb_id == kb_id
            )
            
            result = await self.session.execute(stmt)
            batch = result.scalars().first()
            
            if not batch:
                raise DataNotFoundException(
                    f"Batch not found - name: {batch_name}, kb_id: {kb_id}"
                )
            
            return batch
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get batch by name and KB ID", original_error=e)