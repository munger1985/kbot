from typing import Sequence, Optional
from sqlalchemy import select, delete, and_
from backend.dao.entities.kbot_md_kb_files import (
    KbotMdKbFiles, 
    FileStatus,
    ProcessPriority
)
from backend.dao.entities.kbot_md_kb_batch import KbotMdKbBatch
from backend.dao.entities.kbot_md_kb_chunks import KbotMdKbChunks
from backend.dao.repositories.kbot_md_kb_batch_repo import KbotMdKbBatchRepository
from backend.core.database.oracle import get_session
from backend.utils.common_methods import safe_int

class KbotMdKbFilesRepository:
    """Repository for KBOT_MD_KB_FILES table operations."""
    
    async def get_by_id(self, file_id: int) -> Optional[KbotMdKbFiles]:
        """Get knowledge base file by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbFiles).where(KbotMdKbFiles.file_id == file_id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> Sequence[KbotMdKbFiles]:
        """Get all knowledge base file records."""
        async with get_session() as session:
            result = await session.execute(select(KbotMdKbFiles))
            return result.scalars().all()
    
    async def update(self, file: KbotMdKbFiles) -> KbotMdKbFiles:
        """Update a knowledge base file record."""
        async with get_session() as session:
            session.add(file)
            await session.commit()
            await session.refresh(file)
            return file
    
    async def get_by_app_id(self, app_id: int) -> Sequence[KbotMdKbFiles]:
        """Get knowledge base files by application ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbFiles).where(KbotMdKbFiles.app_id == app_id)
            )
            return result.scalars().all()
    
    async def get_by_kb_id(self, kb_id: int) -> Sequence[KbotMdKbFiles]:
        """Get knowledge base files by knowledge base ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbFiles).where(KbotMdKbFiles.kb_id == kb_id)
            )
            return result.scalars().all()
    
    async def get_by_batch_id(self, batch_id: int) -> Sequence[KbotMdKbFiles]:
        """Get knowledge base files by batch ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbFiles).where(KbotMdKbFiles.batch_id == batch_id)
            )
            return result.scalars().all()
    
    async def get_by_status(self, status: FileStatus) -> Sequence[KbotMdKbFiles]:
        """Get knowledge base files by status."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbFiles).where(KbotMdKbFiles.status == status.value)
            )
            return result.scalars().all()
    
    async def get_by_priority(self, priority: ProcessPriority) -> Sequence[KbotMdKbFiles]:
        """Get knowledge base files by process priority."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbFiles).where(KbotMdKbFiles.process_priority == priority.value)
            )
            return result.scalars().all()
    
    async def get_by_extension(self, extension: str) -> Sequence[KbotMdKbFiles]:
        """Get knowledge base files by file extension."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbFiles).where(KbotMdKbFiles.file_ext == extension)
            )
            return result.scalars().all()
    
    async def get_by_name_and_kb(self, file_name: str, kb_id: int) -> Optional[KbotMdKbFiles]:
        """Get knowledge base file by name and KB ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbFiles)
                .where(and_(
                    KbotMdKbFiles.file_name == file_name,
                    KbotMdKbFiles.kb_id == kb_id
                ))
            )
            return result.scalars().first()
    
    async def _delete_file_and_chunks(self, file: KbotMdKbFiles, session) -> bool:
        """Internal method to delete a file and its chunks.
        
        Args:
            file (KbotMdKbFiles): The file record to delete
            session: The database session
            
        Returns:
            bool: True if deletion succeeded, False otherwise
        """
        await session.delete(file)
        stmt = delete(KbotMdKbChunks).where(KbotMdKbChunks.file_id == file.file_id)
        await session.execute(stmt)
        await session.commit()
        return True

    
    async def delete(self, id: int) -> bool:
        """Delete a KB file record by ID.
        
        Args:
            id (int): The ID of the KB file record to delete
            
        Returns:
            int: Number of records deleted (including related chunks records), 
                 0 if no record was found or deletion failed
        """
        async with get_session() as session:
            file = await self.get_by_id(id)
            if not file:
                return False
            return await self._delete_file_and_chunks(file, session)
    
    async def batch_delete(self, batch_id: int) -> tuple:
        """Delete all KB file records belonging to a specific batch.
        
        Args:
            batch_id (int): The batch ID to delete all related files for
            
        Returns:
            tuple: (success_count, failed_count) where:
                success_count (int): Number of files successfully deleted
                failed_count (int): Number of files that failed to delete
        """
        async with get_session() as session:
            files = await self.get_by_batch_id(batch_id)
            success_count = 0
            failed_count = 0
            
            for file in files:
                if await self._delete_file_and_chunks(file, session):
                    success_count += 1
                else:
                    failed_count += 1
            
            return (success_count, failed_count)
        
    async def create(self, batch: KbotMdKbBatch, files: list = [KbotMdKbFiles]) -> bool:
        """Create a new knowledge base file record."""
        if files is None or len(files) == 0:
            return False
        # check if the batch already exists
        batch_name = str(batch.batch_name)
        kbid = safe_int(batch.kb_id)
        batch_repo = KbotMdKbBatchRepository()
        r = await batch_repo.get_by_name_and_kb(batch_name, kbid)

        if not r: # create the batch if it doesn't exist
            async with get_session() as session:               
                session.add(batch)
                session.add_all(files)
                await session.commit()
                return True
        else:
            async with get_session() as session:
                session.add_all(files)
                await session.commit()
                return True