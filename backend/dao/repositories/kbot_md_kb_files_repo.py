from typing import Sequence, Optional
from sqlalchemy import select, delete, and_, update
from dao.entities.kbot_md_kb_files import KbotMdKbFiles
from dao.data_dict import (
    FileStatus,
    ProcessPriority,
    YesNoEnum,
    SecurityLevel
)
from dao.entities.kbot_md_kb_batch import KbotMdKbBatch
from dao.entities.kbot_md_kb import KbotMdKb
from dao.repositories.kbot_md_kb_batch_repo import KbotMdKbBatchRepository
from core.database.meta_oracle import get_session

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
    
    async def delete_by_batch(self, batch_id: int) -> int:
        """Delete knowledge base files by batch ID.
        batch_id (int): The batch ID to delete all related files for

        Returns:
            int: The number of deleted records
        """
        # Update file status to deleted. //更新文件状态为已删除
        async with get_session() as session:
            stmt = update(KbotMdKbFiles).where(KbotMdKbFiles.batch_id == batch_id).values(status=FileStatus.DELETED.value)
            result = await session.execute(stmt)
            stmt = delete(KbotMdKbBatch).where(KbotMdKbBatch.batch_id == batch_id)
            await session.execute(stmt)
            await session.commit()
            return result.rowcount

    
    async def delete_by_ids(self, file_ids: list[int]) -> int:
        """Delete knowledge base file by ID.
        file_ids (int): The file IDs to delete

        returns:
            int: The number of deleted records
        """
        async with get_session() as session:
            if len(file_ids) == 1:
                stmt = update(KbotMdKbFiles).where(KbotMdKbFiles.file_id==file_ids[0]).values(status=FileStatus.DELETED.value)
            else:
                stmt = update(KbotMdKbFiles).where(KbotMdKbFiles.file_id.in_(file_ids)).values(status=FileStatus.DELETED.value)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount
    
    async def delete_by_kb(self, kb_id: int) -> int:
        """Delete all KB file records belonging to a specific knowledge base.
        
        Args:
            kb_id (int): The knowledge base ID to delete all related files for
            
        Returns:
            int: The number of deleted records
        """
        async with get_session() as session:
            stmt = update(KbotMdKbFiles).where(KbotMdKbFiles.kb_id == kb_id).values(status=FileStatus.DELETED.value)
            result = await session.execute(stmt)
            stmt = delete(KbotMdKbBatch).where(KbotMdKbBatch.kb_id == kb_id)
            await session.execute(stmt)
            stmt = delete(KbotMdKb).where(KbotMdKb.kb_id == kb_id)
            await session.execute(stmt)
            await session.commit()
            return result.rowcount
        
    async def create(self, batch: KbotMdKbBatch, files: list = [KbotMdKbFiles]) -> bool:
        """Create a new knowledge base file record."""
        if files is None or len(files) == 0:
            return False

        batch_repo = KbotMdKbBatchRepository()
        batch_entity = await batch_repo.get_id_by_name(batch.batch_name, batch.kb_id, app_id=batch.app_id)
        async with get_session() as session:  
            if not batch_entity: # create the batch if it doesn't exist   
                session.add(batch)
                await session.flush()
                for file in files:
                    file.batch_id = batch.batch_id
                session.add_all(files)
                await session.commit()
                return True
            else:
                for file in files:
                    file.batch_id = batch_entity.batch_id

                    # check if the file already exists //检查文件是否已存在
                    existing_file = await session.execute(
                        select(KbotMdKbFiles).where(
                            and_(
                                KbotMdKbFiles.app_id == file.app_id,
                                KbotMdKbFiles.kb_id == file.kb_id,
                                KbotMdKbFiles.batch_id == file.batch_id,
                                KbotMdKbFiles.file_path == file.file_path
                            )
                        )
                    )
                    existing_file = existing_file.scalars().first()
                    
                    if existing_file and file.is_overwrite == YesNoEnum.YES.value:
                        # if the file already exists and overwrite is allowed // 如果文件已存在且允许覆盖，则更新现有记录
                        # mark the existing file as deleted and delete its chunks // 将现有文件标记为已删除并删除其块
                        await self.delete_by_ids([existing_file.file_id])

                    session.add(file)
                
                await session.commit()
                return True
            
    async def update_file_status(self, file_id: int, status: FileStatus) -> bool:
        """Update the status of a knowledge base file record."""
        async with get_session() as session:
            await session.execute(
                update(KbotMdKbFiles)
                .where(KbotMdKbFiles.file_id == file_id)
                .values(status=status.value)
                )
            await session.commit()
            return True
    
    async def actual_delete_by_ids(self, file_ids: list[int]) -> int:
        """Delete knowledge base file by ID."""
        async with get_session() as session:
            stmt = delete(KbotMdKbFiles).where(
                and_(
                    KbotMdKbFiles.file_id.in_(file_ids),
                    KbotMdKbFiles.status == FileStatus.DELETED.value
                )
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount
        