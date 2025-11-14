import json
from loguru import logger
from typing import Sequence
from sqlalchemy import select, delete, and_, update, text
from datetime import datetime
from dao.entities.kbot_md_kb_files import KbotMdKbFiles
from core.dictionary import *
from dao.entities.kbot_md_kb_batch import KbotMdKbBatch
from dao.entities.kbot_md_kb import KbotMdKb
from dao.repositories.kbot_md_kb_batch_repo import KbotMdKbBatchRepository
from core.database.meta_oracle import get_session


class KbotMdKbFilesRepository:
    """Repository for KBOT_MD_KB_FILES table operations."""
    
    async def get_by_id(self, file_id: str) -> KbotMdKbFiles | None:
        """Get knowledge base file by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbFiles).where(KbotMdKbFiles.file_id == file_id)
            )
            return result.scalar_one_or_none()
    
    async def get_path_by_id(self, file_id: str) -> str | None:
        """Get file path by file ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbFiles.file_path).where(KbotMdKbFiles.file_id == file_id)
            )
            file_path = result.scalar_one_or_none()
            return file_path

    async def get_all(self) -> Sequence[KbotMdKbFiles]:
        """Get all knowledge base file records."""
        async with get_session() as session:
            result = await session.execute(select(KbotMdKbFiles))
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
    
    async def get_by_name_and_kb(self, file_name: str, kb_id: int) -> KbotMdKbFiles | None:
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
    
    async def delete(self, kb_id: int | None, batch_id: int | None, file_ids: list[str] | None) -> int:
        """Delete knowledge base files.
        kb_id (int): The knowledge base ID to delete all related files for
        batch_id (int): The batch ID to delete all related files for
        file_ids (str): The file IDs to delete

        Returns:
            int: The number of deleted records
        """
        async with get_session() as session:
            if file_ids is not None:
                if len(file_ids) == 1:
                    stmt = delete(KbotMdKbFiles).where(KbotMdKbFiles.file_id==file_ids[0])
                else:
                    stmt = delete(KbotMdKbFiles).where(KbotMdKbFiles.file_id.in_(file_ids))
                result = await session.execute(stmt)
            elif batch_id is not None:
                stmt = delete(KbotMdKbFiles).where(KbotMdKbFiles.batch_id == batch_id)
                result = await session.execute(stmt)
                stmt = delete(KbotMdKbBatch).where(KbotMdKbBatch.batch_id == batch_id)
                await session.execute(stmt)
            elif kb_id is not None:
                stmt = delete(KbotMdKbFiles).where(KbotMdKbFiles.kb_id == kb_id)
                result = await session.execute(stmt)
                stmt = delete(KbotMdKbBatch).where(KbotMdKbBatch.kb_id == kb_id)
                await session.execute(stmt)
                stmt = delete(KbotMdKb).where(KbotMdKb.kb_id == kb_id)
                await session.execute(stmt)
            else:
                pass
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
                        await self.delete(None, None, [existing_file.file_id])

                    session.add(file)
                
                await session.commit()
                return True
            
    async def update_file_status(self, file_id: str, status: FileStatus, log_msg: str | None = None) -> bool:
        """Update the status of a knowledge base file record with log message appending."""
        async with get_session() as session:
            if log_msg is not None:
                # For log appending, use a separate query with text expression
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_log_entry = f"{timestamp}: {log_msg}"
                
                # Use Oracle's string concatenation syntax
                update_query = text("""
                    UPDATE kbot_md_kb_files 
                    SET status = :status, 
                        log_msg = COALESCE(log_msg, '') || CHR(10) || :new_log
                    WHERE file_id = :file_id
                """)
                
                await session.execute(
                    update_query, 
                    {"status": status.value, "new_log": new_log_entry, "file_id": file_id}
                )
            else:
                # If no log message, just update status
                query = update(KbotMdKbFiles)\
                    .where(KbotMdKbFiles.file_id == file_id)\
                    .values(status=status.value)
                await session.execute(query)
            
            await session.commit()
            return True
        
    async def update_file_parsed_metadata(self, file_id: str, parsed_metadata: str) -> bool:
        """Update the parse metadata of a knowledge base file record."""
        async with get_session() as session:
            await session.execute(
                update(KbotMdKbFiles)
                .where(KbotMdKbFiles.file_id == file_id)
                .values(parsed_metadata=parsed_metadata)
                )
            await session.commit()
            return True
        
    async def batch_update_file_status(self, file_ids: list[str], status: FileStatus, log_msg: str | None = None) -> bool:
        """Batch update status of knowledge base file records."""
        async with get_session() as session:
            await session.execute(
                update(KbotMdKbFiles)
                .where(KbotMdKbFiles.file_id.in_(file_ids))
                .values(status=status.value, log_msg=log_msg)
                )
            await session.commit()
            return True
    
    async def update_tags(self, file_id: str, tags: list[str]) -> bool:
        """更新知识库文件的标签"""
        try:
            async with get_session() as session:
                # 先查询现有的biz_metadata
                result = await session.execute(
                    select(KbotMdKbFiles.biz_metadata)
                    .where(KbotMdKbFiles.file_id == file_id)
                )
                record = result.scalar_one_or_none()
                
                if record is None:
                    logger.warning(f"KbotMdKbFiles未找到记录，文件ID: {file_id}")
                    return False
                
                # 处理biz_metadata（可能为None、空字典或有效JSON）
                existing_metadata = {}
                if record:
                    if isinstance(record, dict):
                        existing_metadata = record
                    elif isinstance(record, str):
                        try:
                            existing_metadata = json.loads(record)
                        except json.JSONDecodeError:
                            logger.warning(f"KbotMdKbFiles解析biz_metadata失败，文件ID: {file_id}，内容: {record}")
                            existing_metadata = {}
                
                # 更新tags字段，保留其他字段
                existing_metadata["tags"] = tags
                
                # 将字典转换为JSON字符串
                updated_metadata_json = json.dumps(existing_metadata, ensure_ascii=False)
                
                # 更新数据库
                await session.execute(
                    update(KbotMdKbFiles)
                    .where(KbotMdKbFiles.file_id == file_id)
                    .values(biz_metadata=updated_metadata_json)
                )
                await session.commit()
                
                logger.info(f"KbotMdKbFiles成功更新标签，文件ID: {file_id}, 标签: {tags}")
                return True
                
        except Exception as e:
            logger.error(f"KbotMdKbFiles更新标签失败: {e}")
            return False