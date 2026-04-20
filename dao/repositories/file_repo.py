import json
from loguru import logger
from typing import Sequence, List, Optional
from sqlalchemy import select, delete, and_, update, func, case
from datetime import datetime
from core.exceptions import DatabaseException, DataNotFoundException
from core.dictionary import FileStatus, ProcessPriority, YesNoEnum
from dao.entities import KbEntity, FileEntity, BatchEntity
from .kb_batch_repo import BatchRepository
from .base_repo import BaseRepository


class FileRepository(BaseRepository[FileEntity]):
    """Repository for KBOT_MD_KB_FILES table operations."""
    
    async def get_by_id(self, file_id: str) -> FileEntity:
        """Get knowledge base file by ID."""
        try:
            stmt = select(FileEntity).where(FileEntity.file_id == file_id)
            result = await self.session.execute(stmt)
            file = result.scalar_one_or_none()
            
            if not file:
                raise DataNotFoundException(f"File with ID {file_id} not found")
            
            return file
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get file by ID", original_error=e)
    
    async def get_path_by_id(self, file_id: str) -> str:
        """Get file path by file ID."""
        try:
            stmt = select(FileEntity.file_path).where(FileEntity.file_id == file_id)
            result = await self.session.execute(stmt)
            file_path = result.scalar_one_or_none()
            
            if file_path is None:
                raise DataNotFoundException(f"File path for ID {file_id} not found")
            
            return file_path
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get file path by ID", original_error=e)

    async def get_all(self) -> Sequence[FileEntity]:
        """Get all knowledge base file records."""
        try:
            stmt = select(FileEntity)
            result = await self.session.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            raise DatabaseException("Failed to get all files", original_error=e)
    
    async def get_by_kb_id(self, kb_id: int) -> Sequence[FileEntity]:
        """Get knowledge base files by knowledge base ID."""
        try:
            stmt = select(FileEntity).where(FileEntity.kb_id == kb_id)
            result = await self.session.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            raise DatabaseException("Failed to get files by KB ID", original_error=e)
    
    async def get_by_batch_id(self, batch_id: int) -> Sequence[FileEntity]:
        """Get knowledge base files by batch ID."""
        try:
            stmt = select(FileEntity).where(FileEntity.batch_id == batch_id)
            result = await self.session.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            raise DatabaseException("Failed to get files by batch ID", original_error=e)
    
    async def get_by_status(self, status: FileStatus, amount: int = 20) -> Sequence[FileEntity]:
        """Get knowledge base files by status."""
        try:
            stmt = select(FileEntity).where(
                FileEntity.status == status.value
            ).limit(amount)
            result = await self.session.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            raise DatabaseException("Failed to get files by status", original_error=e)
    
    async def get_by_priority(self, priority: ProcessPriority) -> Sequence[FileEntity]:
        """Get knowledge base files by process priority."""
        try:
            stmt = select(FileEntity).where(
                FileEntity.process_priority == priority.value
            )
            result = await self.session.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            raise DatabaseException("Failed to get files by priority", original_error=e)
    
    async def get_by_extension(self, extension: str) -> Sequence[FileEntity]:
        """Get knowledge base files by file extension."""
        try:
            stmt = select(FileEntity).where(FileEntity.file_ext == extension)
            result = await self.session.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            raise DatabaseException("Failed to get files by extension", original_error=e)
    
    async def get_by_name_and_kb(self, file_name: str, kb_id: int) -> FileEntity:
        """Get knowledge base file by name and KB ID."""
        try:
            stmt = select(FileEntity).where(
                and_(
                    FileEntity.file_name == file_name,
                    FileEntity.kb_id == kb_id
                )
            )
            result = await self.session.execute(stmt)
            file = result.scalars().first()
            
            if not file:
                raise DataNotFoundException(
                    f"File {file_name} not found in KB {kb_id}"
                )
            
            return file
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get file by name and KB ID", original_error=e)
    
    async def delete(self, kb_id: int | None, batch_id: int | None, file_ids: List[str] | None):
        """
        Delete knowledge base files.
        :param kb_id: The knowledge base ID to delete all related files for
        :param batch_id: The batch ID to delete all related files for
        :param file_ids: List of file IDs to delete
        :return: The number of deleted records
        """
        try:
            
            if file_ids is not None and len(file_ids) > 0:
                # Delete specific file IDs
                stmt = delete(FileEntity).where(FileEntity.file_id.in_(file_ids)).returning(FileEntity.file_id)
                await self.session.execute(stmt)
                
            elif batch_id is not None:
                # Delete by batch ID
                stmt = delete(FileEntity).where(FileEntity.batch_id == batch_id).returning(FileEntity.file_id)
                await self.session.execute(stmt)
                
                # Delete batch record
                batch_stmt = delete(BatchEntity).where(BatchEntity.batch_id == batch_id)
                await self.session.execute(batch_stmt)
                
            elif kb_id is not None:
                # Delete by KB ID
                stmt = delete(FileEntity).where(FileEntity.kb_id == kb_id).returning(FileEntity.file_id)
                await self.session.execute(stmt)
                
                # Delete related batches and KB
                batch_stmt = delete(BatchEntity).where(BatchEntity.kb_id == kb_id)
                await self.session.execute(batch_stmt)
                
                kb_stmt = delete(KbEntity).where(KbEntity.kb_id == kb_id)
                await self.session.execute(kb_stmt)
            
        except Exception as e:
            raise DatabaseException("Failed to delete files", original_error=e)

    async def create(self, files: List[FileEntity]):
        """
        Create new knowledge base file records.
        :param batch: Batch entity
        :param files: List of FileEntity objects
        :return: True if successful
        """
        if not files:
            logger.warning("No files provided for creation")
            return

        try:
            self.session.add_all(files)
            logger.info(f"Created {len(files)} file records in the database.")
            
        except Exception as e:
            raise DatabaseException("Failed to create file records", original_error=e)
    
    async def update_file_status(self, file_id: str, status: FileStatus, log_msg: Optional[str] = None):
        """
        Update the status of a knowledge base file record with log message appending.
        :param file_id: File ID to update
        :param status: New file status
        :param log_msg: Optional log message to append
        :return: True if successful
        """
        try:
            if log_msg is not None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_log_entry = f"{timestamp}: {log_msg}"

                # Get current log_msg length
                stmt = select(FileEntity.log_msg).where(FileEntity.file_id == file_id)
                current_log_result = await self.session.execute(stmt)
                current_log = current_log_result.scalar_one_or_none()

                MAX_LOG_LENGTH = 4000

                if current_log:
                    # Calculate available space
                    current_length = len(current_log)
                    new_entry_length = len(new_log_entry)
                    total_length = current_length + 1 + new_entry_length  # +1 for newline

                    if total_length > MAX_LOG_LENGTH:
                        # Truncate old log to make room for new entry
                        keep_length = MAX_LOG_LENGTH - new_entry_length - 10  # Reserve 10 chars buffer
                        if keep_length > 0:
                            truncated_log = current_log[-keep_length:]  # Keep most recent entries
                            truncated_log = "..." + truncated_log if truncated_log.startswith("...") else truncated_log
                            final_log = f"{truncated_log}\n{new_log_entry}"
                        else:
                            # Not enough space, only keep new entry
                            final_log = f"...{new_log_entry[-(MAX_LOG_LENGTH - 50):]}"  # Keep last part of new entry
                    else:
                        # Enough space, append normally
                        final_log = f"{current_log}\n{new_log_entry}"
                else:
                    # No existing log
                    if len(new_log_entry) > MAX_LOG_LENGTH:
                        final_log = new_log_entry[:MAX_LOG_LENGTH]
                    else:
                        final_log = new_log_entry

                # Update status with processed log message
                update_stmt = update(FileEntity).where(
                    FileEntity.file_id == file_id
                ).values(
                    status=status.value,
                    log_msg=final_log
                )
            else:
                # Only update status
                update_stmt = update(FileEntity).where(
                    FileEntity.file_id == file_id
                ).values(status=status.value)

            await self.session.execute(update_stmt)

            logger.debug(f"Updated status for file {file_id} to {status.name}")

        except Exception as e:
            logger.error(f"Database error updating file status - file_id: {file_id}, status: {status}, "
                        f"error type: {type(e).__name__}, error: {str(e)}", exc_info=True)
            raise DatabaseException("Failed to update file status", original_error=e)
    
    async def update_file_parsed_metadata(self, file_id: str, parsed_metadata: str):
        """
        Update the parse metadata of a knowledge base file record.
        :param file_id: File ID to update
        :param parsed_metadata: New parsed metadata string
        :return: True if successful
        """
        try:
            update_stmt = update(FileEntity).where(
                FileEntity.file_id == file_id
            ).values(parsed_metadata=parsed_metadata).returning(FileEntity.file_id)
            
            result = await self.session.execute(update_stmt)
            
            if not result.scalar():
                raise DataNotFoundException(f"File {file_id} not found for metadata update")
            
            logger.debug(f"Updated parsed metadata for file {file_id}")
            return True
            
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to update file parsed metadata", original_error=e)
    
    async def batch_update_file_status(self, file_ids: List[str], status: FileStatus, log_msg: Optional[str] = None) -> bool:
        """
        Batch update status of knowledge base file records.
        :param file_ids: List of file IDs to update
        :param status: New status value
        :param log_msg: Optional log message
        :return: True if successful
        """
        try:
            if not file_ids:
                logger.warning("No file IDs provided for batch update")
                return False
            
            update_data = {"status": status.value}
            if log_msg is not None:
                update_data["log_msg"] = log_msg # type: ignore
            
            update_stmt = update(FileEntity).where(
                FileEntity.file_id.in_(file_ids)
            ).values(** update_data)
            
            await self.session.execute(update_stmt)
            
            logger.info(f"Batch updated status for {len(file_ids)} files to {status.name}")
            return True
            
        except Exception as e:
            raise DatabaseException("Failed to batch update file status", original_error=e)
    
    async def update_tags(self, file_id: str, tags: List[str]):
        """
        Update tags for a knowledge base file.
        :param file_id: File ID to update
        :param tags: List of tags to set
        :return: True if successful
        """
        try:
            # Get current record
            stmt = select(FileEntity).where(FileEntity.file_id == file_id)
            result = await self.session.execute(stmt)
            record = result.scalar_one_or_none()
            
            if not record:
                logger.warning(f"File {file_id} not found for tag update")
                return
            
            # Process business metadata
            existing_metadata = {}
            if record.biz_metadata:
                if isinstance(record.biz_metadata, dict):
                    existing_metadata = record.biz_metadata
                elif isinstance(record.biz_metadata, str):
                    try:
                        existing_metadata = json.loads(record.biz_metadata)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse biz_metadata for file {file_id}: {e}")
            
            # Update tags field
            existing_metadata["tags"] = tags
            
            # Update database
            update_stmt = update(FileEntity).where(
                FileEntity.file_id == file_id
            ).values(biz_metadata=existing_metadata)
            
            await self.session.execute(update_stmt)
            
            logger.info(f"Updated tags for file {file_id}: {tags}")
            
        except Exception as e:
            logger.error(f"Failed to update file tags for file ID {file_id}: {e}", exc_info=True)
            raise DatabaseException("Failed to update file tags", original_error=e)
    
    async def get_name_by_id(self, file_id: str) -> str:
        """
        Get file name by file ID.
        :param file_id: File ID to query
        :return: File name string
        """
        try:
            stmt = select(FileEntity.file_name).where(FileEntity.file_id == file_id)
            result = await self.session.execute(stmt)
            file_name = result.scalar_one_or_none()
            
            if file_name is None:
                raise DataNotFoundException(f"File name for ID {file_id} not found")
            
            return file_name
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get file name by ID", original_error=e)
        
    async def get_names_by_ids(self, file_ids: List[str]) -> dict[str, str]:
        try:
            stmt = select(FileEntity.file_id, FileEntity.file_name).where(FileEntity.file_id.in_(file_ids))
            result = await self.session.execute(stmt)
            file_names = result.all()
            
            return {file.file_id: file.file_name for file in file_names}
        except Exception as e:
            raise DatabaseException("Failed to get file names by IDs", original_error=e)