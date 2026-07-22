import json
from loguru import logger
from typing import Sequence, Any
from sqlalchemy import select, delete, and_, update, func, case, Row
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from platform_core.exceptions import DatabaseException, DataNotFoundException
from platform_core.dictionary import FileStatus, ProcessPriority
from dao.entities import KBEntity, FileEntity
from .base_repo import BaseRepository


class FileRepository(BaseRepository[FileEntity]):
    """Repository for KBOT_MD_KB_FILES table operations."""
    
    async def get(self, file_id: str) -> FileEntity:
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
    
    async def get_file_id_path(self, kb_id: int, 
                               file_ids: list[str] | None = None, 
                               batchs: list[str] | None = None
                            ):
        """获取知识库文件信息"""
        try:
            conditions = [FileEntity.kb_id == kb_id]
            if file_ids:
                conditions.append(FileEntity.file_id == file_ids[0] if len(file_ids) == 1 else FileEntity.file_id.in_(file_ids))
            elif batchs:
                conditions.append(FileEntity.batch == batchs[0] if len(batchs) == 1 else FileEntity.batch.in_(batchs))
                
            query = select(FileEntity.file_id, FileEntity.file_path, FileEntity.status).where(and_(*conditions))
            result = await self.session.execute(query)
            rows = result.fetchall()

            if not rows:
                raise DataNotFoundException(f"知识库文件不存在: {file_ids}")
            return rows
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"获取知识库文件信息失败", original_error=e)
        
    async def get_parser_params(self, file_id: str) -> dict[str, Any]:
        """获取文件解析器参数"""
        try:
            result = await self.session.execute(
                select(FileEntity.parser_params).where(FileEntity.file_id == file_id)
            )
            parser_params = result.scalar_one_or_none()
            if not parser_params:
                raise DataNotFoundException(f"文件 {file_id} 解析器参数不存在")
            return parser_params
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"获取文件解析器参数失败", original_error=e)
    
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

    async def create(self, files: list[FileEntity]) -> None:
        """创建知识库文件记录"""
        try:
            # 1. 批量收集需要检查的文件路径
            kb_ids = []
            file_paths = []
            file_map = {}  # 用于快速查找
            
            for file in files:
                kb_ids.append(file.kb_id)
                file_paths.append(file.file_path)
                file_map[(file.kb_id, file.file_path)] = file
            
            # 2. 批量查询已存在的文件 (使用注入的 session)
            existing_query = select(FileEntity).where(
                and_(
                    FileEntity.kb_id.in_(kb_ids),
                    FileEntity.file_path.in_(file_paths)
                )
            )
            existing_result = await self.session.execute(existing_query)
            existing_files = existing_result.scalars().all()
            
            # 3. 分类处理
            files_to_add = []
            ids_to_delete = []
            
            for existing_file in existing_files:
                key = (existing_file.kb_id, existing_file.file_path)
                if key in file_map:
                    new_file = file_map[key]
                    if new_file.is_overwrite:
                        ids_to_delete.append(existing_file.id)
                    # 如果不需要覆盖，就不添加新文件
                    del file_map[key]
            
            # 剩下的都是不存在的文件，直接添加
            files_to_add.extend(file_map.values())
            
            # 4. 批量删除（如果需要）
            if ids_to_delete:
                await self._batch_delete(self.session, ids_to_delete)
            
            # 5. 批量添加
            if files_to_add:
                self.session.add_all(files_to_add)
                # 事务提交移交给 Service 层
                    
        except Exception as e:
            raise DatabaseException(f"创建知识库文件记录失败", original_error=e)
    
    async def _batch_delete(self, session: AsyncSession, file_ids: list[str]) -> None:
        """批量删除文件记录（内部辅助方法，保持传入 session 的设计）"""
        try:
            if not file_ids:
                return
            if len(file_ids) == 1:
                stmt = delete(FileEntity).where(FileEntity.file_id == file_ids[0])
            else:
                stmt = delete(FileEntity).where(FileEntity.file_id.in_(file_ids))
            
            await session.execute(stmt)
        except Exception as e:
            raise DatabaseException(f"删除知识库文件记录失败", original_error=e)
        
    async def update_file_status(self, file_ids: list[str], status: FileStatus, log_msg: str | None = None):
        """
        Update the status of a knowledge base file record with log message appending.
        :param file_ids: List of File IDs to update
        :param status: New file status
        :param log_msg: Optional log message to append
        :return: True if successful
        """
        try:
            # If multiple file_ids and log_msg is provided, process individually
            # to properly append to each file's existing log
            if len(file_ids) > 1 and log_msg is not None:
                for file_id in file_ids:
                    await self.update_file_status([file_id], status, log_msg)
                return
            
            condition = FileEntity.file_id == file_ids[0] if len(file_ids) == 1 else FileEntity.file_id.in_(file_ids)
            
            if log_msg is not None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_log_entry = f"{timestamp}: {log_msg}"

                # Get current log_msg (only for single file_id case)
                stmt = select(FileEntity.log_msg).where(condition)
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
                    condition
                ).values(
                    status=status.value,
                    log_msg=final_log
                )
            else:
                # Only update status (works for single or multiple file_ids)
                update_stmt = update(FileEntity).where(
                    condition
                ).values(status=status.value)

            await self.session.execute(update_stmt)

            logger.debug(f"[File Repo] Successfully updated status for {len(file_ids)} files.")

        except Exception as e:
            logger.error(f"Database error while updating file status - file_ids: {file_ids}, status: {status}, "
                        f"error type: {type(e).__name__}, error: {str(e)}", exc_info=True)
            raise DatabaseException("Failed to update file status", original_error=e)
    
    async def delete(self, kb_id: int, file_ids: list[str] | None = None) -> None:
        """根据知识库ID和文件ID列表删除知识库文件记录"""
        try:
            if file_ids:
                await self._batch_delete(self.session, file_ids)
            else:
                stmt = delete(FileEntity).where(FileEntity.kb_id == kb_id)
                await self.session.execute(stmt)
        except Exception as e:
            raise DatabaseException(f"删除知识库文件记录失败", original_error=e)
        
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
    
    async def batch_update_file_status(self, file_ids: list[str], status: FileStatus, log_msg: str | None = None) -> bool:
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
    
    async def update_tags(self, file_id: str, tags: list[str]):
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
                raise DataNotFoundException(f"File {file_id} not found for tag update")
            
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

        except DataNotFoundException as e:
            raise e
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
        
    async def get_names_by_ids(self, file_ids: list[str]) -> dict[str, str]:
        try:
            stmt = select(FileEntity.file_id, FileEntity.file_name).where(FileEntity.file_id.in_(file_ids))
            result = await self.session.execute(stmt)
            file_names = result.all()
            
            return {file.file_id: file.file_name for file in file_names}
        except Exception as e:
            raise DatabaseException("Failed to get file names by IDs", original_error=e)
        
    async def update_parser_params(self, file_id: str, parser_params: dict[str, Any]) -> None:
        """更新文件解析器参数"""
        try:
            result = await self.session.execute(
                update(FileEntity)
                .where(FileEntity.file_id == file_id)
                .values(chunk_parser=parser_params)
                .returning(FileEntity.file_id)
            )
            if result.scalar() is None:
                raise DataNotFoundException(f"更新文件 {file_id} 解析器参数返回结果为0行")
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"更新文件解析器参数失败", original_error=e)
        
    async def get_file_ids(self, file_ids: list[str]) -> list[tuple[str, str]]:
        """根据ID获取一个或多个文件名"""
        try:
            if len(file_ids) == 1:
                stmt = select(FileEntity.file_id, FileEntity.file_path).where(FileEntity.file_id == file_ids[0])
                result = await self.session.execute(stmt)
            else:
                result = await self.session.execute(
                    select(FileEntity.file_id, FileEntity.file_path).where(FileEntity.file_id.in_(file_ids))
                )
            files = result.all()
            if not files:
                raise DataNotFoundException(f"未找到知识库文件")
            # 将 Row 对象转换为普通元组
            return [tuple(row) for row in files]
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"获取知识库文件记录失败", original_error=e)