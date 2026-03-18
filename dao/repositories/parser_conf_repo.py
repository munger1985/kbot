from loguru import logger
from sqlalchemy import select, delete
from core.exceptions import DatabaseException, DataNotFoundException
from dao.entities import ParserConfEntity
from .base_repo import BaseRepository


class ParserConfRepository(BaseRepository[ParserConfEntity]):
    """Repository for KBOT_SYS_PARSER_CONF table operations."""
    
    async def delete_by_kb_id(self, kb_id: int):
        """
        Delete system configuration by knowledge base ID.
        :param kb_id: Knowledge base ID to delete configurations for
        :return: True if any records were deleted, False otherwise
        """
        try:
            stmt = delete(ParserConfEntity).where(ParserConfEntity.kb_id == kb_id).returning(ParserConfEntity.conf_id)
            result = await self.session.execute(stmt)
            deleted_count = len(result.fetchall())
            logger.info(f"Deleted {deleted_count} parser configurations for KB ID {kb_id}")
                
        except Exception as e:
            raise DatabaseException("Failed to delete parser configurations by KB ID", original_error=e)
        
    async def get_default_parser(self, file_ext: str, kb_id: int) -> str:
        """
        Get default parser configuration by file extension and KB ID.
        Fix typo: renamed from get_default_paser to get_default_parser
        :param file_ext: File extension to filter (e.g., 'pdf', 'docx')
        :param kb_id: Knowledge base ID
        :return: Chunk parser parameter string
        """
        try:
            stmt = select(ParserConfEntity.chunk_parser_param).where(
                ParserConfEntity.file_ext == file_ext,
                ParserConfEntity.kb_id == kb_id
            )
            result = await self.session.execute(stmt)
            parser_param = result.scalar_one_or_none()
            
            if parser_param is None:
                raise DataNotFoundException(
                    f"No default parser configuration found for file extension '{file_ext}' (KB ID: {kb_id})"
                )
            
            return parser_param
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get default parser configuration", original_error=e)