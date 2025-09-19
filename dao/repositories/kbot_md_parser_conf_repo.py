from typing import Sequence
from sqlalchemy import select, delete
from dao.entities.kbot_md_parser_conf import KbotMdParserConf
from core.database.meta_oracle import get_session


class KbotMdParserConfRepository:
    """Repository for KBOT_SYS_PARSER_CONF table operations."""
    
    async def delete_by_kb_id(self, kb_id: int) -> bool:
        """Delete system configuration by knowledge base ID."""
        async with get_session() as session:
            result = await session.execute(
                delete(KbotMdParserConf).where(KbotMdParserConf.kb_id == kb_id)
            )
            await session.commit()
            return result.rowcount > 0
        
    async def get_default_paser(self, file_ext: str, kb_id: int) -> str | None:
        """Get default parser configuration by file category."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdParserConf.chunk_parser_param)
                .where(KbotMdParserConf.file_ext == file_ext,
                       KbotMdParserConf.kb_id == kb_id)
            )
            
            return result.scalar_one_or_none()
        
    
    