from typing import Sequence
from sqlalchemy import select
from dao.entities.kbot_sys_parser_conf import KbotSysParserConf
from core.database.meta_oracle import get_session


class KbotSysParserConfRepository:
    """Repository for KBOT_SYS_PARSER_CONF table operations."""
    
    async def get_default_paser(self, file_ext: str) -> str | None:
        """Get system configuration by file category."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotSysParserConf.chunk_parser_param)
                .where(KbotSysParserConf.file_ext == file_ext,
                       KbotSysParserConf.status == 1,
                       KbotSysParserConf.is_default == 1)
            )
            
            return result.scalar_one_or_none()
        
    
    