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
        
    
    