
from typing import List
from sqlalchemy import select, delete
from core.database.factory import create_session
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from dao.repositories.kbot_md_db_conf_repo import KbotMdDbConfRepository

class KbotBizTxtEmbeddingRepository:

    async def create(self, kb_id: int, embeddings: List[KbotBizTxtEmbedding]) -> bool:
        """Create a new embedding record."""
        db_repo = KbotMdDbConfRepository()
        db_conf = await db_repo.get_by_kbid(kb_id)
        if db_conf is None:
            return False
        connstr = db_conf.db_conn_str
        db_type = db_conf.db_type
        if connstr is None or db_type is None:
            return False
        async with create_session(db_type=db_type, connection_info=connstr) as session:
            for embedding in embeddings:
                session.add(embedding)
            await session.commit()
            return True
    
    async def delete_by_file_ids(self, kb_id: int, file_ids: List[int]) -> int:
        """Delete embedding records by file IDs."""
        db_repo = KbotMdDbConfRepository()
        db_conf = await db_repo.get_by_kbid(kb_id)
        if db_conf is None:
            return False
        connstr = db_conf.db_conn_str
        db_type = db_conf.db_type
        if connstr is None or db_type is None:
            return False
        async with create_session(db_type=db_type, connection_info=connstr) as session:
            stmt = delete(KbotBizTxtEmbedding).where(KbotBizTxtEmbedding.file_id.in_(file_ids))
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount

