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
            return 0
        connstr = db_conf.db_conn_str
        db_type = db_conf.db_type

        if connstr is None or db_type is None:
            return 0
        async with create_session(db_type=db_type, connection_info=connstr) as session:
            stmt = delete(KbotBizTxtEmbedding).where(KbotBizTxtEmbedding.file_id.in_(file_ids))
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount
        
    async def get_similar_embeddings(self, kb_id: int, embedding: List[float], similarity_threshold: float = 0.8, top_k: int = 10) -> List[KbotBizTxtEmbedding]:
        """Get similar embeddings using vector similarity search.
        
        Args:
            kb_id: Knowledge base ID
            embedding: Target embedding vector to compare with
            similarity_threshold: Minimum similarity score (0.0-1.0)
            top_k: Maximum number of results to return
            
        Returns:
            List of similar embeddings ordered by similarity score
        """
        db_repo = KbotMdDbConfRepository()
        db_conf = await db_repo.get_by_kbid(kb_id)
        if db_conf is None:
            return []
        connstr = db_conf.db_conn_str
        db_type = db_conf.db_type
        if connstr is None or db_type is None:
            return []
            
        async with create_session(db_type=db_type, connection_info=connstr) as session:
            # Use database's vector distance function (cosine similarity)
            stmt = select(
                KbotBizTxtEmbedding,
                KbotBizTxtEmbedding.embedding.vector_distance(embedding, 'cosine').label('similarity')
            ).where(
                KbotBizTxtEmbedding.embedding.vector_distance(embedding, 'cosine') <= 1 - similarity_threshold
            ).order_by(
                'similarity'
            ).limit(top_k)
            
            result = await session.execute(stmt)
            return [row[0] for row in result.all()]