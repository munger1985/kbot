from typing import Sequence
from sqlalchemy import select, delete
from core.database.factory import create_session
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from dao.repositories.kbot_md_db_conf_repo import KbotMdDbConfRepository
from utils.oracle_vec_handler import OracleVecHandler

class KbotBizTxtEmbeddingRepository:

    async def create(self, kb_id: int, embeddings: list[KbotBizTxtEmbedding]) -> bool:
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
    
    async def delete_by_file_ids(self, kb_id: int, file_ids: list[str]) -> int:
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
        
    async def get_similar_embeddings(self,
                                     kb_id: int,
                                     query_vec: str,
                                     security: int,
                                     similarity_threshold: float | None = 0.8,
                                     top_k: int | None = 10
                                     ) -> Sequence:
        """Get similar embeddings using vector similarity search.
        
        Args:
            kb_id: Knowledge base ID
            query_vec: Target embedding vector to compare with
            security: Security level
            similarity_threshold: Minimum similarity score (0.0-1.0)
            top_k: Maximum number of results to return
            
        Returns:
            list of similar embeddings ordered by similarity score
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
            # Generate SQL
            sql = """
                SELECT 
                    EMBED_ID, KB_ID, FILE_ID, CHUNK_DOC, CHUNK_METADATA,
                    1 - VECTOR_DISTANCE(EMBEDDING, :query_vec, COSINE) AS similarity
                FROM KBOT_BIZ_TXT_EMBEDDING
                WHERE 1 - VECTOR_DISTANCE(EMBEDDING, :query_vec, COSINE) >= :threshold
                AND KB_ID = :kb_id
                AND SECURITY_LEVEL <= :security
                ORDER BY similarity DESC
                FETCH FIRST :top_k ROWS ONLY
            """
            # 添加向量和阈值参数
            params = {}
            params["kb_id"] = kb_id
            params["query_vec"] = query_vec
            params["security"] = security
            params["threshold"] = similarity_threshold
            params["top_k"] = top_k

            # 分步获取原生连接
            conn = await session.connection()  # 获取AsyncConnection
            raw_conn = await conn.get_raw_connection()  # 获取底层连接
            driver_conn = raw_conn.driver_connection  # 获取驱动连接             
            driver_conn.outputtypehandler = OracleVecHandler.vector_type_handler # type: ignore
            
            # 执行查询
            cursor = driver_conn.cursor() # type: ignore
            await cursor.execute(sql, params)
            result = await cursor.fetchall()
        
            return result

        
            
            


