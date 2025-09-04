import json
from typing import Sequence
from core.database.vec_oracle_pool import OracleConnParams, AsyncOracleConnectionPoolManager
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from dao.repositories.kbot_md_db_conf_repo import KbotMdDbConfRepository
from utils.oracle_vec_handler import OracleVecHandler
from core.dictionary import DbType

class KbotBizTxtEmbeddingRepository:
    """Repository for KBOT_BIZ_TXT_EMBEDDING table operations."""
    def __init__(self, kb_id: int):
        self.kb_id = kb_id
        self.db_conf = None
        self.conn_params = None
        self.pool_manager = AsyncOracleConnectionPoolManager()

    async def initialize(self):
        db_repo = KbotMdDbConfRepository()
        self.db_conf = await db_repo.get_by_kbid(self.kb_id)
        if self.db_conf is None:
            return False
        connstr = self.db_conf.db_conn_str
        db_type = self.db_conf.db_type
        if connstr is not None and db_type == DbType.ORACLE:
            self.conn_params = OracleConnParams(
                user=connstr.get("user"), # type: ignore
                password=connstr.get("password"), # type: ignore
                dsn=f"{connstr.get('host')}:{connstr.get('port')}/{connstr.get('service_name')}:pooled"
            )
          

    async def create(self, kb_id: int, embeddings: list[KbotBizTxtEmbedding]) -> bool:
        """Create a new embedding record."""
        if self.conn_params is None:
            return False
        
        # Generate SQL for batch insert
        sql = """INSERT INTO KBOT_BIZ_TXT_EMBEDDING
        (EMBED_ID, KB_ID, FILE_ID, SECURITY_LEVEL, CHUNK_METADATA, EMBEDDING, CHUNK_DOC)
        VALUES
        (:embed_id, :kb_id, :file_id, :security_level, :chunk_metadata, :embedding, :chunk_doc)"""
        
        params_list = []
        for embedding in embeddings:
            params = {
                "embed_id": embedding.embed_id,
                "kb_id": kb_id,
                "file_id": embedding.file_id,
                "chunk_doc": embedding.chunk_doc,
                "chunk_metadata": json.dumps(embedding.chunk_metadata) if embedding.chunk_metadata is not None else None,
                "embedding": OracleVecHandler().convert(vec=embedding.embedding, to_string=True),
                "security_level": embedding.security_level
            }
            result = await self.pool_manager.execute_dml(self.conn_params, sql, params)
        return True
    
    async def delete_by_file_ids(self, kb_id: int, file_ids: list[str]) -> int:
        """Delete embedding records by file IDs."""
        if self.conn_params is None:
            return 0
        
        # Generate SQL
        file_ids_str = ", ".join([f"'{file_id}'" for file_id in file_ids])
        sql = f"""DELETE FROM KBOT_BIZ_TXT_EMBEDDING
        WHERE FILE_ID IN ({file_ids_str})"""
        result = await self.pool_manager.execute_dml(self.conn_params, sql, {})
        return result
        
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
        if self.conn_params is None:
            return []
        
        sql = """
            SELECT 
                FILE_ID, CHUNK_DOC, CHUNK_METADATA,
                1 - VECTOR_DISTANCE(EMBEDDING, :query_vec, COSINE) AS similarity
            FROM KBOT_BIZ_TXT_EMBEDDING
            WHERE 1 - VECTOR_DISTANCE(EMBEDDING, :query_vec, COSINE) >= :threshold
            AND KB_ID = :kb_id
            AND SECURITY_LEVEL <= :security
            ORDER BY similarity DESC
            FETCH FIRST :top_k ROWS ONLY
        """
        # 添加向量和阈值参数
        params = {
            "kb_id": kb_id,
            "query_vec": query_vec,
            "security": security,
            "threshold": similarity_threshold,
            "top_k": top_k
        }
        result = await self.pool_manager.query(self.conn_params, sql, params)
           
        return result

        
        
    async def full_text_search(self,
                               kb_id: int,
                               keyword: str,
                               security: int,
                               top_k: int | None = 10,
                               simularity_threshold: float | None = 0.8
                                ) -> Sequence:
        """Get chunk record by full text search.
        
        Args:
            kb_id: Knowledge base ID
            keyword: Target text to compare with
            
        Returns:
            list of chunk records
        """
        if self.conn_params is None:
            return []

        # Generate SQL
        sql = """
            SELECT FILE_ID, 
                    CHUNK_DOC, 
                    CHUNK_METADATA,
                    SCORE(1) AS similarity
            FROM KBOT_BIZ_TXT_EMBEDDING
            WHERE KB_ID = :kb_id
            AND SECURITY_LEVEL <= :security
            AND CONTAINS(CHUNK_DOC, REGEXP_REPLACE(:keyword, '\\W+', ' ACCUM '), 1) > 0
            ORDER BY similarity DESC
            FETCH FIRST :top_k ROWS ONLY
        """
        # 添加向量和阈值参数
        params = {
            "kb_id": kb_id,
            "keyword": keyword,
            "security": security,
            #"simularity_threshold": simularity_threshold,
            "top_k": top_k
        }
        result = await self.pool_manager.query(self.conn_params, sql, params)

        return result

        
            
            


