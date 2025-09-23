import json
import oracledb
from typing import Sequence
from loguru import logger
from core.database.vec_oracle_pool import OracleConnParams, AsyncOracleConnectionPoolManager
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from dao.repositories.kbot_md_db_conf_repo import KbotMdDbConfRepository
from utils.oracle_vec_handler import OracleVecHandler
from core.dictionary import DbType, ChunkType

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
        """批量创建嵌入记录。"""
        if self.conn_params is None or not embeddings:
            return False
        
        # 准备批量插入的SQL语句
        sql = """INSERT INTO KBOT_BIZ_TXT_EMBEDDING
        (EMBED_ID, KB_ID, FILE_ID, SECURITY_LEVEL, CHUNK_METADATA, EMBEDDING, CHUNK_DOC)
        VALUES
        (:1, :2, :3, :4, :5, :6, :7)"""
        
        # 准备批量数据
        data = []
        for embedding in embeddings:
            # 将每个嵌入对象转换为元组格式，适合executemany
            data.append((
                embedding.embed_id,
                kb_id,
                embedding.file_id,
                embedding.security_level,
                json.dumps(embedding.chunk_metadata) if embedding.chunk_metadata is not None else None,
                OracleVecHandler().convert(vec=embedding.embedding, to_string=True),
                embedding.chunk_doc
            ))
        
        try:
            # 使用连接池执行批量插入
            async with self.pool_manager.get_connection_ctx(self.conn_params) as conn:
                cursor = conn.cursor()
                # 使用executemany进行批量插入
                await self.pool_manager._loop.run_in_executor( # type: ignore
                    None, cursor.executemany, sql, data
                )
                # 提交事务
                await self.pool_manager._loop.run_in_executor(None, conn.commit)  # type: ignore
                logger.info(f"成功批量插入 {len(data)} 条记录")
                return True
                
        except oracledb.Error as e:
            logger.error(f"批量插入失败: {e}")
            return False
        except Exception as e:
            logger.error(f"批量插入过程中发生未知错误: {e}")
            return False
    
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
                                     top_k: int | None = 10,
                                     is_summary_search: bool = False
                                     ) -> Sequence:
        """Get similar embeddings using vector similarity search.
        
        Args:
            kb_id: Knowledge base ID
            query_vec: Target embedding vector to compare with
            security: Security level
            similarity_threshold: Minimum similarity score (0.0-1.0)
            top_k: Maximum number of results to return
            is_summary_search: Whether to search in summary or not
            
        Returns:
            list of similar embeddings ordered by similarity score
        """
        if self.conn_params is None:
            return []
        
        sql = """
            SELECT 
                FILE_ID, CHUNK_DOC, CHUNK_METADATA,
                1 - VECTOR_DISTANCE(EMBEDDING, :query_vec, COSINE) AS similarity
            FROM KBOT_BIZ_TXT_EMBEDDING emb
            WHERE 1 - VECTOR_DISTANCE(EMBEDDING, :query_vec, COSINE) >= :threshold
            AND KB_ID = :kb_id
            AND SECURITY_LEVEL <= :security
            AND emb.CHUNK_METADATA.chunk_type = :chunk_type
            ORDER BY similarity DESC
            FETCH FIRST :top_k ROWS ONLY
        """
        # 添加向量和阈值参数
        params = {
            "kb_id": kb_id,
            "query_vec": query_vec,
            "security": security,
            "threshold": similarity_threshold,
            "top_k": top_k,
            "chunk_type": ChunkType.SUMMARY.value if is_summary_search else ChunkType.TEXT.value
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
    
    async def update_chunk(self,
                            embed_id: str,
                            new_chunk: str,
                            new_embedding: list[float]
                            ) -> bool:
        """Update the embedding and content of a specific chunk.
        
        Args:
            embed_id: Embed ID of the chunk to update
            new_chunk: The updated chunk text
            new_embedding: The new embedding vector for the chunk
            
        Returns:
            True if the update was successful, False otherwise
        """
        if self.conn_params is None:
            return False
        
        # Generate SQL
        sql = """
            UPDATE KBOT_BIZ_TXT_EMBEDDING
            SET CHUNK_DOC = :new_chunk,
                EMBEDDING = :new_embedding
            WHERE EMBED_ID = :embed_id
        """
        # 添加参数
        params = {
            "embed_id": embed_id,
            "new_chunk": new_chunk,
            "new_embedding": OracleVecHandler().convert(vec=new_embedding, to_string=True)
        }
        result = await self.pool_manager.execute_dml(self.conn_params, sql, params)
        return result > 0

    async def get_summary_id_by_chunk_id(self, file_id, chunk_id) -> str | None:
        """Get the embed ID of the summary chunk corresponding to a given text chunk ID.
        
        Args:
            file_id: File ID the chunk belongs to
            chunk_id: Chunk number of the text chunk
            
        Returns:
            The embed ID of the corresponding summary chunk, or None if not found
        """
        if self.conn_params is None:
            return None
        
        sql = """
            SELECT EMBED_ID
            FROM KBOT_BIZ_TXT_EMBEDDING
            WHERE FILE_ID = :file_id
            AND JSON_VALUE(CHUNK_METADATA, '$.chunk_type') = :chunk_type
            AND JSON_VALUE(CHUNK_METADATA, '$.source_embed_id') = :chunk_id
        """
        params = {
            "file_id": file_id,
            "chunk_type": ChunkType.SUMMARY.value,
            "chunk_id": chunk_id
        }
        result = await self.pool_manager.query(self.conn_params, sql, params)
        if result and len(result) > 0:
            return result[0][0]  # Return the EMBED_ID
        return None
    
    async def delete_by_embed_ids(self, embed_ids: list[str]) -> int:
        """Delete embedding records by embed IDs."""
        if self.conn_params is None or not embed_ids:
            return 0
        
        # Generate SQL
        embed_ids_str = ", ".join([f"'{embed_id}'" for embed_id in embed_ids])
        sql = f"""DELETE FROM KBOT_BIZ_TXT_EMBEDDING
        WHERE EMBED_ID IN ({embed_ids_str})"""
        result = await self.pool_manager.execute_dml(self.conn_params, sql, {})
        return result
            
            


