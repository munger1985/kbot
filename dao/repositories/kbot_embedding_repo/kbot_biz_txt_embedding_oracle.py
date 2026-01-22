import json
import oracledb
from typing import Sequence
from loguru import logger
from core.database.vec_oracle_pool import OracleConnParams, AsyncOracleConnectionPoolManager
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from utils.oracle_vec_handler import OracleVecHandler
from core.dictionary import ChunkType
from dao.repositories.kbot_biz_txt_embedding_interface import IEmbeddingRepository
from utils.common import safe_read_content


class OracleEmbeddingRepository(IEmbeddingRepository):
    """Repository for KBOT_BIZ_TXT_EMBEDDING table operations."""
    def __init__(self, kb_id: int):
        self.kb_id = kb_id
        self.db_conf = None
        self.conn_params = None
        self.pool_manager = AsyncOracleConnectionPoolManager()

    async def initialize(self, connstr: dict) -> bool:

        if connstr is not None:
            username = connstr.get("user")
            password = connstr.get("password")
            if username is None or password is None:
                logger.error("Oracle连接参数中缺少用户名或密码")
                return False
            
            self.conn_params = OracleConnParams(
                user=username,
                password=password,
                dsn=f"{connstr.get('host')}:{connstr.get('port')}/{connstr.get('service_name')}:pooled"
            )
            return True
        else:
            logger.error("Oracle连接参数为空")
            return False
          

    async def create(self, kb_id: int, embeddings: list[KbotBizTxtEmbedding]) -> bool:
        """批量创建嵌入记录。"""
        if self.conn_params is None or not embeddings:
            return False
        
        # 准备批量插入的SQL语句
        sql = """INSERT INTO KBOT_BIZ_TXT_EMBEDDING
        (EMBED_ID, KB_ID, FILE_ID, SECURITY_LEVEL, CHUNK_METADATA, BIZ_METADATA, EMBEDDING, CHUNK_DOC)
        VALUES
        (:1, :2, :3, :4, :5, :6, :7, :8)"""
        
        # 准备批量数据
        data = []
        for idx, embedding in enumerate(embeddings):
            # 将每个嵌入对象转换为元组格式，适合executemany
            # Oracle VECTOR 类型需要 array.array 或 list，不要转换为字符串
            vec_handler = OracleVecHandler()
            vec_array = vec_handler.convert(vec=embedding.embedding, to_string=False)

            # 检查数据长度
            chunk_metadata_json = json.dumps(embedding.chunk_metadata) if embedding.chunk_metadata is not None else None
            biz_metadata_json = json.dumps(embedding.biz_metadata) if embedding.biz_metadata is not None else None

            logger.debug(f"准备插入第 {idx+1} 条记录: embed_id={embedding.embed_id}, "
                        f"chunk_metadata_len={len(chunk_metadata_json) if chunk_metadata_json else 0}, "
                        f"biz_metadata_len={len(biz_metadata_json) if biz_metadata_json else 0}, "
                        f"embedding_type={type(vec_array).__name__}")

            data.append((
                embedding.embed_id,
                kb_id,
                embedding.file_id,
                embedding.security_level,
                chunk_metadata_json,
                biz_metadata_json,
                vec_array,  # 直接传递 array.array，不要转字符串
                embedding.chunk_doc
            ))
        
        try:
            # 使用连接池执行批量插入
            async with self.pool_manager.get_connection_ctx(self.conn_params) as conn:
                cursor = conn.cursor()
                # 使用executemany进行批量插入
                if self.pool_manager._loop is None:
                    logger.error("连接池事件循环不存在")
                    return False
                
                await self.pool_manager._loop.run_in_executor(None, cursor.executemany, sql, data)
                # 提交事务
                await self.pool_manager._loop.run_in_executor(None, conn.commit)
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
                                     similarity_threshold: float = 0.8,
                                     search_top_k: int = 10,
                                     is_summary_search: bool = False,
                                     tags: list[str] = []
                                     ) -> Sequence:
        """Get similar embeddings using vector similarity search.
        
        Args:
            kb_id: Knowledge base ID
            query_vec: Target embedding vector to compare with
            security: Security level
            similarity_threshold: Minimum similarity score (0.0-1.0)
            search_top_k: Maximum number of results to return
            is_summary_search: Whether to search in summary or not
            tags: List of tags to filter by
            
        Returns:
            list of similar embeddings ordered by similarity score
        """
        if self.conn_params is None:
            return []
        
        # Oracle VECTOR_DISTANCE returns a value between 0 and 2, where 0 means identical vectors
        # We need to convert similarity_threshold to a distance value between 0 and 2 
        # to match the VECTOR_DISTANCE function's output range
        # Then use "1 - VECTOR_DISTANCE(EMBEDDING, :query_vec, COSINE) / 2 AS similarity" clause
        # to get the similarity score between 0 and 1

        distance = (1 - similarity_threshold) * 2

        # 基础SQL
        base_sql = """
            SELECT 
                FILE_ID, CHUNK_DOC, CHUNK_METADATA,
                1 - VECTOR_DISTANCE(EMBEDDING, :query_vec, COSINE) / 2 AS similarity
            FROM KBOT_BIZ_TXT_EMBEDDING emb
            WHERE VECTOR_DISTANCE(EMBEDDING, :query_vec, COSINE) <= :distance
            AND KB_ID = :kb_id
            AND SECURITY_LEVEL <= :security
            
            
        """

        # 添加向量和阈值参数
        params = {
            "kb_id": kb_id,
            "query_vec": query_vec,
            "security": security,
            "distance": distance,
            "top_k": search_top_k
        }

        # if is_summary_search:
        #     params["chunk_type"] = ChunkType.SUMMARY.value
        #     base_sql += " AND JSON_VALUE(emb.CHUNK_METADATA, '$.chunk_type' RETURNING NUMBER) = :chunk_type"

        #     logger.debug(f"根据知识库ID {kb_id} 查询摘要向量相似度，阈值: {similarity_threshold}，返回Top {search_top_k} 条记录")
        #     logger.debug(f"过滤条件: JSON_VALUE(emb.CHUNK_METADATA, '$.chunk_type' RETURNING NUMBER) = {ChunkType.SUMMARY.value}")
        # else:
        #     params["chunk_type"] = ChunkType.SUMMARY.value
        #     base_sql += " AND JSON_VALUE(emb.CHUNK_METADATA, '$.chunk_type' RETURNING NUMBER) <> :chunk_type"

        #     logger.debug(f"根据知识库ID {kb_id} 查询摘要向量相似度，阈值: {similarity_threshold}，返回Top {search_top_k} 条记录")
        #     logger.debug(f"过滤条件: JSON_VALUE(emb.CHUNK_METADATA, '$.chunk_type' RETURNING NUMBER) <> {ChunkType.SUMMARY.value}")
            

        # 如果有tag_list，构建多个OR条件
        if tags and len(tags) > 0:
            tag_conditions = []
            for i, tag in enumerate(tags):
                param_name = f"tag_{i}"
                # 正确的JSON_EXISTS绑定语法
                tag_conditions.append(f"JSON_EXISTS(BIZ_METADATA, '$.tags?(@ == $t)' PASSING :{param_name} AS \"t\")")
                params[param_name] = tag
            
            base_sql += " AND (" + " OR ".join(tag_conditions) + ")"

        # 如果tag_list为空或None，不添加tag条件
        
        # 添加排序和限制
        base_sql += """
            ORDER BY similarity DESC
            FETCH FIRST :top_k ROWS ONLY
        """

        return await self.pool_manager.query(self.conn_params, base_sql, params)


    async def full_text_search(self,
                               kb_id: int,
                               keyword: str,
                               security: int,
                               search_top_k: int = 10,
                               similarity_threshold: float = 0.8,
                               tags: list[str] = []
                                ) -> Sequence:
        """Get chunk record by full text search.
        
        Args:
            kb_id: Knowledge base ID
            keyword: Target text to compare with
            security: Security level
            search_top_k: Maximum number of results to return
            simularity_threshold: Minimum similarity score (0.0-1.0)
            tags: List of tags to filter by
            
        Returns:
            list of chunk records
        """
        if self.conn_params is None:
            return []

        # 基础SQL
        base_sql = """
            SELECT FILE_ID, CHUNK_DOC, CHUNK_METADATA, SCORE(1) AS similarity
            FROM KBOT_BIZ_TXT_EMBEDDING
            WHERE KB_ID = :kb_id
            AND SECURITY_LEVEL <= :security
            AND CONTAINS(CHUNK_DOC, REGEXP_REPLACE(:keyword, '\\W+', ' ACCUM '), 1) > 0
            AND SCORE(1) >= :similarity_threshold
        """
        
        # 参数
        params = {
            'kb_id': kb_id,
            'security': security,
            'keyword': keyword,
            'top_k': search_top_k,
            'similarity_threshold': similarity_threshold
        }
        
        # 如果有tag_list，构建多个OR条件
        if tags and len(tags) > 0:
            tag_conditions = []
            for i, tag in enumerate(tags):
                param_name = f"tag_{i}"
                # 正确的JSON_EXISTS绑定语法
                tag_conditions.append(f"JSON_EXISTS(BIZ_METADATA, '$.tags?(@ == $t)' PASSING :{param_name} AS \"t\")")
                params[param_name] = tag
            
            base_sql += " AND (" + " OR ".join(tag_conditions) + ")"

        # 如果tag_list为空或None，不添加tag条件
        
        # 添加排序和限制
        base_sql += """
            ORDER BY similarity DESC
            FETCH FIRST :top_k ROWS ONLY
        """

        # sql = """
        #     SELECT FILE_ID, 
        #             CHUNK_DOC, 
        #             CHUNK_METADATA,
        #             SCORE(1) AS similarity
        #     FROM KBOT_BIZ_TXT_EMBEDDING
        #     WHERE KB_ID = :kb_id
        #     AND SECURITY_LEVEL <= :security
        #     AND CONTAINS(CHUNK_DOC, REGEXP_REPLACE(:keyword, '\\W+', ' ACCUM '), 1) > 0
        #     ORDER BY similarity DESC
        #     FETCH FIRST :top_k ROWS ONLY
        # """

        # # 添加向量和阈值参数
        # params = {
        #     "kb_id": kb_id,
        #     "keyword": keyword,
        #     "security": security,
        #     #"simularity_threshold": simularity_threshold,
        #     "top_k": top_k,
        #     "tag_list": tags
        # }

        result = await self.pool_manager.query(self.conn_params, base_sql, params)

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
        vec_handler = OracleVecHandler()
        params = {
            "embed_id": embed_id,
            "new_chunk": new_chunk,
            "new_embedding": vec_handler.convert(vec=new_embedding, to_string=False)
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
            AND JSON_VALUE(CHUNK_METADATA, '$.chunk_type' RETURNING NUMBER) = :chunk_type
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

    async def update_status_by_chunk_id(self, chunk_id: str, status: int) -> int:
        """更新块状态 - 包括对应的summary chunk（如果存在）"""
        if self.conn_params is None:
            return 0
        
        updated_count = 0
        
        try:
            # 首先获取对应的summary chunk ID（如果存在）
            sql_find_file_and_summary = """
                SELECT e1.FILE_ID, e2.EMBED_ID as summary_embed_id
                FROM KBOT_BIZ_TXT_EMBEDDING e1
                LEFT JOIN KBOT_BIZ_TXT_EMBEDDING e2 ON (
                    e1.FILE_ID = e2.FILE_ID
                    AND JSON_VALUE(e2.CHUNK_METADATA, '$.chunk_type' RETURNING NUMBER) = :chunk_type
                    AND JSON_VALUE(e2.CHUNK_METADATA, '$.source_embed_id') = e1.EMBED_ID
                )
                WHERE e1.EMBED_ID = :chunk_id
            """
            params_find = {
                "chunk_id": chunk_id,
                "chunk_type": ChunkType.SUMMARY.value
            }
            
            result = await self.pool_manager.query(self.conn_params, sql_find_file_and_summary, params_find)
            
            if not result or len(result) == 0:
                return 0  # 原chunk不存在
            
            file_id = result[0][0]
            summary_embed_id = result[0][1]
            
            # 构建需要更新的embed_id列表
            embed_ids_to_update = [chunk_id]
            if summary_embed_id:
                embed_ids_to_update.append(summary_embed_id)
            
            # 批量更新所有相关的chunk状态
            sql_update = """
                UPDATE KBOT_BIZ_TXT_EMBEDDING
                SET STATUS = :status
                WHERE EMBED_ID = :chunk_id
            """
            
            for embed_id in embed_ids_to_update:
                params_update = {
                    "chunk_id": embed_id,
                    "status": status
                }
                count = await self.pool_manager.execute_dml(self.conn_params, sql_update, params_update)
                updated_count += count
                
        except Exception as e:
            logger.error(f"Oracle更新状态失败: {e}")
            return 0
        
        return updated_count
    
    async def get_chunks_by_file_id(self, file_id: str) -> list[KbotBizTxtEmbedding] | None:
        """根据文件ID获取所有chunk"""
        if self.conn_params is None:
            return None
        
        sql = """
            SELECT EMBED_ID, KB_ID, CHUNK_DOC, CHUNK_METADATA, BIZ_METADATA, SECURITY_LEVEL, STATUS
            FROM KBOT_BIZ_TXT_EMBEDDING
            WHERE FILE_ID = :file_id
        """
        params = {
            "file_id": file_id
        }
        result = await self.pool_manager.query(self.conn_params, sql, params)
        if not result or len(result) == 0:
            return None
        
        chunks = []
        for row in result:
            chunk = KbotBizTxtEmbedding(
                embed_id=row[0],
                kb_id=row[1],
                file_id=file_id,
                chunk_doc=row[2],
                chunk_metadata=row[3],
                biz_metadata=row[4],
                embedding=[], # embedding 不返回，防止接口数据过大
                security_level=row[5],
                status=row[6]
            )
            chunks.append(chunk)
            
        return chunks

    async def get_chunk_doc_by_id(self, embed_id: str) -> str | None:
        """根据ID获取chunk文档"""
        if self.conn_params is None:
            return None

        sql = """
            SELECT CHUNK_DOC
            FROM KBOT_BIZ_TXT_EMBEDDING
            WHERE EMBED_ID = :embed_id
        """
        params = {
            "embed_id": embed_id
        }
        result = await self.pool_manager.query(self.conn_params, sql, params)
        if not result or len(result) == 0:
            return None
        
        row = result[0]
        chunk_doc = safe_read_content(row[0])
        return chunk_doc



    async def update_chunk_description(self, embed_id: str, description: str, embeddings: list[float]) -> bool:
        """更新块描述"""
        if self.conn_params is None:
            return False
        
        try:
            # 构建更新SQL
            sql = """
                UPDATE KBOT_BIZ_TXT_EMBEDDING
                SET BIZ_METADATA = JSON_MERGEPATCH(
                    BIZ_METADATA,
                    JSON_OBJECT('description' VALUE :description)
                ),
                EMBEDDING = :embeddings
                WHERE EMBED_ID = :embed_id
            """
            vec_handler = OracleVecHandler()
            params = {
                "embed_id": embed_id,
                "description": description,
                "embeddings": vec_handler.convert(vec=embeddings, to_string=False),
            }
            result = await self.pool_manager.execute_dml(self.conn_params, sql, params)
            return result > 0
            
        except Exception as e:
            logger.error(f"Oracle更新描述失败: {e}")
            return False
        
    
    async def update_tags(self, file_id: str, tags: list[str]) -> bool:
        """根据文件ID批量更新块标签"""
        if self.conn_params is None:
            return False
        
        try:
            # 先查询现有的biz_metadata
            query_sql = """
                SELECT BIZ_METADATA 
                FROM KBOT_BIZ_TXT_EMBEDDING 
                WHERE FILE_ID = :file_id 
                AND ROWNUM = 1
            """
            query_params = {"file_id": file_id}
            
            result = await self.pool_manager.query(self.conn_params, query_sql, query_params)
            
            if not result:
                logger.warning(f"Oracle未找到记录，文件ID: {file_id}")
                return False
            
            # 处理biz_metadata（可能为None、空字符串或有效JSON）
            existing_metadata = {}
            current_metadata = result[0][0]
            
            if current_metadata:
                if isinstance(current_metadata, str):
                    try:
                        existing_metadata = json.loads(current_metadata)
                    except json.JSONDecodeError:
                        logger.warning(f"Oracle解析biz_metadata失败，文件ID: {file_id}，内容: {current_metadata}")
                        existing_metadata = {}
                elif isinstance(current_metadata, dict):
                    existing_metadata = current_metadata
            
            # 更新tags字段，保留其他字段
            existing_metadata["tags"] = tags
            updated_metadata_json = json.dumps(existing_metadata, ensure_ascii=False)
            
            # 更新数据库
            update_sql = """
                UPDATE KBOT_BIZ_TXT_EMBEDDING
                SET BIZ_METADATA = :biz_metadata
                WHERE FILE_ID = :file_id
            """
            update_params = {
                "file_id": file_id,
                "biz_metadata": updated_metadata_json
            }
            
            result = await self.pool_manager.execute_dml(self.conn_params, update_sql, update_params)
            return result > 0
                      
        except Exception as e:
            logger.error(f"Oracle更新标签失败: {e}")
            return False