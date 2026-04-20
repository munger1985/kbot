import json
import re
from loguru import logger
from typing import Any, Sequence
from sqlalchemy import text, select, update, delete, func, and_, or_, Float, literal_column, bindparam
from sqlalchemy.sql import ClauseElement
from dao.entities import TxtChunkEntity
from core.exceptions import DatabaseException, DataNotFoundException
from .base_repo import BaseRepository
from utils.oracle_vec_handler import OracleVecHandler
from utils.common import safe_read_content


class TxtChunkRepository(BaseRepository[TxtChunkEntity]):
    """Oracle SQLAlchemy 2.0 implementation for TxtChunkEntity operations
    Fully compatible with Elasticsearch version interface
    """
    

    async def create(self, chunks: list[TxtChunkEntity]):
        """
        Batch create text chunk records
        Adapt to structured fields: path_names, structure_level, chunk_type

        :param chunks: list of TxtChunkEntity instances to create
        """
        if not chunks:
            logger.warning("Empty chunk list provided for creation, skipping execution")
            return

        try:
            # Convert embeddings to Oracle-compatible format
            vec_handler = OracleVecHandler()

            # Batch processing (Oracle bulk insert best practice: 100 records per batch)
            batch_size = 100
            total_success = 0

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]

                # Convert embeddings to Oracle array format
                for chunk in batch_chunks:
                    chunk.embedding = vec_handler.convert(chunk.embedding) # type: ignore

                # Add batch entities to session
                self.session.add_all(batch_chunks)
                await self.session.flush()  # Execute insert without commit for better performance

                total_success += len(batch_chunks)
                logger.info(
                    f"Successfully batch inserted {len(batch_chunks)} text chunk records to Oracle, "
                    f"progress: {total_success}/{len(chunks)}"
                )

            logger.info(f"Completed all batch insertions, total successful records: {total_success}")

        except Exception as e:
            logger.error("Oracle batch insert text chunks failed", e, max_length=500)
            raise DatabaseException("Oracle batch insert text chunks failed", original_error=e)
        
    async def vector_search(
        self,
        kb_id: int,
        keywords: str,
        query_vec: list[float],
        security: int,
        similarity_threshold: float = 0.5,
        search_top_k: int = 10,
        tags: list[str] = []
    ) -> list[dict[str, Any]]:
        """
        Oracle 向量检索增强版（针对虚拟标题和 search_helper 优化）
        """
        try:
            # 1. 转换阈值
            dist_limit = (1 - (similarity_threshold or 0.5)) * 2
            vec_handler = OracleVecHandler()
            vec_array = vec_handler.convert(vec=query_vec, to_string=False)

            # 2. 基础参数
            all_params: dict[str, Any] = {
                "kb_id": kb_id,
                "security": security,
                "qv": vec_array,
                "dist_limit": dist_limit,
                "top_k": search_top_k,
                "q_keywords": keywords  # LLM 提取的关键词
            }

            # --- 核心改动：文本加权逻辑 ---
            # 权重分配：向量相似度 0.4 + search_helper 命中分 0.4 + header 命中分 0.2
            # 这里的 SCORE(1) 对应 search_helper，SCORE(2) 对应 header
            text_score_sql = "(SCORE(1) * 0.4 + SCORE(2) * 0.2)"
            
            # 使用 CONTAINS 触发 Oracle Text 的全文评分，不设置过滤阈值以免剔除潜在结果
            contains_clauses = """
                AND (CONTAINS(search_helper, :q_keywords, 1) >= 0)
                AND (CONTAINS(header, :q_keywords, 2) >= 0)
            """

            # 3. 核心 SQL 构造
            sql_query = f"""
                SELECT 
                    chunk_id, chunk_type, file_id, kb_id, content, header, chunk_num, chunk_metadata,
                    (
                        ((1 - VECTOR_DISTANCE(embedding, :qv, COSINE)) * 100 * 0.4) 
                        + 
                        {text_score_sql}
                    ) / 100 as similarity_score
                FROM KBOT_BIZ_TXT_EMBEDDING
                WHERE kb_id = :kb_id 
                AND is_active = 1 
                AND security_level <= :security
                AND VECTOR_DISTANCE(embedding, :qv, COSINE) <= :dist_limit
                {contains_clauses}
            """

            # 4. 标签过滤 (JSON)
            if tags:
                tag_clauses = []
                for i, tag in enumerate(tags):
                    t_key = f"t_{i}"
                    tag_clauses.append(f"JSON_EXISTS(biz_metadata, '$.tags[*]?(@ == $t)' PASSING :{t_key} AS \"t\")")
                    all_params[t_key] = tag
                sql_query += f" AND ({' OR '.join(tag_clauses)})"

            sql_query += """
                ORDER BY similarity_score DESC
                FETCH FIRST :top_k ROWS ONLY
            """

            # 5. 执行与结果映射
            stmt = text(sql_query)
            result = await self.session.execute(stmt, all_params)
            chunks = result.fetchall()

            results = []
            for chunk in chunks:
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "chunk_num": chunk.chunk_num,
                    "chunk_type": chunk.chunk_type,
                    "file_id": chunk.file_id,
                    "kb_id": chunk.kb_id,
                    "content": chunk.content,
                    "header": chunk.header,
                    "metadata": chunk.chunk_metadata,
                    "score": float(chunk.similarity_score or 0.0)
                })
            return results

        except Exception as e:
            logger.error(f"Oracle hybrid vector search failed: {str(e)}")
            raise DatabaseException("Exception occurred during vector search execution", original_error=e)

    async def full_text_search(
        self,
        kb_id: int,
        keywords: str,
        security: int,
        search_top_k: int = 10,
        tags: list[str] = []
    ) -> list[dict[str, Any]]:
        try:
            # 1. 清理：保留中英文数字和空格
            clean_keyword = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', keywords.strip())
            words = [w for w in clean_keyword.split() if w]
            if not words:
                return []

            # 2. 组装为 ACCUM 语法，并增加词组精准匹配的权重建议
            # ACCUM 会匹配多个词，命中越多分数越高
            formatted_key = " ACCUM ".join([f"{{{w}}}" for w in words])
            logger.debug(f"Executing Oracle Full-Text Search with: {formatted_key}")

            all_params: dict[str, Any] = {
                "kb_id": kb_id,
                "security": security,
                "keyword": formatted_key,
                "top_k": search_top_k
            }

            # 3. 核心 SQL 逻辑：
            # SCORE(1): search_helper (权重 50%) - 包含全局摘要+虚拟标题+正文前缀
            # SCORE(2): header (权重 30%) - 纯语义虚拟标题
            # SCORE(4): content (权重 20%) - 原始正文
            # 归一化处理：/ 100 得到 0-1 之间的分数
            sql_query = f"""
                SELECT 
                    chunk_id, chunk_type, file_id, kb_id, content, header, chunk_num, chunk_metadata,
                    ((SCORE(1) * 0.5) + (SCORE(2) * 0.3) + (SCORE(4) * 0.2)) / 100 as similarity_score
                FROM KBOT_BIZ_TXT_EMBEDDING
                WHERE kb_id = :kb_id 
                    AND is_active = 1 
                    AND security_level <= :security
                    AND (
                        CONTAINS(search_helper, :keyword, 1) > 0 
                        OR
                        CONTAINS(header, :keyword, 2) > 0
                        OR
                        CONTAINS(content, :keyword, 4) > 0
                    )
                ORDER BY similarity_score DESC
                FETCH FIRST :top_k ROWS ONLY
            """

            # 4. 标签过滤 (JSON)
            if tags:
                tag_clauses = []
                for i, tag in enumerate(tags):
                    t_key = f"tag_{i}"
                    tag_clauses.append(f"JSON_EXISTS(biz_metadata, '$.tags[*]?(@ == $t)' PASSING :{t_key} AS \"t\")")
                    all_params[t_key] = tag
                # 修正 WHERE 注入逻辑，确保在符合 SQL 语法的地方插入
                tag_filter = f" AND ({' OR '.join(tag_clauses)})"
                sql_query = sql_query.replace("WHERE", f"WHERE {tag_filter} AND")

            # 6. 执行查询
            stmt = text(sql_query)
            result = await self.session.execute(stmt, all_params)
            chunks = result.fetchall()

            # 7. 格式化结果
            results = []
            for chunk in chunks:
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "chunk_num": chunk.chunk_num,
                    "chunk_type": chunk.chunk_type,
                    "file_id": chunk.file_id,
                    "kb_id": chunk.kb_id,
                    "content": chunk.content,
                    "header": chunk.header,
                    "metadata": chunk.chunk_metadata,
                    "score": float(chunk.similarity_score or 0.0)
                })
            return results

        except Exception as e:
            logger.error("Oracle full text search failed", e)
            raise DatabaseException("Search execution failed", original_error=e)

    async def get_chunks_by_range(
        self, 
        file_id: str, 
        center_chunk_num: int, 
        window_size: int = 1
    ) -> list[dict]:
        """
        获取指定分片及其前后范围内的分片
        :param window_size: 向上/向下扩展的数量。1表示取 [n-1, n, n+1]
        """
        sql = """
            SELECT chunk_id, chunk_type, file_id, kb_id, content, header, chunk_num, chunk_metadata
            FROM KBOT_BIZ_TXT_EMBEDDING
            WHERE file_id = :file_id 
            AND is_active = 1
            AND chunk_num BETWEEN :min_n AND :max_n
            ORDER BY chunk_num ASC
        """
        params = {
            "file_id": file_id,
            "min_n": center_chunk_num - window_size,
            "max_n": center_chunk_num + window_size
        }
        
        result = await self.session.execute(text(sql), params)
        return [dict(row._mapping) for row in result.fetchall()]

    async def delete_by_file_ids(self, file_ids: list[str]):
        """
        Delete text chunk records by file IDs
        
        :param kb_id: Knowledge base unique identifier
        :param file_ids: list of file IDs to delete chunks for
        :return: Number of deleted records
        :raises DataNotFoundException: If no records found for the given file IDs
        """
        try:
            # Build delete statement
            stmt = (
                delete(TxtChunkEntity)
                .where(TxtChunkEntity.file_id.in_(file_ids))
            )

            # Execute deletion and return affected rows
            await self.session.execute(stmt)
            logger.info(f"Successfully deleted text chunk records, file IDs: {file_ids[:5]}...")

        except DataNotFoundException as e:
            raise e
        except Exception as e:
            logger.error("Oracle delete text chunks by file IDs failed", e, max_length=500)
            raise DatabaseException("Oracle delete text chunks by file IDs failed", original_error=e)

    async def delete_by_kb_id(self, kb_id: int):
        """
        Delete all text chunk records by knowledge base ID
        
        :param kb_id: Knowledge base unique identifier
        :return: Number of deleted records
        :raises DataNotFoundException: If no records found for the given KB ID
        """
        try:
            # Build delete statement
            stmt = delete(TxtChunkEntity).where(TxtChunkEntity.kb_id == kb_id)
            
            # Execute deletion
            await self.session.execute(stmt)            
            logger.info(f"Successfully deleted text chunk records for KB ID: {kb_id}")

        except Exception as e:
            logger.error("Oracle delete text chunks by KB ID failed", e, max_length=500)
            raise DatabaseException("Oracle delete text chunks by KB ID failed", original_error=e)

    async def get_by_file_id(self, file_id: str) -> Sequence[TxtChunkEntity]:
        """
        Get all text chunks for a file (exclude embedding to save memory)
        
        :param file_id: File unique identifier
        :return: list of TxtChunkEntity instances (without embedding data)
        """
        try:
            # Build query
            stmt = (
                select(
                    TxtChunkEntity
                )
                .where(TxtChunkEntity.file_id == file_id)
                .order_by(
                    TxtChunkEntity.chunk_num.asc()
                )
            )

            result = await self.session.execute(stmt)
            return result.scalars().all()

        except Exception as e:
            logger.error(f"Oracle get file chunks failed for file ID {file_id}", e, max_length=500)
            raise DatabaseException(f"Oracle get text chunks for file {file_id} failed", original_error=e)

    async def update_chunk(self, chunk_id: str, new_content: str, new_embedding: list[float]) -> bool:
        """
        Update text chunk content by ID
        
        :param chunk_id: Text chunk unique identifier
        :param new_content: New content for the text chunk
        :return: True if update successful
        :raises DataNotFoundException: If chunk ID not found
        """
        oracle_embedding = OracleVecHandler().convert(new_embedding)
        try:
            stmt = (
                update(TxtChunkEntity)
                .where(TxtChunkEntity.chunk_id == chunk_id)
                .values(content=new_content)
                .values(embedding=oracle_embedding)
                .returning(func.count(TxtChunkEntity.chunk_id))
            )

            rowcount = await self.session.execute(stmt)
            
            if rowcount == 0:
                raise DataNotFoundException(f"Chunk ID not found: {chunk_id}")

            logger.info(f"Successfully updated content for chunk {chunk_id}")
            return True

        except DataNotFoundException as e:
            raise e
        except Exception as e:
            logger.error("Oracle update chunk failed", e, max_length=500)
            raise DatabaseException("Oracle update text chunk content failed", original_error=e)

    async def delete(self, chunk_id: str):
        """
        Delete text chunk record by ID
        
        :param chunk_id: Text chunk unique identifier
        :return: True if deletion successful
        :raises DataNotFoundException: If chunk ID not found
        """
        try:
            stmt = delete(TxtChunkEntity).where(TxtChunkEntity.chunk_id == chunk_id)
            await self.session.execute(stmt)
            logger.info(f"Successfully deleted chunk {chunk_id}")

        except DataNotFoundException as e:
            raise e
        except Exception as e:
            logger.error("Oracle delete chunk failed", e, max_length=500)
            raise DatabaseException("Oracle delete text chunk failed", original_error=e)
        
    async def get_content(self, chunk_id: str) -> str:
        """
        Get text chunk content by ID
        
        :param chunk_id: Text chunk unique identifier
        :return: Text chunk content
        :raises DataNotFoundException: If chunk ID not found
        """
        try:
            stmt = select(TxtChunkEntity.content).where(TxtChunkEntity.chunk_id == chunk_id)
            result = await self.session.execute(stmt)
            content = result.scalar_one()
            if content is None:
                raise DataNotFoundException(f"Chunk ID not found: {chunk_id}")
            return safe_read_content(content)
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            logger.error("Oracle get chunk content failed", e, max_length=500)
            raise DatabaseException("Oracle get text chunk content failed", original_error=e)

    async def update_tag(self, file_id: str, tags: list[str]):
        """
        Update tags for text chunks by file ID
        
        :param file_id: File unique identifier
        :param tags: list of tags to update
        :raises DataNotFoundException: If no records found for the given file ID
        """
        try:
            # Use raw SQL with correct Oracle JSON syntax
            # Oracle json_object requires KEY/VALUE keywords
            tags_json = json.dumps(tags, ensure_ascii=False)
            sql = text("""
                UPDATE KBOT_BIZ_TXT_EMBEDDING 
                SET biz_metadata = JSON_MERGEPATCH(
                    NVL(biz_metadata, '{}'),
                    JSON_OBJECT('tags' VALUE :tags FORMAT JSON)
                )
                WHERE file_id = :file_id
            """)
                        
            # Execute update
            result = await self.session.execute(sql, {"file_id": file_id, "tags": tags_json})
            
            if result.rowcount == 0: # type: ignore
                logger.warning(f"No records found for file ID: {file_id}")
            else:  
                logger.info(f"Successfully updated tags for file {file_id}: {tags}")

        except Exception as e:
            logger.error(f"Failed to update file tags for file ID {file_id}: {e}", exc_info=True)
            raise DatabaseException("Failed to update file tags", original_error=e)
          
    async def update_description(self, chunk_id: str, description: str, new_embedding: list[float]):
        """
        Update description for text chunk by chunk ID
        
        :param chunk_id: Chunk unique identifier
        :param description: Chunk description to update
        :param new_embedding: New embedding vector for the chunk
        :raises DataNotFoundException: If no records found for the given chunk ID
        """
        try:
            # Use raw SQL with correct Oracle JSON syntax
            # Oracle json_object requires KEY/VALUE keywords
            oracle_embedding = OracleVecHandler().convert(new_embedding)
            sql = text("""
                UPDATE KBOT_BIZ_TXT_EMBEDDING 
                SET biz_metadata = JSON_MERGEPATCH(
                    NVL(biz_metadata, JSON_OBJECT()),
                    JSON_OBJECT('description' VALUE :description)
                ),
                embedding = :embedding
                WHERE chunk_id = :chunk_id
            """)
            
            # Execute update
            result = await self.session.execute(sql, {
                "chunk_id": chunk_id,
                "description": description,
                "embedding": oracle_embedding
            })
            
            if result.rowcount == 0: # type: ignore
                raise DataNotFoundException(f"No records found for chunk ID: {chunk_id}")

            logger.info(f"Successfully updated description for chunk {chunk_id}")

        except DataNotFoundException as e:
            raise e
        except Exception as e:
            logger.error("Oracle update description failed", e, max_length=500)
            raise DatabaseException("Oracle update text chunk description failed", original_error=e)
        
    async def toggle_active_status(self, chunk_id: str, is_active: bool):
        """
        Toggle active status for text chunk by chunk ID
        
        :param chunk_id: Chunk unique identifier
        :param is_active: Active status to update
        :raises DataNotFoundException: If no records found for the given file ID
        """
        try:
            # Build update statement using pure SQLAlchemy 2.0 ORM syntax
            # Oracle JSON_MERGEPATCH equivalent in SQLAlchemy
            update_stmt = (
                update(TxtChunkEntity)
                .where(TxtChunkEntity.chunk_id == chunk_id)
                .values(is_active=is_active)
            )
            
            # Execute update using SQLAlchemy ORM
            await self.session.execute(update_stmt)
            logger.info(f"Successfully updated active status for chunk {chunk_id}, is_active: {is_active}")
        
        except Exception as e:
            logger.error("Oracle update active status failed", e, max_length=500)
            raise DatabaseException("Oracle update text chunk active status failed", original_error=e)