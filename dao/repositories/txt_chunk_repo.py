import math
import re
from loguru import logger
from typing import Any, Sequence
from sqlalchemy import text, select, update, delete, func, and_, or_, Float, literal_column, bindparam
from sqlalchemy.sql import ClauseElement
from dao.entities import TxtChunkEntity
from core.exceptions import DatabaseException, DataNotFoundException, safe_log_error
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
                    # Log first chunk details for debugging
                    if i == 0 and chunk == batch_chunks[0]:
                        logger.debug(f"First chunk before insert: chunk_id={chunk.chunk_id}")
                        logger.debug(f"  content type={type(chunk.content)}, content[:200]={repr(chunk.content[:200])}")
                        logger.debug(f"  chunk_metadata type={type(chunk.chunk_metadata)}, value={str(chunk.chunk_metadata)[:200]}")
                        logger.debug(f"  biz_metadata type={type(chunk.biz_metadata)}, value={str(chunk.biz_metadata)[:200]}")
                        logger.debug(f"  embedding type={type(chunk.embedding)}, len={len(chunk.embedding) if hasattr(chunk.embedding, '__len__') else 'N/A'}")

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
            safe_log_error("Oracle batch insert text chunks failed", e, max_length=500)
            raise DatabaseException("Oracle batch insert text chunks failed", original_error=e)
        
    async def vector_search(
        self,
        kb_id: int,
        keywords: str,
        query_vec: list[float],
        security: int,
        similarity_threshold: float = 0.5,
        search_top_k: int = 10,
        tags: list[str] = [],
        path_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Vector similarity search (Oracle version)
        Note: Oracle 21c+ or vector extensions required for vector operations
        
        :param query_vec: Query embedding vector
        :param security: Security level filter (records with level <= security will be returned)
        :param similarity_threshold: Minimum similarity score threshold (0.0-1.0)
        :param search_top_k: Maximum number of results to return
        :param tags: list of tags to filter results
        :param path_filter: Path name filter (match any in path_names array)
        :return: list of search results with similarity scores
        """
        try:
            # 1. 转换阈值：相似度 0.8 -> 距离 (1-0.8)*2 = 0.4
            dist_limit = (1 - (similarity_threshold or 0.5)) * 2

            # 将 list[float] 类型的向量转换为 Oracle 数组类型 array.array
            vec_handler = OracleVecHandler()
            vec_array = vec_handler.convert(vec=query_vec, to_string=False)

            # 2. 核心参数字典
            all_params: dict[str, Any] = {
                "kb_id": kb_id,
                "security": security,
                "qv": vec_array,
                "dist_limit": dist_limit,
                "top_k": search_top_k
            }

            # 2. 路径加权逻辑优化
            path_weight_sql = "0"
            contains_clause = ""
            

            # 使用传入的、经过 LLM 优化的关键词
            all_params["q_keywords"] = keywords
            # 赋予路径命中 0.7 的权重系数
            path_weight_sql = "SCORE(99) * 0.7"
            # 在 WHERE 中增加一个不带过滤效果的 CONTAINS 用于触发 SCORE 计算
            contains_clause = " AND (CONTAINS(path_names, :q_keywords, 99) >= 0) "

            # 3. 核心 SQL：(向量分 30%) + (路径分 70%), 归一化：向量分转为 0-100 再计算
            sql_query = f"""
                SELECT 
                    chunk_id, file_id, content, path_names, structure_level, chunk_metadata,
                    (
                        ((1 - VECTOR_DISTANCE(embedding, :qv, COSINE)) * 100 * 0.3) 
                        + 
                        ({path_weight_sql})
                    ) / 100 as similarity_score
                FROM KBOT_BIZ_TXT_EMBEDDING
                WHERE kb_id = :kb_id 
                AND is_active = 1 
                AND security_level <= :security
                AND VECTOR_DISTANCE(embedding, :qv, COSINE) <= :dist_limit
                {contains_clause}
            """

            # 4. 路径硬过滤 (path_filter)
            if path_filter:
                # 注意：这里继续沿用全文检索的清理逻辑以保持一致
                clean_path = re.sub(r'[^\w\s]', '', path_filter.strip())
                all_params["p_filter"] = clean_path
                sql_query += " AND CONTAINS(path_names, :p_filter, 98) > 0 "

            # 5. 动态拼接标签过滤
            if tags:
                tag_clauses = []
                for i, tag in enumerate(tags):
                    t_key = f"t_{i}"
                    tag_clauses.append(f"JSON_EXISTS(biz_metadata, '$.tags[*]?(@ == $t)' PASSING :{t_key} AS \"t\")")
                    all_params[t_key] = tag
                if tag_clauses:
                    sql_query += f" AND ({' OR '.join(tag_clauses)})"

            sql_query += """
                ORDER BY similarity_score DESC
                FETCH FIRST :top_k ROWS ONLY
            """

            # 6. 执行查询
            stmt = text(sql_query)
            result = await self.session.execute(stmt, all_params)
            chunks = result.fetchall()

            # 7. 格式化结果
            results = []
            for chunk in chunks:
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "file_id": chunk.file_id,
                    "content": chunk.content,
                    "path_names": chunk.path_names,
                    "structure_level": chunk.structure_level,
                    "metadata": chunk.chunk_metadata,
                    "score": float(chunk.similarity_score or 0.0)
                })
            return results

        except Exception as e:
            logger.error(f"Oracle vector search failed: {str(e)}")
            raise DatabaseException("Exception occurred during vector search execution", original_error=e)

    async def full_text_search(
        self,
        kb_id: int,
        keywords: str,
        security: int,
        search_top_k: int = 10,
        tags: list[str] = [],
        path_filter: str | None = None
    ) -> list[dict[str, Any]]:
        try:
            # 1. 关键词预处理：处理空格和特殊字符
            clean_keyword = re.sub(r'[^\w\s]', '', keywords.strip())
            formatted_key = re.sub(r'\s+', ' ACCUM ', clean_keyword)

            all_params: dict[str, Any] = {
                "kb_id": kb_id,
                "security": security,
                "keyword": formatted_key,
                "top_k": search_top_k
            }
            
            # 2. 动态路径过滤逻辑
            path_clause = ""
            if path_filter:
                # 如果提供了 path_filter，则要求路径中必须包含此关键词
                clean_path = re.sub(r'[^\w\s]', '', path_filter.strip())
                all_params["p_filter"] = clean_path
                path_clause = " AND CONTAINS(path_names, :p_filter, 3) > 0 "

            # 3. 核心 SQL：双重加权评分, 路径命中占 70% 权重，正文命中占 30%
            sql_query = f"""
                SELECT 
                    chunk_id, file_id, content, path_names, structure_level, chunk_metadata,
                    ((SCORE(1) * 0.3) + (SCORE(2) * 0.7)) / 100 as similarity_score
                FROM KBOT_BIZ_TXT_EMBEDDING
                WHERE kb_id = :kb_id 
                    AND is_active = 1 
                    AND security_level <= :security
                    AND (
                        CONTAINS(content, :keyword, 1) > 0 
                        OR
                        CONTAINS(path_names, :keyword, 2) > 0
                    )
                    {path_clause}
                ORDER BY similarity_score DESC
                FETCH FIRST :top_k ROWS ONLY
            """

            # 4. 标签过滤
            if tags:
                tag_clauses = []
                for i, tag in enumerate(tags):
                    t_key = f"tag_{i}"
                    tag_clauses.append(f"JSON_EXISTS(biz_metadata, '$.tags[*]?(@ == $t)' PASSING :{t_key} AS \"t\")")
                    all_params[t_key] = tag
                sql_query = sql_query.replace("WHERE", f"WHERE ({' OR '.join(tag_clauses)}) AND")

            # 5. 执行查询
            stmt = text(sql_query)
            result = await self.session.execute(stmt, all_params)
            chunks = result.fetchall()

            # 6. 格式化结果以适配 TxtBaseSearchResult
            results = []
            for chunk in chunks:
                # 从 chunk_metadata (JSON) 中安全提取 docling 的具体字段
                meta = chunk.chunk_metadata or {}
                
                results.append({
                    "chunk_id": str(chunk.chunk_id),
                    "file_id": chunk.file_id,
                    "content": chunk.content,
                    "path_names": chunk.path_names or "",
                    "structure_level": int(chunk.structure_level or 0),
                    "node_path": meta.get("node_path", ""),
                    "page_num": meta.get("page_num", 0),
                    "chunk_num": meta.get("chunk_num", 0),
                    "sub_index": meta.get("sub_index", 0),
                    "chunk_type": meta.get("chunk_type", "text"),
                    "score": float(chunk.similarity_score or 0.0),
                    "search_type": "fulltext"
                })
            return results

        except Exception as e:
            safe_log_error("Oracle full text search failed", e)
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
            SELECT chunk_id, file_id, content, path_names, structure_level, chunk_metadata
            FROM KBOT_BIZ_TXT_EMBEDDING
            WHERE file_id = :file_id 
            AND is_active = 1
            AND JSON_VALUE(chunk_metadata, '$.chunk_num' RETURNING NUMBER) 
                BETWEEN :min_n AND :max_n
            ORDER BY JSON_VALUE(chunk_metadata, '$.chunk_num' RETURNING NUMBER) ASC
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
            safe_log_error("Oracle delete text chunks by file IDs failed", e, max_length=500)
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
            safe_log_error("Oracle delete text chunks by KB ID failed", e, max_length=500)
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
                    text("JSON_VALUE(chunk_metadata, '$.chunk_num') ASC"),
                    text("JSON_VALUE(chunk_metadata, '$.sub_index') ASC")
                )
            )

            result = await self.session.execute(stmt)
            return result.scalars().all()

        except Exception as e:
            safe_log_error(f"Oracle get file chunks failed for file ID {file_id}", e, max_length=500)
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
            safe_log_error("Oracle update chunk failed", e, max_length=500)
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
            safe_log_error("Oracle delete chunk failed", e, max_length=500)
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
            safe_log_error("Oracle get chunk content failed", e, max_length=500)
            raise DatabaseException("Oracle get text chunk content failed", original_error=e)

    async def update_tag(self, file_id: str, tags: list[str]):
        """
        Update tags for text chunks by file ID
        
        :param file_id: File unique identifier
        :param tags: list of tags to update
        :raises DataNotFoundException: If no records found for the given file ID
        """
        try:
            # Build update statement using pure SQLAlchemy 2.0 ORM syntax
            # Oracle JSON_MERGEPATCH equivalent in SQLAlchemy
            update_stmt = (
                update(TxtChunkEntity)
                .where(TxtChunkEntity.file_id == file_id)
                .values(
                    biz_metadata=func.json_mergepatch(
                        func.nvl(TxtChunkEntity.biz_metadata, func.json_object()),  # Handle null biz_metadata
                        func.json_object('tags', tags)  # Create JSON object with tags array
                    )
                )
                .execution_options(synchronize_session="fetch")
                .returning(func.count(TxtChunkEntity.chunk_id))
            )

            # Execute update using SQLAlchemy ORM
            await self.session.execute(update_stmt)
            logger.info(f"Successfully updated tags for file {file_id}: {tags}")

        except Exception as e:
            safe_log_error("Oracle update tags failed", e, max_length=500)
            raise DatabaseException("Oracle update text chunk tags failed", original_error=e)
          
    async def update_description(self, chunk_id: str, description: str, new_embedding: list[float]):
        """
        Update description for text chunk by chunk ID
        
        :param chunk_id: Chunk unique identifier
        :param description: Chunk description to update
        :raises DataNotFoundException: If no records found for the given file ID
        """
        try:
            # Build update statement using pure SQLAlchemy 2.0 ORM syntax
            # Oracle JSON_MERGEPATCH equivalent in SQLAlchemy
            oracle_embedding = OracleVecHandler().convert(new_embedding)
            update_stmt = (
                update(TxtChunkEntity)
                .where(TxtChunkEntity.chunk_id == chunk_id)
                .values(
                    biz_metadata=func.json_mergepatch(
                        func.nvl(TxtChunkEntity.biz_metadata, func.json_object()),  # Handle null biz_metadata
                        func.json_object('description', description)  # Create JSON object with description
                    )
                )
                .values(embedding=oracle_embedding)
                .execution_options(synchronize_session="fetch")
                .returning(func.count(TxtChunkEntity.chunk_id))
            )

            # Execute update using SQLAlchemy ORM
            updated_count = await self.session.execute(update_stmt)
            
            if updated_count == 0:
                raise DataNotFoundException(f"No records found for chunk ID: {chunk_id}")

            logger.info(f"Successfully updated description for chunk {chunk_id}, affected {updated_count} records")

        except DataNotFoundException as e:
            raise e
        except Exception as e:
            safe_log_error("Oracle update description failed", e, max_length=500)
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
            safe_log_error("Oracle update active status failed", e, max_length=500)
            raise DatabaseException("Oracle update text chunk active status failed", original_error=e)