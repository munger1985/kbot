import json
import re
import asyncio
from loguru import logger
from typing import Any, Sequence
from sqlalchemy import text, select, update, delete, func, and_, or_, Float, literal_column, bindparam
import array
from dao.entities import TxtChunkEntity
from core.exceptions import DatabaseException, DataNotFoundException
from .base_repo import BaseRepository
from utils.codec import OracleVecHandler
from utils.thread import safe_read_content


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
        query_vec: "array.array[float]",
        security: int,
        similarity_threshold: float = 0.5,
        search_top_k: int = 20,
        tags: list[str] = [],
    ) -> list[dict[str, Any]]:
        """
        独立的向量语义检索（余弦相似度），不做全文混合。

        Args:
            kb_id: 知识库 ID
            query_vec: 查询向量（已转换为 Oracle array 格式）
            security: 安全等级上限
            similarity_threshold: 相似度阈值，默认 0.5
            search_top_k: 召回数量
            tags: 标签硬过滤
        """
        try:
            dist_limit = (1 - (similarity_threshold or 0.5)) * 2

            all_params: dict[str, Any] = {
                "kb_id": kb_id,
                "security": security,
                "qv": query_vec,
                "dist_limit": dist_limit,
                "top_k": search_top_k,
            }

            conditions = [
                "kb_id = :kb_id",
                "is_active = 1",
                "security_level <= :security",
                "VECTOR_DISTANCE(embedding, :qv, COSINE) <= :dist_limit",
            ]

            if tags:
                tag_clauses = []
                for i, tag in enumerate(tags):
                    t_key = f"t_{i}"
                    tag_clauses.append(
                        f'JSON_EXISTS(biz_metadata, \'$.tags[*]?(@ == $t)\' PASSING :{t_key} AS "t")'
                    )
                    all_params[t_key] = tag
                conditions.append(f"({' OR '.join(tag_clauses)})")

            where_clause = " AND ".join(conditions)

            sql_query = f"""
                SELECT
                    chunk_id, chunk_type, file_id, kb_id, content, header, chunk_num,
                    chunk_metadata, biz_metadata, search_helper,
                    (1 - VECTOR_DISTANCE(embedding, :qv, COSINE)) as similarity_score
                FROM KBOT_BIZ_TXT_EMBEDDING
                WHERE {where_clause}
                ORDER BY similarity_score DESC, chunk_id ASC
                FETCH FIRST :top_k ROWS ONLY
            """

            logger.debug(f"[TxtChunkRepo] vector_search top_k={search_top_k}, threshold={similarity_threshold}")
            stmt = text(sql_query)
            result = await self.session.execute(stmt, all_params)
            chunks = result.fetchall()

            return self._format_chunk_results(chunks)

        except Exception as e:
            logger.error(f"Oracle vector search failed: {str(e)}")
            raise DatabaseException("Exception occurred during vector search execution", original_error=e)

    async def full_text_search(
        self,
        kb_id: int,
        keywords: str,
        security: int,
        search_top_k: int = 20,
        tags: list[str] = [],
    ) -> list[dict[str, Any]]:
        """
        独立的 Oracle Text 全文检索（CONTAINS + ACCUM），不做向量混合。

        检索字段与权重：
          - SCORE(1): search_helper (权重 50%) — 全局摘要 + 虚拟标题 + 内容前缀
          - SCORE(2): header (权重 30%) — LLM 生成的虚拟标题
          - SCORE(3): content (权重 20%) — 原始正文全文

        Args:
            kb_id: 知识库 ID
            keywords: 已清洗的关键词串（空格分隔，不含大括号）
            security: 安全等级上限
            search_top_k: 召回数量
            tags: 标签硬过滤
        """
        if not keywords or not keywords.strip():
            return []

        try:
            words = [w for w in keywords.split() if w]
            if not words:
                return []

            formatted_key = " ACCUM ".join([f"{{{w}}}" for w in words])
            logger.debug(f"[TxtChunkRepo] full_text_search: {formatted_key[:120]}")

            all_params: dict[str, Any] = {
                "kb_id": kb_id,
                "security": security,
                "keyword": formatted_key,
                "top_k": search_top_k,
            }

            conditions = [
                "kb_id = :kb_id",
                "is_active = 1",
                "security_level <= :security",
                "(CONTAINS(search_helper, :keyword, 1) > 0 "
                "OR CONTAINS(header, :keyword, 2) > 0 "
                "OR CONTAINS(content, :keyword, 3) > 0)",
            ]

            if tags:
                tag_clauses = []
                for i, tag in enumerate(tags):
                    t_key = f"tag_{i}"
                    tag_clauses.append(
                        f'JSON_EXISTS(biz_metadata, \'$.tags[*]?(@ == $t)\' PASSING :{t_key} AS "t")'
                    )
                    all_params[t_key] = tag
                conditions.append(f"({' OR '.join(tag_clauses)})")

            where_clause = " AND ".join(conditions)

            sql_query = f"""
                SELECT
                    chunk_id, chunk_type, file_id, kb_id, content, header, chunk_num,
                    chunk_metadata, biz_metadata, search_helper,
                    (SCORE(1) * 0.5 + SCORE(2) * 0.3 + SCORE(3) * 0.2) as similarity_score
                FROM KBOT_BIZ_TXT_EMBEDDING
                WHERE {where_clause}
                ORDER BY similarity_score DESC, chunk_id ASC
                FETCH FIRST :top_k ROWS ONLY
            """

            stmt = text(sql_query)
            result = await self.session.execute(stmt, all_params)
            chunks = result.fetchall()

            return self._format_chunk_results(chunks)

        except Exception as e:
            logger.error(f"Oracle full text search failed: {str(e)}")
            raise DatabaseException("Search execution failed", original_error=e)

    @staticmethod
    def _format_chunk_results(chunks) -> list[dict[str, Any]]:
        """将 SQL 查询结果统一格式化为 dict 列表"""
        results: list[dict[str, Any]] = []
        for c in chunks:
            results.append({
                "chunk_id": c.chunk_id,
                "chunk_num": c.chunk_num,
                "chunk_type": c.chunk_type,
                "file_id": c.file_id,
                "kb_id": c.kb_id,
                "content": c.content,
                "header": c.header,
                "search_helper": getattr(c, "search_helper", ""),
                "metadata": c.chunk_metadata,
                "biz_metadata": c.biz_metadata,
                "score": float(c.similarity_score or 0.0),
            })
        return results

    async def native_hybrid_search(
            self,
            kb_id: int,
            keywords: str,
            query_vec: array.array[float],
            security: int,
            has_vec: int,
            similarity_threshold: float = 0.5,
            search_top_k: int = 30,
            tags: list[str] = []
        ) -> list[dict[str, Any]]:
            """
            Oracle 26ai 内核级混合查询（动态 SQL 安全修正版）
            """            
            # 1. 基础过滤条件
            conditions = [
                "kb_id = :kb_id",
                "is_active = 1",
                "security_level <= :security"
            ]
            
            all_params: dict[str, Any] = {
                "kb_id": kb_id,
                "security": security,
                "top_k": search_top_k,
            }

            # 2. 动态 JSON 标签硬过滤
            if tags:
                tag_clauses = []
                for i, tag in enumerate(tags):
                    t_key = f"t_{i}"
                    tag_clauses.append(f"JSON_EXISTS(biz_metadata, '$.tags[*]?(@ == $t)' PASSING :{t_key} AS \"t\")")
                    all_params[t_key] = tag
                conditions.append(f"({' OR '.join(tag_clauses)})")

            # 3. 核心：动态构建权重分数和搜索条件
            score_parts = []
            search_parts = []

            # 处理向量检索分支
            if has_vec == 1 and query_vec:
                score_parts.append("(1 - VECTOR_DISTANCE(embedding, :qv, COSINE)) * 100 * 0.4")
                search_parts.append("VECTOR_DISTANCE(embedding, :qv, COSINE) <= :dist_limit")
                all_params["qv"] = query_vec
                all_params["dist_limit"] = (1 - (similarity_threshold or 0.5)) * 2

            # 处理文本检索分支（严格过滤空字符串，并做基础清洗）
            clean_keywords = keywords.strip() if keywords else ""
            if clean_keywords:
                words = [w.strip() for w in clean_keywords.split() if w.strip()]

                # 核心拼装：有且仅有一层大括号包裹每一个词，中间用 ACCUM 拼接
                if words:
                    final_oracle_text_query = " ACCUM ".join([f"{{{w}}}" for w in words])
                else:
                    final_oracle_text_query = ""
                
                score_parts.append("SCORE(1) * 0.4 + SCORE(2) * 0.2")
                search_parts.append("(CONTAINS(search_helper, :q_keywords, 1) > 0 OR CONTAINS(header, :q_keywords, 2) > 0)")
                all_params["q_keywords"] = final_oracle_text_query

            # 防御：如果既没有向量也没有文本，直接返回空
            if not score_parts:
                return []

            # 组装相似度得分字段
            similarity_score_sql = f"({' + '.join(score_parts)}) / 100"
            
            # 组装 WHERE 核心搜索条件 (向量与文本之间是 OR 关系)
            conditions.append(f"({' OR '.join(search_parts)})")
            where_clause = " AND ".join(conditions)

            # 4. 构建最终纯净 SQL
            sql_query = f"""
                SELECT 
                    chunk_id, chunk_type, file_id, kb_id, content, header, chunk_num, chunk_metadata, biz_metadata,
                    {similarity_score_sql} as similarity_score
                FROM KBOT_BIZ_TXT_EMBEDDING
                WHERE {where_clause}
                ORDER BY similarity_score DESC, chunk_id ASC
                FETCH FIRST :top_k ROWS ONLY
            """

            logger.debug(f"[TxtChunkRepo] 执行 SQL: {sql_query}")
            
            stmt = text(sql_query)
            result = await self.session.execute(stmt, all_params)
            chunks = result.fetchall()
            
            return [{
                "chunk_id": c.chunk_id,
                "chunk_num": c.chunk_num,
                "chunk_type": c.chunk_type,
                "file_id": c.file_id,
                "kb_id": c.kb_id,
                "content": c.content,
                "header": c.header,
                "metadata": c.chunk_metadata,
                "biz_metadata": c.biz_metadata,
                "score": float(c.similarity_score or 0.0)
            } for c in chunks]

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
            SELECT chunk_id, chunk_type, content
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
        # 🛡️ 关键修复：对 row._mapping 进行键名清洗，防止 Oracle 驱动返回带单引号的列名（如 'CHUNK_ID' 变成 chunk_id）
        cleaned_rows = []
        for row in result.fetchall():
            row_dict = {}
            for k, v in row._mapping.items():
                clean_key = str(k).lower().strip("'\"")
                row_dict[clean_key] = v
            cleaned_rows.append(row_dict)
        return cleaned_rows

    async def get_chunks_by_ranges_batch(
        self,
        queries: list[tuple[str, int]],
        window_size: int = 1,
    ) -> dict[tuple[str, int], list[dict]]:
        """
        Phase 4: 批量获取多个 chunk 的邻居分片，避免 N+1 查询。

        Args:
            queries: [(file_id, chunk_num), ...] 需要查询邻居的 chunk 列表
            window_size: 窗口半径，1 表示取 [n-1, n, n+1]

        Returns:
            {(file_id, chunk_num): [dict(chunk_id, chunk_num, content), ...], ...}
        """
        if not queries:
            return {}

        # 收集所有唯一的 (file_id, chunk_num) 范围
        file_ranges: dict[str, tuple[int, int]] = {}
        for file_id, cnum in queries:
            cmin = cnum - window_size
            cmax = cnum + window_size
            if file_id not in file_ranges:
                file_ranges[file_id] = (cmin, cmax)
            else:
                existing_min, existing_max = file_ranges[file_id]
                file_ranges[file_id] = (
                    min(existing_min, cmin),
                    max(existing_max, cmax),
                )

        # 为每个 file_id 构建一条 SQL，批量拉取
        all_rows: dict[str, list[dict]] = {}  # file_id → [rows]
        for file_id, (global_min, global_max) in file_ranges.items():
            sql = """
                SELECT chunk_id, chunk_num, content
                FROM KBOT_BIZ_TXT_EMBEDDING
                WHERE file_id = :file_id
                AND is_active = 1
                AND chunk_num BETWEEN :min_n AND :max_n
                ORDER BY chunk_num ASC
            """
            params = {
                "file_id": file_id,
                "min_n": global_min,
                "max_n": global_max,
            }
            result = await self.session.execute(text(sql), params)
            rows = []
            for row in result.fetchall():
                row_dict = {}
                for k, v in row._mapping.items():
                    clean_key = str(k).lower().strip("'\"")
                    row_dict[clean_key] = v
                rows.append(row_dict)
            all_rows[file_id] = rows

        # 将结果按 queries 分组
        result_map: dict[tuple[str, int], list[dict]] = {}
        for file_id, cnum in queries:
            rows = all_rows.get(file_id, [])
            # 过滤出在 [cnum - window_size, cnum + window_size] 范围内的
            neighbors = [
                r
                for r in rows
                if abs(r.get("chunk_num", 0) - cnum) <= window_size
            ]
            result_map[(file_id, cnum)] = neighbors

        return result_map

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

    async def update_tags(self, file_id: str, tags: list[str]):
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
        
    async def get_chunks_by_ids(self, chunk_ids: list[str], security_level: int) -> dict[str, dict]:
        """
        根据文本块ID列表批量获取文本块详情，并进行安全级别过滤。
        
        :param chunk_ids: 文本块唯一标识符列表
        :param security_level: 安全级别过滤阈值 (只召回 <= 该级别的文本块)
        :return: 字典格式，键为纯净的 chunk_id 字符串，值为大小写归一化后的实体字典
        """
        if not chunk_ids:
            return {}

        try:
            # ========================================================
            # 核心优化一：无损动态分批，优雅攻克 Oracle IN 1000 条上限限制
            # ========================================================
            # 不采用暴力的 [:1000] 截断策略，而是通过合理切片加 asyncio.gather 并发无损召回
            batch_size = 900
            batches = [chunk_ids[i:i + batch_size] for i in range(0, len(chunk_ids), batch_size)]
            
            chunk_dict = {}

            # 定义内部异步闭包，专门负责单个分批切片的安全驱动与清洗
            async def fetch_batch(sub_ids: list[str]) -> dict[str, dict]:
                # 构建局部动态绑定字典，防 SQL 注入
                sub_bind_params = {f"cid_{i}": cid for i, cid in enumerate(sub_ids)}
                sub_bind_params["security_level"] = security_level # type: ignore
                
                # 构造符合当前批次长度的 IN 占位符片段
                in_clause = ", ".join(f":cid_{i}" for i in range(len(sub_ids)))

                # 原生 SQL 显式指定（大小写与 Oracle 数据字典对齐）
                sub_sql = text(f"""
                    SELECT * FROM KBOT_BIZ_TXT_EMBEDDING
                    WHERE CHUNK_ID IN ({in_clause})
                    AND SECURITY_LEVEL <= :security_level
                """)
                
                # 执行当前批次的异步会话游标
                res = await self.session.execute(sub_sql, sub_bind_params)
                sub_rows = res.fetchall()
                
                sub_batch_dict = {}
                for row in sub_rows:
                    # ========================================================
                    # 核心优化二：大小写免疫与多余单双引号剥离（从根源斩断 "'kb_id'" 异常）
                    # ========================================================
                    row_lowercase_dict = {}
                    for k, v in row._mapping.items():
                        # 🛡️ 终极清洗：先转小写，然后彻底剔除由于 Oracle 驱动反射或者多表别名导致的各类单、双引号
                        clean_key = str(k).lower().strip("'\"")
                        row_lowercase_dict[clean_key] = v
                    
                    # 提取归一化后、纯净安全的文本块主键
                    real_chunk_id = row_lowercase_dict.get("chunk_id")
                    if real_chunk_id:
                        # 🛡️ 辅助防护：确保存储的 kb_id 在底层就归一化为纯净的 int 类型，拒绝单双引号字符串残留
                        if "kb_id" in row_lowercase_dict and row_lowercase_dict["kb_id"] is not None:
                            try:
                                row_lowercase_dict["kb_id"] = int(row_lowercase_dict["kb_id"])
                            except (ValueError, TypeError):
                                logger.warning(f"[TxtChunkRepo] 强转 kb_id 为整型失败，当前原始值为: {row_lowercase_dict['kb_id']}")
                                pass
                                
                        sub_batch_dict[str(real_chunk_id)] = row_lowercase_dict
                return sub_batch_dict

            # ========================================================
            # 核心优化三：多任务并行驱动
            # ========================================================
            logger.debug(f"[TxtChunkRepo] 准备执行图谱文本回表，总 ID 数: {len(chunk_ids)}，已安全切分为 {len(batches)} 个批次并行下沉查询")
            
            tasks = [fetch_batch(batch) for batch in batches]
            completed_batches = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 合并并发结果集
            for i, batch_res in enumerate(completed_batches):
                if isinstance(batch_res, Exception):
                    logger.error(f"[TxtChunkRepo] 并发查询非结构化块在第 {i} 批次遭遇意外崩溃，错误详情: {batch_res}")
                    continue
                if isinstance(batch_res, dict):
                    # 🔍 诊断日志：记录每批次返回的数据样本键名
                    if batch_res:
                        sample_key = next(iter(batch_res))
                        sample_val = batch_res[sample_key]
                        sample_val_keys = list(sample_val.keys()) if isinstance(sample_val, dict) else type(sample_val).__name__
                        logger.debug(
                            f"[TxtChunkRepo] 批次 {i} 返回 {len(batch_res)} 条, "
                            f"样本 key={sample_key!r}, 样本 value keys/type={sample_val_keys!r}"
                        )
                    chunk_dict.update(batch_res)
            
            # 🔍 诊断日志：汇总所有键名，确认无异常
            if chunk_dict:
                all_keys_set = set()
                for v in chunk_dict.values():
                    if isinstance(v, dict):
                        all_keys_set.update(v.keys())
                logger.debug(f"[TxtChunkRepo] 汇总所有 chunk 的字典键名: {sorted(all_keys_set)!r}")
            
            logger.info(f"[TxtChunkRepo] 完美通过 Oracle 安全过滤 (<= {security_level}) 成功召回并清洗完成 {len(chunk_dict)} 条标准文本块。")
            return chunk_dict
            
        except Exception as e:
            logger.error(f"Oracle get chunks by IDs failed. Error: {str(e)}", exc_info=True)
            raise DatabaseException("Oracle get text chunks by IDs failed", original_error=e)
        
    async def search_by_file_and_page(self, file_id: str, page_no: int) -> list[dict]:
        """按 file_id + page_no 查询单页全部分块（跨分区查询）"""
        try:
            sql = text("""
                SELECT chunk_id, chunk_num, chunk_type, content, header, chunk_metadata
                FROM KBOT_BIZ_TXT_EMBEDDING
                WHERE file_id = :file_id
                  AND chunk_metadata.page_num = :page_no
                ORDER BY chunk_num ASC
            """)
            rows = await self.session.execute(sql, {
                "file_id": file_id, "page_no": str(page_no),
            })
            return [dict(row._mapping) for row in rows.fetchall()]
        except Exception as e:
            raise DatabaseException(f"按页码查询分块失败", original_error=e)