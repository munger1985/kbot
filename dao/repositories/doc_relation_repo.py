"""
Oracle 23ai 兼容的文档引用关系仓库 — doc_relation 表的 CRUD 操作。

从 NexusCube PG 版本适配，使用 Oracle 兼容语法：
  - MERGE INTO 替代 PG INSERT ... ON CONFLICT
  - 使用 Oracle JSON 处理方式
"""
import json
from typing import Sequence
from loguru import logger
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession
from core.exceptions import DatabaseException
from dao.entities import DocRelationEntity
from .base_repo import BaseRepository


class DocRelationRepository(BaseRepository[DocRelationEntity]):
    """文档引用关系表 (doc_relation) 的数据访问层"""

    async def upsert(self, kb_id: int, source_file_id: str,
                     target_file_id: str, relation: dict) -> None:
        """插入或更新文档引用关系

        使用 Oracle MERGE INTO 实现 upsert 语义，
        以 (source_file_id, target_file_id) 为匹配键。
        """
        stmt = text("""
            MERGE INTO doc_relation t
            USING (SELECT :source_file_id AS source_file_id,
                          :target_file_id AS target_file_id FROM DUAL) s
            ON (t.source_file_id = s.source_file_id
                AND t.target_file_id = s.target_file_id)
            WHEN MATCHED THEN
                UPDATE SET
                    kb_id = :kb_id,
                    target_doc_name = :target_doc_name,
                    target_chapter = :target_chapter,
                    target_section = :target_section,
                    relation_type = :relation_type,
                    context_snippet = :context_snippet,
                    confidence = :confidence,
                    biz_metadata = :biz_metadata
            WHEN NOT MATCHED THEN
                INSERT (kb_id, source_file_id, target_file_id,
                        target_doc_name, target_chapter, target_section,
                        relation_type, context_snippet, confidence, biz_metadata)
                VALUES (:kb_id, :source_file_id, :target_file_id,
                        :target_doc_name, :target_chapter, :target_section,
                        :relation_type, :context_snippet, :confidence, :biz_metadata)
        """)
        try:
            await self.session.execute(stmt, {
                "kb_id": kb_id,
                "source_file_id": source_file_id,
                "target_file_id": target_file_id,
                "target_doc_name": relation.get("target_doc_name", ""),
                "target_chapter": relation.get("target_chapter", ""),
                "target_section": relation.get("target_section", ""),
                "relation_type": relation.get("relation_type", "reference"),
                "context_snippet": relation.get("context_snippet", ""),
                "confidence": relation.get("confidence", 1.0),
                "biz_metadata": json.dumps(relation.get("biz_metadata", {}), ensure_ascii=False),
            })
            await self.session.flush()
            logger.debug(
                f"[DocRelationRepo] upsert 成功: "
                f"source={source_file_id}, target={target_file_id}"
            )
        except Exception as e:
            logger.error(f"[DocRelationRepo] upsert 失败: {e}")
            raise DatabaseException("保存文档引用关系失败", original_error=e)

    async def get_by_source_file(self, source_file_id: str) -> list[dict]:
        """查询某文档引用了哪些文档"""
        try:
            stmt = select(DocRelationEntity).where(
                DocRelationEntity.source_file_id == source_file_id
            ).order_by(DocRelationEntity.created_at.desc())
            result = await self.session.execute(stmt)
            entities = result.scalars().all()
            return [
                {
                    "id": e.id,
                    "kb_id": e.kb_id,
                    "source_file_id": e.source_file_id,
                    "target_file_id": e.target_file_id,
                    "target_doc_name": e.target_doc_name,
                    "target_chapter": e.target_chapter,
                    "target_section": e.target_section,
                    "relation_type": e.relation_type,
                    "context_snippet": e.context_snippet,
                    "confidence": e.confidence,
                    "biz_metadata": e.biz_metadata,
                    "created_at": e.created_at,
                }
                for e in entities
            ]
        except Exception as e:
            logger.error(f"[DocRelationRepo] get_by_source_file 失败: {e}")
            raise DatabaseException("查询文档引用关系失败", original_error=e)

    async def delete_by_file(self, file_id: str) -> None:
        """删除指定文件相关的所有引用关系（source 或 target）"""
        try:
            stmt = text("""
                DELETE FROM doc_relation
                WHERE source_file_id = :fid OR target_file_id = :fid
            """)
            await self.session.execute(stmt, {"fid": file_id})
            await self.session.flush()
            logger.debug(f"[DocRelationRepo] 删除文件相关引用成功: file_id={file_id}")
        except Exception as e:
            logger.error(f"[DocRelationRepo] delete_by_file 失败: {e}")
            raise DatabaseException("删除文档引用关系失败", original_error=e)
