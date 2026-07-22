"""
Oracle 23ai 兼容的文档引用关系仓库 — doc_relation 表的 CRUD 操作。

从 NexusCube PG 版本适配，使用 Oracle 兼容语法：
  - MERGE INTO 替代 PG INSERT ... ON CONFLICT
  - 使用 Oracle JSON 处理方式
"""
import json
from typing import Sequence
from loguru import logger
from sqlalchemy import text, select, delete, or_
from sqlalchemy.ext.asyncio import AsyncSession
from platform_core.exceptions import DatabaseException
from dao.entities import DocRelationEntity
from .base_repo import BaseRepository


class DocRelationRepository(BaseRepository[DocRelationEntity]):
    """文档引用关系表 (doc_relation) 的数据访问层"""

    async def upsert(self, kb_id: int, source_file_id: str,
                     target_file_id: str, relation: dict) -> None:
        """插入或更新文档引用关系，以 (source_file_id, target_file_id) 为匹配键"""
        try:
            existing = await self.session.execute(
                select(DocRelationEntity).where(
                    DocRelationEntity.source_file_id == source_file_id,
                    DocRelationEntity.target_file_id == target_file_id
                )
            )
            entity = existing.scalar_one_or_none()

            data = {
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
            }

            if entity:
                for key, value in data.items():
                    setattr(entity, key, value)
            else:
                self.session.add(DocRelationEntity(**data))

            await self.session.flush()
            logger.debug(
                f"[DocRelationRepo] upsert 成功: "
                f"source={source_file_id}, target={target_file_id}"
            )
        except Exception as e:
            logger.error(f"[DocRelationRepo] upsert 失败: {e}")
            raise DatabaseException("保存文档引用关系失败", original_error=e)

    async def batch_insert(self, relations: list[dict]) -> None:
        """批量插入引用关系，已存在则跳过"""
        if not relations:
            return
        try:
            for r in relations:
                existing = await self.session.execute(
                    select(DocRelationEntity).where(
                        DocRelationEntity.source_file_id == r.get("source_file_id"),
                        DocRelationEntity.target_file_id == r.get("target_file_id")
                    )
                )
                if existing.scalar_one_or_none() is not None:
                    continue
                entity = DocRelationEntity(
                    kb_id=r.get("kb_id"),
                    source_file_id=r.get("source_file_id"),
                    target_file_id=r.get("target_file_id"),
                    target_doc_name=r.get("target_doc_name", ""),
                    target_chapter=r.get("target_chapter", ""),
                    target_section=r.get("target_section", ""),
                    relation_type=r.get("relation_type", "reference"),
                    context_snippet=r.get("context_snippet", ""),
                    confidence=r.get("confidence", 1.0),
                )
                self.session.add(entity)
            await self.session.flush()
            logger.debug(f"[DocRelationRepo] batch_insert 成功: {len(relations)} 条")
        except Exception as e:
            logger.error(f"[DocRelationRepo] batch_insert 失败: {e}")
            raise DatabaseException("批量插入文档引用关系失败", original_error=e)

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
            stmt = delete(DocRelationEntity).where(
                or_(
                    DocRelationEntity.source_file_id == file_id,
                    DocRelationEntity.target_file_id == file_id
                )
            )
            await self.session.execute(stmt)
            await self.session.flush()
            logger.debug(f"[DocRelationRepo] 删除文件相关引用成功: file_id={file_id}")
        except Exception as e:
            logger.error(f"[DocRelationRepo] delete_by_file 失败: {e}")
            raise DatabaseException("删除文档引用关系失败", original_error=e)

    async def delete_by_kb_id(self, kb_id: int) -> None:
        """删除指定知识库的所有引用关系"""
        try:
            stmt = delete(DocRelationEntity).where(
                DocRelationEntity.kb_id == kb_id
            )
            await self.session.execute(stmt)
            await self.session.flush()
            logger.debug(f"[DocRelationRepo] 删除知识库引用成功: kb_id={kb_id}")
        except Exception as e:
            logger.error(f"[DocRelationRepo] delete_by_kb_id 失败: {e}")
            raise DatabaseException("删除知识库文档引用关系失败", original_error=e)
