"""文档元数据仓库 — doc_metadata 表的 CRUD 操作（ORM 风格）。

严格遵循 Repo 规范：使用 self.session 直接操作，commit/rollback 由调用方管理。
"""

from datetime import date as date_type, datetime
from sqlalchemy import select, delete, func
from loguru import logger
from platform_core.exceptions import DatabaseException
from dao.entities import DocMetadataEntity
from .base_repo import BaseRepository


def _parse_date(raw: str) -> date_type | None:
    """将 LLM 返回的各种日期字符串转为 date 对象，无法解析则返回 None。"""
    if not raw or not isinstance(raw, str):
        return None
    formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
        "%Y-%m", "%Y/%m",
        "%Y年%m月%d日", "%Y年%m月", "%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


class DocMetaRepository(BaseRepository[DocMetadataEntity]):
    """文档元数据表 (doc_metadata) 的数据访问层"""

    async def upsert(self, kb_id: int, file_id: str, meta: dict) -> None:
        """插入或更新文档元数据，以 file_id 为匹配键"""
        try:
            existing = await self.session.execute(
                select(DocMetadataEntity).where(
                    DocMetadataEntity.file_id == file_id
                )
            )
            entity = existing.scalar_one_or_none()

            doc_date_str = meta.get("doc_date")
            doc_date = _parse_date(doc_date_str) if doc_date_str else None

            if entity:
                entity.kb_id = kb_id
                entity.doc_name = meta.get("doc_name", "")
                entity.doc_type = meta.get("doc_type", "other")
                entity.doc_number = meta.get("doc_number", "")
                entity.doc_version = meta.get("doc_version", "")
                entity.doc_date = doc_date
                entity.page_count = meta.get("page_count") or 0
                entity.chunk_count = meta.get("chunk_count") or 0
                entity.doc_abstract = meta.get("doc_abstract", "")
                entity.doc_keywords = meta.get("doc_keywords", [])
                entity.doc_references = meta.get("doc_references", [])
                entity.biz_metadata = meta.get("biz_metadata", {})
            else:
                self.session.add(DocMetadataEntity(
                    kb_id=kb_id,
                    file_id=file_id,
                    doc_name=meta.get("doc_name", ""),
                    doc_type=meta.get("doc_type", "other"),
                    doc_number=meta.get("doc_number", ""),
                    doc_version=meta.get("doc_version", ""),
                    doc_date=doc_date,
                    page_count=meta.get("page_count") or 0,
                    chunk_count=meta.get("chunk_count") or 0,
                    doc_abstract=meta.get("doc_abstract", ""),
                    doc_keywords=meta.get("doc_keywords", []),
                    doc_references=meta.get("doc_references", []),
                    biz_metadata=meta.get("biz_metadata", {}),
                ))

            await self.session.flush()
            logger.debug(f"[DocMetaRepo] upsert 成功: file_id={file_id}")
        except Exception as e:
            logger.error(f"[DocMetaRepo] upsert 失败: {e}")
            raise DatabaseException("保存文档元数据失败", original_error=e)

    async def search_by_name(self, kb_ids: list[str],
                              name_keyword: str) -> list[dict]:
        """按文档名模糊搜索（ILIKE + pg_trgm）"""
        if not name_keyword:
            return []
        try:
            stmt = select(
                DocMetadataEntity.file_id,
                DocMetadataEntity.doc_name,
                DocMetadataEntity.doc_number,
                DocMetadataEntity.doc_type,
            ).where(
                DocMetadataEntity.kb_id.in_(kb_ids),
                DocMetadataEntity.doc_name.ilike(f"%{name_keyword}%"),
            ).order_by(
                func.similarity(DocMetadataEntity.doc_name, name_keyword).desc()
            ).limit(10)
            result = await self.session.execute(stmt)
            return [dict(row._mapping) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"[DocMetaRepo] search_by_name 失败: {e}")
            raise DatabaseException("搜索文档元数据失败", original_error=e)

    async def delete_by_file_ids(self, file_ids: list[str]) -> None:
        """按文件 ID 删除元数据"""
        if not file_ids:
            return
        try:
            stmt = delete(DocMetadataEntity).where(
                DocMetadataEntity.file_id.in_(file_ids)
            )
            await self.session.execute(stmt)
            await self.session.flush()
        except Exception as e:
            logger.error(f"[DocMetaRepo] delete_by_file_ids 失败: {e}")
            raise DatabaseException("删除文档元数据失败", original_error=e)

    async def delete_by_kb_id(self, kb_id: str) -> None:
        """按知识库 ID 删除元数据"""
        try:
            stmt = delete(DocMetadataEntity).where(
                DocMetadataEntity.kb_id == kb_id
            )
            await self.session.execute(stmt)
            await self.session.flush()
        except Exception as e:
            logger.error(f"[DocMetaRepo] delete_by_kb_id 失败: {e}")
            raise DatabaseException("删除知识库文档元数据失败", original_error=e)
