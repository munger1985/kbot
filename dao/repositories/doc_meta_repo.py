"""
Oracle 23ai 兼容的文档元数据仓库 — doc_metadata 表的 CRUD 操作。

从 NexusCube PG 版本适配，使用 Oracle 兼容语法：
  - MERGE INTO 替代 PG INSERT ... ON CONFLICT
  - LIKE 替代 ILIKE（Oracle 默认大小写不敏感取决于 NLS 设置，使用 UPPER 保证）
  - IN (...) 替代 = ANY(:array)
  - Python date 对象直接传入日期字段
  - OracleJSON type handler 处理 JSON 序列化
"""
import json
from datetime import date as date_type, datetime
from typing import Sequence
from loguru import logger
from sqlalchemy import text, select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from core.exceptions import DatabaseException
from dao.entities import DocMetadataEntity
from .base_repo import BaseRepository


def _parse_date(raw: str) -> date_type | None:
    """将 LLM 返回的各种日期字符串转为 date 对象，无法解析则返回 None。"""
    if not raw or not isinstance(raw, str):
        return None
    formats = [
        "%Y-%m-%d",     # 2022-06-15
        "%Y/%m/%d",     # 2022/06/15
        "%Y.%m.%d",     # 2022.06.15
        "%Y-%m",        # 2022-06
        "%Y/%m",        # 2022/06
        "%Y年%m月%d日",  # 2022年06月15日
        "%Y年%m月",     # 2022年06月
        "%Y",           # 2022
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

            data = {
                "kb_id": kb_id,
                "file_id": file_id,
                "doc_name": meta.get("doc_name", ""),
                "doc_type": meta.get("doc_type", "other"),
                "doc_number": meta.get("doc_number", ""),
                "doc_version": meta.get("doc_version", ""),
                "doc_date": doc_date,
                "page_count": meta.get("page_count") or 0,
                "chunk_count": meta.get("chunk_count") or 0,
                "doc_abstract": meta.get("doc_abstract", ""),
                "doc_keywords": json.dumps(meta.get("doc_keywords", []), ensure_ascii=False),
                "doc_references": json.dumps(meta.get("doc_references", []), ensure_ascii=False),
                "biz_metadata": json.dumps(meta.get("biz_metadata", {}), ensure_ascii=False),
            }

            if entity:
                for key, value in data.items():
                    setattr(entity, key, value)
            else:
                self.session.add(DocMetadataEntity(**data))

            await self.session.flush()
            logger.debug(f"[DocMetaRepo] upsert 成功: file_id={file_id}")
        except Exception as e:
            logger.error(f"[DocMetaRepo] upsert 失败: {e}")
            raise DatabaseException("保存文档元数据失败", original_error=e)

    async def search_by_name(self, kb_ids: list[int],
                              name_keyword: str) -> list[dict]:
        """按文档名模糊搜索

        Oracle LIKE 默认大小写敏感，使用 UPPER() 保证不区分大小写。
        """
        if not name_keyword or not kb_ids:
            return []

        try:
            # 构建动态 IN 参数列表
            placeholders = ", ".join(f":kid_{i}" for i in range(len(kb_ids)))
            params = {f"kid_{i}": kid for i, kid in enumerate(kb_ids)}
            pattern = f"%{name_keyword}%"
            params["pattern"] = pattern
            params["name"] = name_keyword

            sql = text(f"""
                SELECT file_id, doc_name, doc_number, doc_type
                FROM kbot_doc_metadata
                WHERE kb_id IN ({placeholders})
                  AND (UPPER(doc_name) LIKE UPPER(:pattern)
                       OR UPPER(doc_number) LIKE UPPER(:pattern))
                ORDER BY doc_name
                FETCH FIRST 10 ROWS ONLY
            """)
            result = await self.session.execute(sql, params)
            rows = result.fetchall()
            return [dict(row._mapping) for row in rows]
        except Exception as e:
            logger.error(f"[DocMetaRepo] search_by_name 失败: {e}")
            raise DatabaseException("搜索文档元数据失败", original_error=e)

    async def get_by_file_id(self, file_id: str) -> dict | None:
        """根据 file_id 获取文档元数据"""
        try:
            stmt = select(DocMetadataEntity).where(
                DocMetadataEntity.file_id == file_id
            )
            result = await self.session.execute(stmt)
            entity = result.scalar_one_or_none()
            if entity is None:
                return None
            # 将 ORM 实体转换为 dict
            return {
                "id": entity.id,
                "kb_id": entity.kb_id,
                "file_id": entity.file_id,
                "doc_name": entity.doc_name,
                "doc_type": entity.doc_type,
                "doc_number": entity.doc_number,
                "doc_version": entity.doc_version,
                "doc_date": entity.doc_date,
                "page_count": entity.page_count,
                "chunk_count": entity.chunk_count,
                "doc_abstract": entity.doc_abstract,
                "doc_keywords": entity.doc_keywords,
                "doc_references": entity.doc_references,
                "biz_metadata": entity.biz_metadata,
                "created_at": entity.created_at,
                "updated_at": entity.updated_at,
            }
        except Exception as e:
            logger.error(f"[DocMetaRepo] get_by_file_id 失败: {e}")
            raise DatabaseException("查询文档元数据失败", original_error=e)

    async def delete_by_file_id(self, file_id: str) -> None:
        """根据 file_id 删除文档元数据"""
        try:
            stmt = delete(DocMetadataEntity).where(
                DocMetadataEntity.file_id == file_id
            )
            await self.session.execute(stmt)
            await self.session.flush()
            logger.debug(f"[DocMetaRepo] 删除成功: file_id={file_id}")
        except Exception as e:
            logger.error(f"[DocMetaRepo] delete_by_file_id 失败: {e}")
            raise DatabaseException("删除文档元数据失败", original_error=e)

    async def delete_by_kb_id(self, kb_id: int) -> None:
        """根据 kb_id 删除整个知识库的文档元数据"""
        try:
            stmt = delete(DocMetadataEntity).where(
                DocMetadataEntity.kb_id == kb_id
            )
            await self.session.execute(stmt)
            await self.session.flush()
            logger.debug(f"[DocMetaRepo] 删除知识库元数据成功: kb_id={kb_id}")
        except Exception as e:
            logger.error(f"[DocMetaRepo] delete_by_kb_id 失败: {e}")
            raise DatabaseException("删除知识库文档元数据失败", original_error=e)

    async def get_by_kb_id(self, kb_id: int) -> list[dict]:
        """根据 kb_id 获取所有文档元数据"""
        try:
            stmt = select(DocMetadataEntity).where(
                DocMetadataEntity.kb_id == kb_id
            ).order_by(DocMetadataEntity.created_at.desc())
            result = await self.session.execute(stmt)
            entities = result.scalars().all()
            return [
                {
                    "id": e.id,
                    "kb_id": e.kb_id,
                    "file_id": e.file_id,
                    "doc_name": e.doc_name,
                    "doc_type": e.doc_type,
                    "doc_number": e.doc_number,
                    "doc_version": e.doc_version,
                    "doc_date": e.doc_date,
                    "page_count": e.page_count,
                    "chunk_count": e.chunk_count,
                    "doc_abstract": e.doc_abstract,
                    "doc_keywords": e.doc_keywords,
                    "doc_references": e.doc_references,
                    "biz_metadata": e.biz_metadata,
                    "created_at": e.created_at,
                    "updated_at": e.updated_at,
                }
                for e in entities
            ]
        except Exception as e:
            logger.error(f"[DocMetaRepo] get_by_kb_id 失败: {e}")
            raise DatabaseException("查询知识库文档元数据失败", original_error=e)
