"""统一视觉索引仓库 — extracted_images 的 Oracle 23ai CRUD + 搜索操作。

已合并 page_visual_index 的全部功能：
  - 页面截图用 image_type='page' 标识，caption → description
  - get_page_image() 从 PageVisualIndexRepository 迁移
  - search_by_embedding() 支持可选 image_types 过滤

严格遵循其他 Repo 风格：构造时传入 session，使用 self.session 操作。
"""

import json
import array as array_module
from loguru import logger
from sqlalchemy import text, delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from core.exceptions import DatabaseException
from dao.entities import ExtractedImageEntity
from .base_repo import BaseRepository


class ExtractedImageRepository(BaseRepository[ExtractedImageEntity]):
    """统一视觉索引表 (extracted_images) 的数据访问层。"""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ── 写入 ──────────────────────────────────────────────

    async def insert(
        self,
        file_id: str,
        kb_id: str,
        page_no: int,
        image_path: str,
        embedding: list[float] | None,
        description: str = "",
        image_type: str = "figure",
        bbox: dict | None = None,
        chunk_id: str | None = None,
    ) -> None:
        """插入一条视觉索引记录。"""
        try:
            entity = ExtractedImageEntity(
                file_id=file_id, kb_id=kb_id, page_no=page_no,
                image_path=image_path, embedding=embedding,
                description=description, image_type=image_type,
                bbox=bbox or {}, chunk_id=chunk_id,
            )
            self.session.add(entity)
            await self.session.flush()
        except Exception as e:
            logger.error(f"[ExtractedImgRepo] insert failed: {e}")
            raise DatabaseException("保存视觉索引记录失败", original_error=e)

    # ── 点查 ──────────────────────────────────────────────

    async def get_page_image(self, file_id: str, page_no: int) -> str | None:
        """按 file_id + page_no 查询页面截图路径。"""
        try:
            stmt = (
                select(ExtractedImageEntity.image_path)
                .where(ExtractedImageEntity.file_id == file_id)
                .where(ExtractedImageEntity.page_no == page_no)
                .where(ExtractedImageEntity.image_type == "page")
            )
            result = await self.session.execute(stmt)
            row = result.first()
            return str(row[0]) if row else None
        except Exception as e:
            logger.warning(f"[ExtractedImgRepo] get_page_image: {e}")
            return None

    # ── 向量搜索 ──────────────────────────────────────────

    async def search_by_embedding(
        self,
        emb: list[float],
        kb_ids: list[int] | None,
        top_k: int,
        image_types: list[str] | None = None,
    ) -> list[dict]:
        """Oracle VECTOR 余弦相似度搜索。"""
        if not emb:
            return []
        try:
            conditions = ["embedding IS NOT NULL"]
            params: dict = {
                "qv": array_module.array("f", emb),
                "limit": top_k,
            }

            if kb_ids:
                kb_conditions = []
                for i, kid in enumerate(kb_ids):
                    key = f"kb_{i}"
                    kb_conditions.append(f":{key}")
                    params[key] = kid
                conditions.append(f"kb_id IN ({', '.join(kb_conditions)})")

            if image_types:
                img_conditions = []
                for i, it in enumerate(image_types):
                    key = f"imgt_{i}"
                    img_conditions.append(f":{key}")
                    params[key] = it
                conditions.append(f"image_type IN ({', '.join(img_conditions)})")

            where_clause = " AND ".join(conditions)

            rows = await self.session.execute(
                text(f"""
                    SELECT file_id, kb_id, page_no, image_path,
                           COALESCE(description, '') AS description,
                           COALESCE(image_type, 'page') AS image_type,
                           COALESCE(chunk_id, '') AS chunk_id,
                           (1 - VECTOR_DISTANCE(embedding, :qv, COSINE)) AS similarity
                    FROM extracted_images
                    WHERE {where_clause}
                    ORDER BY similarity DESC
                    FETCH FIRST :limit ROWS ONLY
                """),
                params,
            )
            return [dict(row._mapping) for row in rows.fetchall()]
        except Exception as e:
            logger.error(f"[ExtractedImgRepo] search_by_embedding failed: {e}")
            return []

    # ── 删除 ──────────────────────────────────────────────

    async def delete_by_file_ids(self, file_ids: list[str]) -> None:
        """按文件 ID 列表删除记录"""
        if not file_ids:
            return
        try:
            await self.session.execute(
                delete(ExtractedImageEntity).where(
                    ExtractedImageEntity.file_id.in_(file_ids)
                )
            )
            await self.session.flush()
        except Exception as e:
            logger.error(f"[ExtractedImgRepo] delete_by_file_ids failed: {e}")

    async def delete_by_kb_id(self, kb_id: str) -> None:
        """按知识库 ID 删除记录"""
        try:
            await self.session.execute(
                delete(ExtractedImageEntity).where(ExtractedImageEntity.kb_id == kb_id)
            )
            await self.session.flush()
        except Exception as e:
            logger.error(f"[ExtractedImgRepo] delete_by_kb_id failed: {e}")
