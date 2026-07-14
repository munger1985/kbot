"""页面视觉索引仓库 — page_visual_index 的 Oracle 23ai CRUD 操作。

从 NexusCube PG 版本适配，使用 Oracle 兼容语法：
  - VECTOR_DISTANCE() 替代 PG 的 <=> 算子
  - FETCH FIRST n ROWS ONLY 替代 LIMIT n
"""

import array as array_module
from loguru import logger
from sqlalchemy import text, select, delete
from core.database.oracle import get_session
from core.exceptions import DatabaseException
from dao.entities import PageVisualIndexEntity


class PageVisualIndexRepository:
    """页面视觉索引表 (page_visual_index) 的数据访问层"""

    async def insert(
        self,
        file_id: str,
        kb_id: str,
        page_no: int,
        image_path: str,
        embedding: list[float] | None,
        caption: str = "",
    ) -> None:
        """插入一条页面视觉索引记录"""
        try:
            async with get_session() as session:
                entity = PageVisualIndexEntity(
                    file_id=file_id,
                    kb_id=kb_id,
                    page_no=page_no,
                    image_path=image_path,
                    embedding=embedding,
                    caption=caption,
                )
                session.add(entity)
                await session.flush()
                await session.commit()
        except Exception as e:
            logger.error(f"[PageVisualIdxRepo] insert failed: {e}")

    async def get_page_image(self, file_id: str, page_no: int) -> str | None:
        """按 file_id + page_no 查询页面截图路径"""
        try:
            async with get_session() as session:
                stmt = (
                    select(PageVisualIndexEntity.image_path)
                    .where(PageVisualIndexEntity.file_id == file_id)
                    .where(PageVisualIndexEntity.page_no == page_no)
                    .limit(1)
                )
                row = (await session.execute(stmt)).first()
                return str(row[0]) if row else None
        except Exception as e:
            logger.warning(f"[PageVisualIdxRepo] get_page_image: {e}")
            return None

    async def search_by_embedding(
        self,
        emb: list[float],
        kb_ids: list[str] | None,
        top_k: int,
    ) -> list[dict]:
        """Oracle VECTOR 余弦相似度搜索"""
        try:
            async with get_session() as session:
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

                where_clause = " AND ".join(conditions)

                rows = await session.execute(
                    text(f"""
                        SELECT file_id, page_no, image_path,
                               COALESCE(caption, '') AS caption,
                               (1 - VECTOR_DISTANCE(embedding, :qv, COSINE)) AS sim
                        FROM page_visual_index
                        WHERE {where_clause}
                        ORDER BY sim DESC
                        FETCH FIRST :limit ROWS ONLY
                    """),
                    params,
                )
                return [dict(row._mapping) for row in rows.fetchall()]
        except Exception as e:
            logger.error(f"[PageVisualIdxRepo] search_by_embedding failed: {e}")
            return []

    async def delete_by_file_ids(self, file_ids: list[str]) -> None:
        """按文件 ID 列表删除记录"""
        if not file_ids:
            return
        try:
            async with get_session() as session:
                await session.execute(
                    delete(PageVisualIndexEntity).where(
                        PageVisualIndexEntity.file_id.in_(file_ids)
                    )
                )
                await session.commit()
        except Exception as e:
            logger.error(f"[PageVisualIdxRepo] delete_by_file_ids failed: {e}")

    async def delete_by_kb_id(self, kb_id: str) -> None:
        """按知识库 ID 删除记录"""
        try:
            async with get_session() as session:
                await session.execute(
                    delete(PageVisualIndexEntity).where(PageVisualIndexEntity.kb_id == kb_id)
                )
                await session.commit()
        except Exception as e:
            logger.error(f"[PageVisualIdxRepo] delete_by_kb_id failed: {e}")
