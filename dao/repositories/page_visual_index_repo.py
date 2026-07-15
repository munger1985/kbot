"""页面视觉索引仓库 — nxcube.page_visual_index 的 CRUD 操作。"""

from sqlalchemy import select, delete, text
from loguru import logger
from core.database import db_instance
from core.exceptions import DatabaseException
from dao.entities import PageVisualIndexEntity


class PageVisualIndexRepository:
    """页面视觉索引表 (page_visual_index) 的数据访问层"""

    @property
    def db_session(self):
        return db_instance().get_session()

    async def insert(
        self,
        file_id: str,
        kb_id: str,
        page_no: int,
        image_path: str,
        embedding: list[float] | None,
        caption: str = "",
    ) -> None:
        """插入一条页面视觉索引记录（冲突则忽略）"""
        try:
            async with self.db_session as session:
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
            async with self.db_session as session:
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
        emb_str: str,
        kb_ids: list[int] | None,
        top_k: int,
    ) -> list[dict]:
        """pgvector 余弦相似度搜索（返回 dict 列表，跨表查询时不依赖 entity）"""
        try:
            async with self.db_session as session:
                if kb_ids:
                    kb_list = "{" + ",".join(f'"{k}"' for k in kb_ids) + "}"
                    rows = await session.execute(
                        text("""
                            SELECT file_id, page_no, image_path, COALESCE(caption, ''),
                                   1 - (embedding <=> CAST(:emb AS vector)) AS sim
                            FROM page_visual_index
                            WHERE kb_id = ANY(:kids::uuid[])
                              AND embedding IS NOT NULL
                            ORDER BY embedding <=> CAST(:emb AS vector)
                            LIMIT :limit
                        """),
                        {"emb": emb_str, "kids": kb_list, "limit": top_k},
                    )
                else:
                    rows = await session.execute(
                        text("""
                            SELECT file_id, page_no, image_path, COALESCE(caption, ''),
                                   1 - (embedding <=> CAST(:emb AS vector)) AS sim
                            FROM page_visual_index
                            WHERE embedding IS NOT NULL
                            ORDER BY embedding <=> CAST(:emb AS vector)
                            LIMIT :limit
                        """),
                        {"emb": emb_str, "limit": top_k},
                    )
                return [dict(row._mapping) for row in rows.fetchall()]
        except Exception as e:
            logger.error(f"[PageVisualIdxRepo] search_by_embedding failed: {e}")
            return []

    async def delete_by_file_ids(self, file_ids: list[str]) -> None:
        """按文件 ID 列表删除记录"""
        try:
            async with self.db_session as session:
                for fid in file_ids:
                    await session.execute(
                        delete(PageVisualIndexEntity).where(PageVisualIndexEntity.file_id == fid)
                    )
                await session.commit()
        except Exception as e:
            logger.error(f"[PageVisualIdxRepo] delete_by_file_ids failed: {e}")

    async def delete_by_kb_id(self, kb_id: str) -> None:
        """按知识库 ID 删除记录"""
        try:
            async with self.db_session as session:
                await session.execute(
                    delete(PageVisualIndexEntity).where(PageVisualIndexEntity.kb_id == kb_id)
                )
                await session.commit()
        except Exception as e:
            logger.error(f"[PageVisualIdxRepo] delete_by_kb_id failed: {e}")
