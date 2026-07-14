"""提取图片仓库 — extracted_images 的 Oracle 23ai CRUD 操作。

从 NexusCube PG 版本适配，使用 Oracle 兼容语法：
  - VECTOR_DISTANCE() 替代 PG 的 <=> 算子
  - FETCH FIRST n ROWS ONLY 替代 LIMIT n
  - json.dumps() 处理 JSON 字段绑定
"""

import json
import array as array_module
from loguru import logger
from sqlalchemy import text, delete
from core.database.oracle import get_session
from core.exceptions import DatabaseException
from dao.entities import ExtractedImageEntity


class ExtractedImageRepository:
    """提取图片表 (extracted_images) 的数据访问层"""

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
        """插入一条提取图片记录"""
        try:
            async with get_session() as session:
                entity = ExtractedImageEntity(
                    file_id=file_id,
                    kb_id=kb_id,
                    page_no=page_no,
                    image_path=image_path,
                    embedding=embedding,
                    description=description,
                    image_type=image_type,
                    bbox=bbox or {},
                    chunk_id=chunk_id,
                )
                session.add(entity)
                await session.flush()
                await session.commit()
        except Exception as e:
            logger.error(f"[ExtractedImgRepo] insert failed: {e}")
            raise DatabaseException("保存提取图片记录失败", original_error=e)

    async def search_by_embedding(
        self,
        emb: list[float],
        kb_ids: list[str],
        top_k: int,
    ) -> list[dict]:
        """Oracle VECTOR 余弦相似度搜索"""
        if not emb:
            return []
        try:
            async with get_session() as session:
                # 构建 kb_id IN 子句
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
                        SELECT file_id, kb_id, page_no, image_path,
                               COALESCE(description, '') AS description,
                               COALESCE(image_type, '') AS image_type,
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

    async def delete_by_file_ids(self, file_ids: list[str]) -> None:
        """按文件 ID 列表删除记录"""
        if not file_ids:
            return
        try:
            async with get_session() as session:
                await session.execute(
                    delete(ExtractedImageEntity).where(
                        ExtractedImageEntity.file_id.in_(file_ids)
                    )
                )
                await session.commit()
        except Exception as e:
            logger.error(f"[ExtractedImgRepo] delete_by_file_ids failed: {e}")

    async def delete_by_kb_id(self, kb_id: str) -> None:
        """按知识库 ID 删除记录"""
        try:
            async with get_session() as session:
                await session.execute(
                    delete(ExtractedImageEntity).where(ExtractedImageEntity.kb_id == kb_id)
                )
                await session.commit()
        except Exception as e:
            logger.error(f"[ExtractedImgRepo] delete_by_kb_id failed: {e}")
