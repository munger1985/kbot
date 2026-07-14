"""视觉索引服务 — 通过 Visual 微服务生成 embedding 并存入 pgvector。"""

import asyncio
from pathlib import Path
from PIL import Image

from loguru import logger
from utils.clients import AIModelClient
from dao.repositories import PageVisualIndexRepository, ExtractedImageRepository


class VisualIndexer:
    """视觉 embedding 生成与入库（通过 Visual 微服务 API）。

    用法:
        indexer = VisualIndexer()
        await indexer.index_page(file_id, kb_id, page_no, image_path, caption)
    """

    def __init__(self, model_client: AIModelClient | None = None):
        self.model_client = model_client or AIModelClient()

    # ── 公开 API ──────────────────────────────────────────

    async def get_embedding(self, image_path: str, model_name: str = "") -> list[float]:
        """本地图片文件 → 视觉 embedding"""
        import base64
        from io import BytesIO

        img = Image.open(image_path).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return await self.model_client.get_visual_embedding(img_b64, model_name=model_name)

    async def index_page(
        self,
        file_id: str,
        kb_id: str,
        page_no: int,
        image_path: str,
        caption: str = "",
        visual_model: str = "",
    ) -> None:
        """索引单页：生成 embedding 并写入 page_visual_index。

        visual_model 为空时仅保存截图路径和描述，不生成 embedding。
        """
        try:
            emb = await self.get_embedding(image_path, model_name=visual_model) if visual_model else None
            repo = PageVisualIndexRepository()
            await repo.insert(
                file_id=file_id, kb_id=kb_id, page_no=page_no,
                image_path=image_path, embedding=emb, caption=caption,
            )
            logger.debug(f"[VisualIndexer] page {page_no} indexed (embed={len(emb) if emb else 0}d)")
        except Exception as e:
            logger.error(f"[VisualIndexer] page {page_no} index failed: {e}")

    async def index_extracted_image(
        self,
        file_id: str,
        kb_id: str,
        page_no: int,
        image_path: str,
        description: str = "",
        image_type: str = "figure",
        bbox: dict | None = None,
        visual_model: str = "",
    ) -> None:
        """索引提取的图片：生成 embedding 并写入 extracted_images。

        visual_model 为空时仅保存图片路径和描述，不生成 embedding。
        """
        try:
            emb = await self.get_embedding(image_path, model_name=visual_model) if visual_model else None
            repo = ExtractedImageRepository()
            await repo.insert(
                file_id=file_id, kb_id=kb_id, page_no=page_no,
                image_path=image_path, embedding=emb,
                description=description, image_type=image_type,
                bbox=bbox or {},
            )
            logger.debug(f"[VisualIndexer] image indexed: {description[:40] if description else '(no desc)'}")
        except Exception as e:
            logger.error(f"[VisualIndexer] image index failed: {e}")
