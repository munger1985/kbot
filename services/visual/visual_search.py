"""视觉检索服务 — 文本查询 + 以图搜图。

通过 Visual 微服务 API 获取 embedding，pgvector 余弦搜索。
"""

from dataclasses import dataclass, field
from loguru import logger
from utils.clients import AIModelClient
from dao.repositories import PageVisualIndexRepository, ExtractedImageRepository


@dataclass
class VisualSearchResult:
    """视觉检索结果"""
    file_id: str = ""
    kb_id: str = ""
    page_no: int = 0
    image_path: str = ""
    description: str = ""
    image_type: str = "page"
    similarity: float = 0.0
    chunk_id: str = ""


class VisualSearch:
    """视觉检索：文本/图片 → embedding → pgvector 余弦搜索。

    用法:
        search = VisualSearch()
        results = await search.search_by_text("设备可靠性框图", kb_ids=["..."])
        results = await search.search_by_image(base64_image, kb_ids=["..."])
    """

    def __init__(self, model_client: AIModelClient | None = None):
        self.model_client = model_client or AIModelClient()

    # ── 公开 API ──────────────────────────────────────────

    async def search_by_text(
        self,
        query: str,
        kb_ids: list[str],
        top_k: int = 5,
        search_pages: bool = True,
        search_images: bool = True,
    ) -> list[VisualSearchResult]:
        """文本查询 → 视觉检索"""
        emb = await self._get_text_embedding(query)
        if not emb:
            return []
        return await self._search_by_embedding(
            emb, kb_ids, top_k, search_pages, search_images
        )

    async def search_by_image(
        self,
        image_base64: str,
        kb_ids: list[str],
        top_k: int = 5,
    ) -> list[VisualSearchResult]:
        """以图搜图：图片 → 视觉 embedding → pgvector 搜索"""
        try:
            emb = await self.model_client.get_visual_embedding(image_base64)
        except Exception as e:
            logger.error(f"[VisualSearch] visual embedding failed: {e}")
            return []
        return await self._search_by_embedding(emb, kb_ids, top_k, True, True)

    # ── 内部 ──────────────────────────────────────────────

    async def _search_by_embedding(
        self,
        emb: list[float],
        kb_ids: list[str],
        top_k: int,
        search_pages: bool,
        search_images: bool,
    ) -> list[VisualSearchResult]:
        results: list[VisualSearchResult] = []
        emb_str = "[" + ",".join(str(x) for x in emb) + "]"

        if search_pages:
            page_repo = PageVisualIndexRepository()
            for row in await page_repo.search_by_embedding(emb_str, kb_ids, top_k):
                results.append(VisualSearchResult(
                    file_id=str(row.get("file_id", "")),
                    kb_id=str(row.get("kb_id", "")),
                    page_no=int(row.get("page_no", 0)),
                    image_path=str(row.get("image_path", "")),
                    similarity=float(row.get("sim", 0)),
                    description=str(row.get("caption", "")),
                    image_type="page",
                    chunk_id="",
                ))

        if search_images:
            img_repo = ExtractedImageRepository()
            for row in await img_repo.search_by_embedding(emb_str, kb_ids, top_k):
                results.append(VisualSearchResult(
                    file_id=str(row.get("file_id", "")),
                    kb_id=str(row.get("kb_id", "")),
                    page_no=int(row.get("page_no", 0)),
                    image_path=str(row.get("image_path", "")),
                    similarity=float(row.get("similarity", 0)),
                    description=str(row.get("description", "")),
                    image_type=str(row.get("image_type", "")),
                    chunk_id=str(row.get("chunk_id", "")),
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    async def _get_text_embedding(self, text: str) -> list[float] | None:
        """文本 → embedding（复用文本嵌入服务）"""
        try:
            from core.config.settings import get_embed_config
            config = get_embed_config()
            model_name = getattr(config, 'model_name', 'bge-m3')
            return await self.model_client.get_embedding(model_name, text)
        except Exception as e:
            logger.error(f"[VisualSearch] text embedding failed: {e}")
            return None
