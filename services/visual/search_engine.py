"""视觉-文本双向互检索引擎。

主路径: 图片 → pgvector 视觉搜 → 按 file_id+page_no 反查文本
补齐路径: 纯文本 query → ParadeDB 搜文本 → 按 file_id+page_no 反查页面图
输出: 图文配对列表
"""

from dataclasses import dataclass, field
from loguru import logger
from dao.repositories import ExtractedImageRepository
from core.database import db_instance


@dataclass
class VisualTextPair:
    """图文配对结果"""
    file_id: str = ""
    file_name: str = ""
    page_no: int = 0
    page_image_path: str = ""
    image_description: str = ""
    extracted_image_paths: list[str] = field(default_factory=list)
    text_snippets: list[str] = field(default_factory=list)
    similarity: float = 0.0
    source: str = "visual"


class VisualSearchEngine:
    """双向互检索"""

    async def search(
        self,
        query: str = "",
        image_base64: str = "",
        kb_ids: list[int] | None = None,
        top_k: int = 5,
        visual_model: str = "",
    ) -> list[VisualTextPair]:
        """主入口"""

        if not image_base64 and not query:
            return []

        # ── 视觉搜索（主路径）──
        pairs: dict[str, VisualTextPair] = {}
        if image_base64:
            visual_rows = await self._visual_search(image_base64, kb_ids, top_k, visual_model)
            logger.info(f"[VSE] 视觉搜索原始返回 {len(visual_rows)} 条结果")
            for vr in visual_rows:
                pairs[self._key(vr)] = vr
                logger.info(f"[VSE]   结果: file={vr.file_id} page={vr.page_no} sim={vr.similarity:.4f} desc_len={len(vr.image_description)} text_snippets={len(vr.text_snippets)}")

        # ── 文本搜索（补齐路径）──
        if query:
            text_rows = await self._text_search(query, kb_ids, top_k)
            logger.info(f"[VSE] 文本搜索返回 {len(text_rows)} 条结果")
            for tr in text_rows:
                k = self._key(tr)
                if k in pairs:
                    pairs[k].text_snippets.extend(tr.text_snippets)
                    pairs[k].source = "both"
                    logger.info(f"[VSE]   文本补齐: file={tr.file_id} page={tr.page_no} snippets={len(tr.text_snippets)}")
                else:
                    pairs[k] = tr

        # ── 补全缺失 ──
        for k, p in pairs.items():
            if p.page_image_path and not p.text_snippets:
                p.text_snippets = await self._get_texts(p.file_id, p.page_no)
                logger.info(f"[VSE]   文本回填: file={p.file_id} page={p.page_no} got {len(p.text_snippets)} snippets")
            if p.text_snippets and not p.page_image_path:
                p.page_image_path = await self._get_page_image(p.file_id, p.page_no) or ""
                logger.info(f"[VSE]   图片回填: file={p.file_id} page={p.page_no} path={p.page_image_path[:80] if p.page_image_path else 'none'}")

        return sorted(pairs.values(), key=lambda x: x.similarity, reverse=True)[:top_k]

    # ── 内部 ──────────────────────────────────────────────

    def _key(self, p: VisualTextPair) -> str:
        return f"{p.file_id}:{p.page_no}"

    async def _visual_search(
        self, image_base64: str, kb_ids: list[int] | None, top_k: int, visual_model: str = ""
    ) -> list[VisualTextPair]:
        from utils.clients import AIModelClient
        try:
            emb = await AIModelClient().get_visual_embedding(image_base64, model_name=visual_model)
        except Exception as e:
            logger.error(f"[VSE] embed failed: {e}")
            return []

        repo = ExtractedImageRepository()
        rows = await repo.search_by_embedding(emb, kb_ids, top_k, image_types=["page"])

        return [
            VisualTextPair(
                file_id=str(r.get("file_id", "")), page_no=int(r.get("page_no", 0)),
                page_image_path=str(r.get("image_path", "")),
                image_description=str(r.get("description", "")),
                similarity=float(r.get("similarity", 0)), source="visual",
            )
            for r in rows
        ]

    async def _text_search(
        self, query: str, kb_ids: list[int] | None, top_k: int
    ) -> list[VisualTextPair]:
        """ParadeDB 文本搜索 → 按 file_id+page_no 分组"""
        try:
            from services.search.kb_search import TxtBaseSearch
            from utils.clients import AIModelClient

            kb_list = kb_ids or []
            client = AIModelClient()
            emb_cfg = __import__('core.config.settings', fromlist=['get_embed_config']).get_embed_config()

            grouped: dict[str, VisualTextPair] = {}

            for kb_id in kb_list[:5]:
                try:
                    q_emb = await client.get_embedding(
                        getattr(emb_cfg, 'model_name', 'bge-m3'), query
                    ) if hasattr(client, 'get_embedding') else None
                except Exception:
                    q_emb = None

                async with db_instance().get_session() as session:
                    searcher = TxtBaseSearch()
                    chunks = await searcher.search(
                        kb_id=kb_id, keywords=query, search_top_k=top_k,
                        threshold=0.3, weight=0.5, security=3, query_vec=q_emb,
                    )

                for c in chunks:
                    fid = getattr(c, "file_id", "")
                    pn = c.page_num if c.page_num else 1
                    content = getattr(c, "content", "")
                    if not fid or not content:
                        continue
                    key = f"{fid}:{pn}"
                    if key not in grouped:
                        grouped[key] = VisualTextPair(
                            file_id=fid, page_no=pn, source="text",
                            similarity=getattr(c, "score", 0.0) or 0.0,
                        )
                    grouped[key].text_snippets.append(content)

            return list(grouped.values())[:top_k]
        except Exception as e:
            logger.warning(f"[VSE] text search skipped: {e}")
            return []

    async def _get_texts(self, file_id: str, page_no: int) -> list[str]:
        """图 → 文: ParadeDB 按 file_id+page_no 查文本"""
        try:
            from dao.repositories import TxtChunkRepository
            async with db_instance().get_session() as session:
                repo = TxtChunkRepository(session)
                chunks = await repo.search_by_file_and_page(file_id, page_no)
            return [c.get("content", "") for c in chunks if c.get("content")]
        except Exception as e:
            logger.warning(f"[VSE] _get_texts: {e}")
            return []

    async def _get_page_image(self, file_id: str, page_no: int) -> str | None:
        """文 → 图: 查 extracted_images (image_type='page')"""
        repo = ExtractedImageRepository()
        return await repo.get_page_image(file_id, page_no)
