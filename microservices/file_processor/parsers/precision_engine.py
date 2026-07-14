"""精准引擎 — 主流程编排。

简化管线: Docling 渲染 → VLM 逐页 Markdown → MarkdownSectionChunker → ChunkResult。
无 TextMapper、无 bbox_hint、无 JSON 中间格式。
"""
from typing import Any

from loguru import logger
from docling_core.types.doc.document import DoclingDocument

from utils.clients import AIModelClient
from ..parser_schema import DocParserParams, ChunkResult

from .precision_analyzer import PrecisionAnalyzer
from .section_chunker import MarkdownSectionChunker


class PrecisionEngine:
    """精准解析引擎 — VLM 直出 Markdown。

    用法:
        engine = PrecisionEngine(params)
        chunks = await engine.process(doc, file_id)
    """

    def __init__(self, params: DocParserParams, model_client: AIModelClient | None = None):
        self.params = params
        self.model_client = model_client or AIModelClient()
        self.analyzer = PrecisionAnalyzer(self.model_client)
        self.chunker = MarkdownSectionChunker()

    async def process(
        self,
        doc: DoclingDocument,
        file_id: str = "",
        global_summary: str = "",
    ) -> list[ChunkResult]:
        """精准引擎主流程。

        Args:
            doc: Docling 解析后的文档对象
            file_id: 文件唯一标识
            global_summary: 文档全局摘要

        Returns:
            ChunkResult 列表
        """
        vlm_model = self.params.vlm_model
        if not vlm_model:
            logger.warning("[PrecisionEngine] VLM 模型未配置，跳过")
            return []

        # ── Stage 1: VLM 逐页生成 Markdown ──
        logger.info(f"[PrecisionEngine] Stage 1: VLM 逐页 Markdown ({len(doc.pages)} 页)")
        pages = await self.analyzer.analyze_document(
            doc=doc,
            vlm_model=vlm_model,
            file_id=file_id,
        )

        # ── Stage 2: Markdown → Chunks ──
        logger.info(f"[PrecisionEngine] Stage 2: Markdown 切分 → chunks")
        chunks = self.chunker.convert(
            pages=pages,
            global_summary=global_summary,
            file_id=file_id,
        )

        content_pages = sum(1 for p in pages if p.meaningful)
        logger.success(
            f"[PrecisionEngine] 完成: {len(pages)} 页 "
            f"({content_pages} 有内容) → {len(chunks)} chunks"
        )
        return chunks
