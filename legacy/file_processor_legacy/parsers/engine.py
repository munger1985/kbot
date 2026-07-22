import re, os
import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from enum import Enum
from typing import Any
from loguru import logger
from typing_extensions import override

# Docling Core 分块与序列化模块
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingSerializerProvider
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownTableSerializer,
    MarkdownPictureSerializer,
)
from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.types.doc.document import (
    DoclingDocument,
    DescriptionAnnotation,
    PictureItem,
    TableItem,
    TextItem, 
    TitleItem,
    SectionHeaderItem  # 用于标题处理的专用类
)
# 项目依赖
from .utils import ModelTask, ParserToolLib
from ..parser_schema import DocParserParams, ChunkResult
from platform_clients import AIModelClient
from agent.prompt import default_prompt

class OutputFormat(str, Enum):
    """输出格式枚举类
    
    定义文档转换支持的输出格式类型
    """
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    DOCTAGS = "doctags"
    CHUNKS = "chunks"


class VLMAnnotationPictureSerializer(MarkdownPictureSerializer):
    """自定义图片序列化器：输出注入VLM描述的图片内容
    
    继承自MarkdownPictureSerializer，重写序列化逻辑，增加VLM生成的图片描述信息，
    并处理图片的物理存储。
    """
    @override
    def serialize(self, *, item: PictureItem, doc: DoclingDocument, **kwargs: Any) -> tuple[SerializationResult, str]:
        """序列化图片项，包含VLM描述和图片存储
        
        Args:
            item: 图片项对象
            doc: 文档根对象
            **kwargs: 额外参数，需包含image_dir指定图片存储目录
            
        Returns:
            元组，包含序列化结果和图片文件名
        """
        text_parts = []
        image_name = ""  # 初始化，避免UnboundLocalError
        
        # 获取外部传入的图片保存目录
        image_root = Path(kwargs.get("image_dir", "data/images"))
        image_root.mkdir(parents=True, exist_ok=True)

        # 1. 物理保存图片文件
        if item.image and item.image.pil_image:
            # 使用唯一ID命名，避免同名PDF的图片被覆盖
            image_name = f"pic_{item.self_ref.replace('/', '_')}.png"
            image_path = image_root / image_name
            item.image.pil_image.save(image_path)
            # logger.debug(f"图片提取并保存成功：{image_path}")

        # 2. 注入VLM生成的描述信息作为引用块
        for annotation in item.annotations:
            if isinstance(annotation, DescriptionAnnotation):
                # 放入引用块，方便RAG识别为补充上下文
                text_parts.append(f"\n> [AI视觉描述]: {annotation.text}\n")
        
        text_res = "\n".join(text_parts) if text_parts else ""
        return create_ser_result(text=text_res, span_source=item), image_name

class VLMEnabledMarkdownProvider(ChunkingSerializerProvider):
    """支持VLM图片描述的Markdown序列化器提供类
    
    扩展ChunkingSerializerProvider，为Markdown序列化器配置自定义的图片序列化器，
    使其能够处理包含VLM描述的图片项。
    """
    def get_serializer(self, doc: DoclingDocument) -> MarkdownDocSerializer:
        """获取配置好的Markdown文档序列化器
        
        Args:
            doc: 文档根对象
            
        Returns:
            配置了VLM图片序列化器的MarkdownDocSerializer实例
        """
        return MarkdownDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            picture_serializer=VLMAnnotationPictureSerializer(),
        )


class DoclingEngine:
    """
    负责文档物理转换的核心引擎。
    封装了 Docling 的配置、OCR 选择以及多进程转换逻辑。
    """

    def __init__(self, artifacts_path: str, pool_executor: ProcessPoolExecutor | None = None):
        self.artifacts_path = artifacts_path
        self.executor = pool_executor
        self.model_task = ModelTask()
        self.model_client = AIModelClient()
        # [新增] VLM 描述缓存，Key 为图片指纹(hash)，Value 为描述文本
        self._vlm_cache = {}
        self._vlm_enhancement_cache = {}

    async def convert_document(
        self,
        file_id: str,
        file_path: str,
        params: DocParserParams,
        output_format: OutputFormat = OutputFormat.MARKDOWN
    ) -> str | dict | list[ChunkResult]:
        """文档解析方法的入口"""
        # 自动推导 do_ocr：配置了 AI OCR 模型则跳过内置 OCR
        effective_do_ocr = params.effective_do_ocr
        logger.info(
            f"OCR 策略: effective_do_ocr={effective_do_ocr}, "
            f"ocr_model={params.ocr_model or 'None'}, "
            f"vlm_model={params.vlm_model or 'None'}"
        )

        loop = asyncio.get_running_loop()
        try:
            # vlm 模式：仅渲染，不做文本提取
            engine_mode = params.engine_mode or "auto"
            if engine_mode == "precision":
                engine_mode = "vlm"
            file_ext = Path(file_path).suffix.lower()
            if file_ext == ".pdf" and engine_mode == "vlm":
                doc = await loop.run_in_executor(
                    self.executor,
                    _do_convert_render_only,
                    str(file_path),
                    self.artifacts_path,
                    params.image_scale,
                )
            else:
                doc = await loop.run_in_executor(
                    self.executor,
                    _do_convert,
                    str(file_path),
                    self.artifacts_path,
                    effective_do_ocr,
                    params.ocr_engine or "tesseract",
                    params.image_scale,
                )
        except Exception as e:
            logger.error(f"子进程转换异常: {file_path}, 详情: {e}")
            raise

        # 判断后续流程
        if output_format == OutputFormat.CHUNKS:
            # engine_mode 和 file_ext 已在上面计算

            # ═══════════════════════════════════════════════════════
            # vlm 模式：PDF 纯视觉解析（走完全独立的 pipeline）
            # ═══════════════════════════════════════════════════════
            if file_ext == ".pdf" and engine_mode == "vlm":
                return await self._process_pdf_vlm(doc, params, file_id)

            # ═══════════════════════════════════════════════════════
            # Docling 文本提取 pipeline
            # Word/PPT/Excel 也走这里
            # ═══════════════════════════════════════════════════════

            # --- Stage 1: 扫描件检测 + DS OCR 全页增强 ---
            is_scanned = self._detect_scanned_document(doc, file_ext)
            if is_scanned and params.ocr_model:
                logger.info(f"检测到扫描件 PDF，启用 DS OCR 全页增强")
                await self._enhance_scanned_with_dsocr(doc, params, file_id)
            elif is_scanned:
                logger.info(f"检测到扫描件 PDF，但未配置 AI OCR 模型，依赖内置 OCR 结果")

            # --- Stage 2: AI 增强（DS OCR + VLM） ---
            if params.ocr_model or params.vlm_model:
                await self._enhance_document_content(doc, params, file_ext, file_id)

            # --- Stage 3: 统一级联分块生成 ---
            from .chunk_generator import ChunkerGenerator
            from .hierarchy_builder import HierarchyBuilder
            from .layout_cluster import LayoutClusterer
            from .page_span_stitcher import PageSpanStitcher

            chunker = ChunkerGenerator(params)

            # 文本解析质量门阈值
            threshold = 0.70

            logger.info(
                f"[Cascade] {engine_mode} 模式启动 | "
                f"threshold={threshold} | vlm={params.vlm_model or '无'}"
            )

            # ── Step A: 构建层级树 ──
            builder = HierarchyBuilder(doc)
            hierarchy = builder.build()
            clusterer = LayoutClusterer()
            hierarchy = clusterer.correct_reading_order(hierarchy)
            stitcher = PageSpanStitcher()
            hierarchy = stitcher.stitch(hierarchy)

            # ── Step B: 全局摘要（1 次 LLM 调用）──
            global_summary = await chunker._get_global_summary(doc)

            # ── Step C: 结构质量门 ──
            from .structure_quality_gate import StructureQualityGate
            gate = StructureQualityGate(threshold=threshold)
            reports = gate.assess(hierarchy)
            repair_pages = [r.page_no for r in reports if r.needs_vlm_repair]

            if not repair_pages:
                logger.info(
                    f"[Cascade] 质量门全部通过，"
                    f"0/{len(reports)} 页需要修复 → 直接分块"
                )
                chunks = await chunker.generate_chunks(
                    doc, file_ext, {}, prebuilt_hierarchy=hierarchy
                )
                await self._index_visual_content(doc, params, file_id, pages=[])
                return await self._maybe_reflect_chunks(
                    chunks, doc, params, global_summary=global_summary
                )

            # ── Step D: VLM 结构修复 ──
            if not params.vlm_model:
                logger.warning(
                    f"[Cascade] {len(repair_pages)} 页需要修复，"
                    f"但 VLM 模型未配置，跳过修复直接分块"
                )
                chunks = await chunker.generate_chunks(
                    doc, file_ext, {}, prebuilt_hierarchy=hierarchy
                )
                await self._index_visual_content(doc, params, file_id, pages=[])
                return await self._maybe_reflect_chunks(
                    chunks, doc, params, global_summary=global_summary
                )

            # 准备每页的全局上下文
            prev_stacks: dict[int, list[str]] = {}
            next_headings: dict[int, str] = {}
            docling_hds: dict[int, list[dict]] = {}
            for page_no in repair_pages:
                prev_stacks[page_no] = _get_prev_heading_stack(hierarchy, page_no)
                next_headings[page_no] = _get_next_first_heading(hierarchy, page_no)
                docling_hds[page_no] = _get_docling_headings_for_page(hierarchy, page_no)

            from .structure_repairer import StructureRepairer
            repairer = StructureRepairer(self.model_client)
            repair_results = await repairer.repair_pages(
                doc=doc,
                repair_pages=repair_pages,
                global_summary=global_summary,
                prev_heading_stacks=prev_stacks,
                next_first_headings=next_headings,
                docling_headings=docling_hds,
                vlm_model=params.vlm_model,
            )

            # ── Step E: 合并修复结果到层级树 ──
            from .hierarchy_merger import HierarchyMerger
            merger = HierarchyMerger()
            hierarchy = merger.merge(hierarchy, repair_results)

            # ── Step F: 分块（传入修复后的层级树）──
            chunks = await chunker.generate_chunks(
                doc, file_ext, {}, prebuilt_hierarchy=hierarchy
            )
            await self._index_visual_content(doc, params, file_id, pages=[])
            return await self._maybe_reflect_chunks(
                chunks, doc, params, global_summary=global_summary
            )

        # 其他格式序列化
        return self._serialize(doc, output_format)

    # ═══════════════════════════════════════════════════════════
    # vlm 模式：PDF 纯视觉解析 pipeline
    # ═══════════════════════════════════════════════════════════

    async def _process_pdf_vlm(
        self,
        doc: DoclingDocument,
        params: DocParserParams,
        file_id: str,
    ) -> list[ChunkResult]:
        """vlm 模式：逐页 VLM Markdown → MarkdownSectionChunker → reflect → chunks。"""
        if not params.vlm_model:
            logger.error("[VLM] vlm 模式需要 VLM 模型，但未配置")
            return []

        from .precision_analyzer import PrecisionAnalyzer
        from .section_chunker import MarkdownSectionChunker
        from .chunk_generator import ChunkerGenerator

        # Step 1: VLM 逐页 Markdown
        analyzer = PrecisionAnalyzer(self.model_client)
        pages = await analyzer.analyze_document(
            doc=doc, vlm_model=params.vlm_model, file_id=file_id,
        )

        # Step 2: 全局摘要
        chunker = ChunkerGenerator(params)
        global_summary = await chunker._get_global_summary(doc)

        # Step 3: Markdown → ChunkResult
        chunker_md = MarkdownSectionChunker()
        chunks = chunker_md.convert(
            pages=pages, global_summary=global_summary, file_id=file_id,
        )

        # Step 4: 图片序列化 + VLM 描述 + 视觉向量入库
        try:
            await self._index_visual_content(doc, params, file_id, pages)
        except Exception as e:
            logger.warning(f"[VLM] 视觉索引失败: {e}")

        # Step 5: ChunkReflector
        return await self._maybe_reflect_chunks(
            chunks, doc, params, global_summary=global_summary,
        )

    async def _index_visual_content(
        self,
        doc,
        params,
        file_id: str,
        pages: list,
    ) -> None:
        """图片保存 + VLM 描述 + VisualIndexer 入库 → extracted_images 表。

        - doc.pictures → 裁剪保存 + VLM 描述 (image_type='figure')
        - doc.pages 整页截图 (image_type='page')
        """
        import os

        kb_id = getattr(params, 'kb_id', None) or "unknown"
        image_dir = getattr(params, 'image_dir', None)
        visual_model = getattr(params, 'visual_model', '')

        if not image_dir:
            logger.warning("[VisualIndex] image_dir 未设置，跳过")
            return
        if not visual_model:
            logger.info(
                "[VisualIndex] visual_model 未配置，跳过 embedding 生成，"
                "仅保存图片及描述到 extracted_images"
            )

        pic_count = len(doc.pictures)
        if pic_count > 0:
            logger.info(f"[VisualIndex] {pic_count} 张嵌入图片")

        # --- 裁剪图片 ---
        pic_serializer = VLMAnnotationPictureSerializer()
        for i, pic in enumerate(doc.pictures):
            try:
                _, img_name = pic_serializer.serialize(item=pic, doc=doc, image_dir=image_dir)
                if not img_name:
                    continue
                img_path = os.path.join(image_dir, img_name)
                page_no = getattr(pic, 'page_no', 0) or 0

                # VLM 描述：优先复用已有 annotation，否则调用 VLM 获取
                desc = ""
                for annotation in pic.annotations:
                    from docling_core.types.doc.document import DescriptionAnnotation
                    if isinstance(annotation, DescriptionAnnotation):
                        text = annotation.text.strip() if annotation.text else ""
                        prov = getattr(annotation, 'provenance', '') or ''
                        if text and text != "[NONE]" and prov not in ("hash_marker", ""):
                            desc = text
                            break
                if not desc:
                    if params.vlm_model and hasattr(pic, 'image') and pic.image and pic.image.pil_image:
                        try:
                            prompt = getattr(params, 'img2txt_prompt', '') or "请用一句话描述这张图片的内容和用途。"
                            desc = await self.model_client.get_vlm_answer(
                                params.vlm_model, pic.image.pil_image, prompt=prompt,
                            )
                        except Exception as e:
                            logger.warning(f"[VisualIndex] VLM desc pic {i}: {e}")

                # 入库（有 visual_model 则同时生成 embedding，否则仅保存图片路径和描述）
                from services.visual.visual_indexer import VisualIndexer
                try:
                    await VisualIndexer().index_extracted_image(
                        file_id=file_id, kb_id=kb_id, page_no=page_no,
                        image_path=img_path, description=desc, image_type="figure",
                        bbox=getattr(pic, 'bbox', None) or {},
                        visual_model=visual_model,
                    )
                except Exception as e:
                    logger.warning(f"[VisualIndex] pic {i} index: {e}")

            except Exception as e:
                logger.warning(f"[VisualIndex] pic {i}: {e}")

        # --- 整页截图（统一写入 extracted_images，image_type='page'）---
        from services.visual.visual_indexer import VisualIndexer
        idx = VisualIndexer()
        for pn, po in doc.pages.items():
            try:
                if not po.image or not po.image.pil_image:
                    continue
                pp = os.path.join(image_dir, f"page_{pn}.png")
                po.image.pil_image.save(pp)
                cap = ""
                for p in pages:
                    if getattr(p, 'page', 0) == pn and getattr(p, 'markdown', ''):
                        cap = p.markdown[:200]
                        break
                await idx.index(
                    file_id=file_id, kb_id=kb_id, page_no=pn,
                    image_path=pp, description=cap, image_type="page",
                    visual_model=visual_model,
                )
            except Exception as e:
                logger.warning(f"[VisualIndex] page {pn}: {e}")

        if pic_count > 0 or doc.pages:
            logger.success(f"[VisualIndex] done: {pic_count} pics, {len(doc.pages)} pages")

    async def _maybe_reflect_chunks(
        self,
        chunks: list[ChunkResult],
        doc,
        params: DocParserParams,
        global_summary: str,
    ) -> list[ChunkResult]:
        """可选：LLM 后反思重组短 chunk。

        仅在 enable_chunk_reflection=True 时执行。
        """
        if not getattr(params, 'enable_chunk_reflection', False):
            logger.info("[ChunkReflector] 开关未开启 (enable_chunk_reflection=false)，跳过")
            return chunks
        if not params.llm_model:
            logger.info("[ChunkReflector] LLM 模型未配置，跳过 chunk 反思")
            return chunks

        from .chunk_reflector import ChunkReflector
        reflector = ChunkReflector(
            global_summary=global_summary or "未知文档",
            llm_model=params.llm_model,
            model_client=self.model_client,
        )
        return await reflector.reflect(chunks)

    def _serialize(self, doc: DoclingDocument, fmt: OutputFormat) -> str | dict:
        """将DoclingDocument序列化为指定格式
        
        Args:
            doc: 文档根对象
            fmt: 输出格式枚举值
            
        Returns:
            序列化结果（字符串或字典）
        """
        if fmt == OutputFormat.MARKDOWN:
            return VLMEnabledMarkdownProvider().get_serializer(doc).serialize().text
        if fmt == OutputFormat.HTML: 
            return doc.export_to_html()
        if fmt == OutputFormat.JSON: 
            return doc.export_to_dict()
        if fmt == OutputFormat.DOCTAGS: 
            return doc.export_to_doctags()
        return ""
    
    def _detect_scanned_document(self, doc: DoclingDocument, file_ext: str) -> bool:
        """检测是否为扫描件 PDF（无原生文本层，内容全为图片）。

        判断条件:
        1. 仅对 PDF 文件检测
        2. 文本项极少（不足页面数），图片项占多数
        """
        ext = file_ext.lower()
        if ext not in [".pdf"]:
            return False

        page_count = len(doc.pages)
        text_count = len(doc.texts)
        picture_count = len(doc.pictures)

        # 扫描件特征：几乎没有文本，但有很多图片（每页至少一张整页图）
        is_scanned = (
            text_count <= page_count * 0.3  # 文本项不足页面数的 30%
            and picture_count >= page_count * 0.5  # 图片项至少覆盖一半页面
        )

        if is_scanned:
            logger.info(
                f"扫描件判定: pages={page_count}, texts={text_count}, "
                f"pictures={picture_count}, text_ratio={text_count/max(page_count,1):.2f}"
            )
        return is_scanned

    async def _enhance_scanned_with_dsocr(
        self, doc: DoclingDocument, params: DocParserParams, file_id: str
    ) -> None:
        """扫描件 PDF 全页 DS OCR 增强。

        对每页调用 DS OCR (grounding prompt) 获取结构化 markdown 文字，
        将识别结果注入为 docling 的 TextItem，供后续 ChunkGenerator 正常处理。

        原始的页面 PictureItem 保留不动，VLM 增强阶段仍可对其生成图片描述。
        """
        if not params.ocr_model:
            return

        tasks = []
        page_map: dict[str, int] = {}

        for page_no, page_obj in doc.pages.items():
            if not page_obj.image or not page_obj.image.pil_image:
                continue

            task_key = f"page:{page_no}"
            page_map[task_key] = page_no

            tasks.append(self.model_task.dsocr_task(
                self.model_client, params.ocr_model,
                "<|grounding|>Convert the document to markdown.",
                task_key,
                page_obj.image.pil_image
            ))

        if not tasks:
            logger.warning("扫描件无有效页面图片，跳过 DS OCR 增强")
            return

        logger.info(f"启动 {len(tasks)} 个全页 DS OCR 任务")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        injected_count = 0
        for res in results:
            if isinstance(res, (BaseException, type(None))) or not isinstance(res, tuple):
                continue

            key, content = res
            if not content or not isinstance(content, str):
                continue

            page_no = page_map.get(key)
            if page_no is None:
                continue

            # 解析 grounding 标记，按元素类型拆分并注入文档
            elements = self._parse_grounding_elements(content, page_no)
            for el in elements:
                item = doc.add_text(
                    label=el["label"],
                    text=el["text"],
                    orig=el["text"],
                    prov=el["prov"],
                )
                # 标注 DS OCR 识别的元素类型，供 ChunkGenerator 映射到 chunk_type
                if item and el.get("dsocr_type") and hasattr(item, "annotations"):
                    item.annotations.append( # type: ignore
                        DescriptionAnnotation(text=el["dsocr_type"], provenance="dsocr_type")
                    )
                injected_count += 1

        logger.success(f"DS OCR 扫描件增强完成: {injected_count} 个文本元素 / {len(results)} 页")

    @staticmethod
    def _parse_grounding_elements(raw: str, page_no: int) -> list[dict]:
        """解析 DS OCR grounding 输出，按元素类型拆分为含 bbox 的结构化列表。

        DS OCR 输出格式：
            text[[x1, y1, x2, y2]]          ← 行级标记（坐标 0-999）
            文本内容...

            table[[x1, y1, x2, y2]]
            | 表头 | ...

            image[[x1, y1, x2, y2]]

        返回: [{"label": DocItemLabel, "text": str, "prov": ProvenanceItem}, ...]
        """
        import re
        from docling_core.types.doc.labels import DocItemLabel
        from docling_core.types.doc.document import ProvenanceItem
        from docling_core.types.doc.base import BoundingBox
        from docling_core.types.doc.base import CoordOrigin

        # 行级标记: type[[x1, y1, x2, y2]]
        block_pattern = re.compile(
            r'^(text|title|table_caption|table|image|code|formula)\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]',
            re.MULTILINE
        )

        elements = []
        matches = list(block_pattern.finditer(raw))

        if not matches:
            # 无标记 → 整段作为纯文本
            clean = re.sub(r'<\|ref\|>.*?<\|/ref\|><\|det\|>\[\[.*?\]\]<\|/det\|>', '', raw)
            clean = clean.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').strip()
            if clean:
                prov = ProvenanceItem(
                    page_no=page_no,
                    bbox=BoundingBox(l=0, t=0, r=1, b=1, coord_origin=CoordOrigin.TOPLEFT),
                    charspan=(0, len(clean)),
                )
                elements.append({"label": DocItemLabel.TEXT, "text": clean, "prov": prov})
            return elements

        # 有标记 → 逐段提取
        for i, m in enumerate(matches):
            elem_type = m.group(1)
            x1, y1, x2, y2 = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))

            # 确定该元素覆盖的文本范围
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            text = raw[start:end].strip()

            # 清洗内联 grounding 标记
            text = re.sub(r'<\|ref\|>.*?<\|/ref\|><\|det\|>\[\[.*?\]\]<\|/det\|>', '', text)
            text = text.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').strip()

            if not text:
                continue

            # 坐标从 0-999 归一化到 0-1
            box = BoundingBox(
                l=max(0.0, min(1.0, x1 / 999.0)),
                t=max(0.0, min(1.0, y1 / 999.0)),
                r=max(0.0, min(1.0, x2 / 999.0)),
                b=max(0.0, min(1.0, y2 / 999.0)),
                coord_origin=CoordOrigin.TOPLEFT,
            )

            # 映射元素类型 → DocItemLabel (TABLE/PICTURE 需特殊处理 → 用 TEXT)
            label_map = {
                "title": DocItemLabel.TITLE,
                "table_caption": DocItemLabel.CAPTION,
                "table": DocItemLabel.TEXT,     # TextItem 不接受 TABLE label
                "image": DocItemLabel.TEXT,     # TextItem 不接受 PICTURE label
                "code": DocItemLabel.CODE,
                "formula": DocItemLabel.FORMULA,
            }
            # DS OCR 元素类型 → 现有 chunk_type (text / table / picture)
            chunk_type_map = {
                "title": "text",
                "text": "text",
                "table_caption": "text",
                "table": "table",
                "image": "picture",
                "code": "text",
                "formula": "text",
            }
            label = label_map.get(elem_type, DocItemLabel.TEXT)
            dsocr_type = chunk_type_map.get(elem_type, "text")

            prov = ProvenanceItem(
                page_no=page_no,
                bbox=box,
                charspan=(0, len(text)),
            )
            elements.append({
                "label": label, "text": text, "prov": prov, "dsocr_type": dsocr_type
            })

        return elements

    async def _enhance_document_content(self, doc: DoclingDocument, params: DocParserParams, file_ext: str, file_id: str) -> None:
        """
        视觉增强：DS OCR 文字识别 + VLM 图片描述。
        对 PictureItem：DS OCR 优先（文字型图片），VLM 兜底（真实照片）。
        对 TableItem：VLM 视觉重构（现有逻辑不变）。
        """
        has_vlm = params.vlm_model is not None
        has_ocr = params.ocr_model is not None

        if not has_vlm and not has_ocr:
            return

        is_ppt = file_ext.lower() in [".pptx", ".ppt"]

        tasks = []

        # --- 情况 A：如果是 PPT，执行整页视觉增强 ---
        if is_ppt:
            logger.info("PPT 模式：忽略单图，启动整页 VLM 增强")
            for page_no, page_obj in doc.pages.items():
                # 现在可以安全访问 .image 了
                if not page_obj.image or not page_obj.image.pil_image:
                    logger.warning(f"第 {page_no} 页未能成功渲染图片")
                    continue
                
                vlm_prompt = f"请简要描述这张幻灯片的内容，不要长篇大论，不要编造不存在的内容。重点识别其中的逻辑关系、架构图、流程图和核心结论。"
                
                tasks.append(self.model_task.vlm_task(
                    self.model_client, params.vlm_model,  # type: ignore
                    vlm_prompt, f"slide:index:{page_no}", page_obj.image.pil_image
                ))
                
        # --- 情况 B：普通文档，执行单图/表格增强 ---
        else:
            # 哈希映射表，用于存储 hash -> [item_indices] 的关系
            hash_to_pic_indices = {}

            # --- 1. 预读文档流，建立上下文映射表 ---
            # 建立 item 内存地址到最近标题文本的映射
            item_id_to_header = {}
            current_header = "前言/背景"

            for item, level in doc.iterate_items():
                # 精确匹配标题类型
                if isinstance(item, (SectionHeaderItem, TitleItem)):
                    current_header = item.text.strip().replace("\n", " ")
                
                # 记录图片和表格的上下文映射
                if isinstance(item, (PictureItem, TableItem)):
                    item_id_to_header[id(item)] = current_header

            # --- 2. 视觉增强：处理图片 (VLM) ---
            for i, pic in enumerate(doc.pictures):
                # 1. 获取唯一 MD5
                img_hash = ParserToolLib.get_image_hash(pic)
                if not img_hash:
                    logger.warning(f"无法计算索引为 {i} 的图片 Hash")
                    continue
                
                # 注入 Hash 标记，供 generate_chunks 物理去重使用
                pic.annotations.append(DescriptionAnnotation(text=img_hash, provenance="hash_marker"))

                # 2. 物理过滤
                raw_img = getattr(pic.image, "pil_image", None)
                # 如果图片不存在，直接跳过
                if raw_img is None:
                    continue

                # 3：小图直接打标 [NONE]，不进 VLM 队列，节省成本
                if raw_img.width < 60 or raw_img.height < 60:
                    pic.annotations.append(DescriptionAnnotation(text="[NONE]", provenance="vlm_inference"))
                    continue

                # 4：检查全局缓存
                if img_hash in self._vlm_cache:
                    pic.annotations.append(DescriptionAnnotation(text=self._vlm_cache[img_hash], provenance="vlm_inference"))
                    continue

                # 4. 任务合并：同一个文档内相同的图只跑一次
                if img_hash not in hash_to_pic_indices:
                    hash_to_pic_indices[img_hash] = [i]
                    pic_context = item_id_to_header.get(id(pic), "未知章节")

                    # 图片处理策略：
                    # 两者都配 → DS OCR 提取文字 + VLM 理解语义，并行协作（图文并茂最佳）
                    # 仅 DS OCR → 纯文字提取（成本优先）
                    # 仅 VLM    → 语义描述（图表/照片）
                    # 都没有    → 依赖 Docling 内置 OCR
                    if has_ocr:
                        ocr_prompt = "Parse the figure."
                        tasks.append(self.model_task.dsocr_task(
                            self.model_client, params.ocr_model, # type: ignore
                            ocr_prompt, f"ocr:hash:{img_hash}", raw_img
                        ))
                    if has_vlm:
                        vlm_final_prompt = default_prompt.format(
                            params.img2txt_prompt,
                            current_header=pic_context
                        )
                        tasks.append(self.model_task.vlm_task(
                            self.model_client, params.vlm_model, # type: ignore
                            vlm_final_prompt, f"pic:hash:{img_hash}", raw_img
                        ))
                else:
                    hash_to_pic_indices[img_hash].append(i)
            
            logger.debug(f"构建完成 Hash 映射表，共 {len(hash_to_pic_indices)} 组唯一图片")

            # --- 3. 处理表格 (VLM) ---
            if has_vlm:
                for i, table in enumerate(doc.tables):
                    # 路径：有图（复杂 Excel 渲染或 PDF 表格）-> VLM 视觉重构
                    if table.image and table.image.pil_image:
                        table_context = item_id_to_header.get(id(table), "未知章节")

                        table_prompt = f"""你是一个专业的文档解析专家。当前表格处于文本上下文：【{table_context}】中。
请将图片中的表格解析为 JSON 格式。
必须严格遵守以下约束：
1. 返回格式必须为：{{"header": "Markdown格式表头", "rows": ["数据行1", "数据行2", ...]}}
2.  rows 数组中的每一项必须是单行 Markdown。
3. 严禁输出任何 JSON 以外的文字。"""

                        tasks.append(self.model_task.vlm_task(
                            self.model_client, params.vlm_model, # type: ignore
                            table_prompt, f"table:index:{i}", table.image.pil_image
                        ))

            if not tasks:
                return
            
        # --- D. 并发执行与智能回填 ---
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, (BaseException, type(None))) or not isinstance(res, tuple):
                if isinstance(res, BaseException):
                    logger.error(f"AI 增强任务执行崩溃: {res}")
                continue
                
            key, content = res 
            if not content: continue
                
            parts = key.split(':')
            if len(parts) < 3: continue

            category, sub_type, identifier = parts[0], parts[1], parts[2]

            # 1. 处理 DS OCR 结果 (ocr:hash:abcd...)
            if category == "ocr":
                clean_content = content.strip() if isinstance(content, str) else ""
                if sub_type == "hash":
                    img_hash = identifier
                    if clean_content:
                        # 回填给所有引用此 Hash 的图片项
                        for p_idx in hash_to_pic_indices.get(img_hash, []):
                            doc.pictures[p_idx].annotations.append(
                                DescriptionAnnotation(
                                    text=clean_content,
                                    provenance="ocr_inference"
                                )
                            )
                    else:
                        logger.debug(f"图片 {identifier} 的 DS OCR 返回内容为空，将使用 VLM 兜底")

            # 2. 处理表格 (table:index:0)
            elif category == "table":
                if identifier.isnumeric():
                    idx = int(identifier)
                    doc.tables[idx].annotations.append(
                        DescriptionAnnotation(text=content, provenance="vlm_table_rebuild")
                    )

            # 3. 处理图片 VLM 结果 (pic:hash:abcd...)
            elif category == "pic":
                clean_content = content.strip() if content else ""
                if not clean_content:
                    clean_content = "[NONE]" # 改为 [NONE] 方便 generate_chunks 统一过滤
                    logger.warning(f"图片 {identifier} 的 VLM 返回内容为空，已使用占位符")

                if sub_type == "hash":
                    img_hash = identifier
                    self._vlm_cache[img_hash] = clean_content # 存入缓存

                    # 回填给所有引用此 Hash 的图片项
                    for p_idx in hash_to_pic_indices.get(img_hash, []):
                        doc.pictures[p_idx].annotations.append(
                            DescriptionAnnotation(
                                text=clean_content,
                                provenance="vlm_inference"
                            )
                        )
                elif sub_type == "index" and identifier.isnumeric():
                    # 兼容旧的按索引回填模式
                    idx = int(identifier)
                    if idx < len(doc.pictures):
                        doc.pictures[idx].annotations.append(
                            DescriptionAnnotation(text=clean_content, provenance=f"vlm_{params.vlm_model}")
                        )
                    else:
                        logger.error(f"索引越界：尝试回填不存在的图片索引 {idx}")

            # 4. 处理标题/层级 (heading:index:0)
            elif category == "heading":
                if identifier.isnumeric():
                    idx = int(identifier)
                    # 安全获取文本项并回填层级
                    try:
                        target_item = doc.texts[idx]
                        annos = getattr(target_item, "annotations", None)
                        if isinstance(annos, list):
                            level_val = int(re.sub(r'\D', '', str(content)))
                            annos.append(
                                DescriptionAnnotation(text=str(level_val), provenance="structure_level")
                            )
                        else:
                            logger.warning(f"处理标题/层级时获取的annotations不是list，跳过处理")
                    except (IndexError, ValueError) as e:
                        logger.error(f"处理标题/层级时获取annotations失败： {e}")
                        pass

            # 5. 处理PPT slide
            elif category == "slide":
                if identifier.isnumeric():
                    page_no = int(identifier)
                    try:
                        current_page = doc.pages.get(page_no)
                        if current_page:
                            # 写入 PPT 整页截图的 image_name
                            slide_img_name = None
                            if current_page.image and current_page.image.pil_image:
                                # 使用唯一ID命名，避免同名PDF的图片被覆盖
                                slide_img_name = f"slide_page_{page_no}_{file_id}.png"
                                image_root = Path(params.image_dir or "data/images")
                                image_root.mkdir(parents=True, exist_ok=True)
                                image_path = image_root / slide_img_name
                                current_page.image.pil_image.save(image_path)
                                if file_id not in self._vlm_enhancement_cache:
                                    self._vlm_enhancement_cache[file_id] = {}
                                self._vlm_enhancement_cache[file_id][page_no] = {
                                    "description": content,
                                    "image_name": slide_img_name
                                }
                                logger.debug(f"第 {page_no} 页 VLM 描述已动态挂载")
                        else:
                            logger.warning(f"Doc对象中未找到页码为 {page_no} 的页面，跳过处理")
                        
                        # logger.debug(f"生成的PPT描述：{self._vlm_enhancement_cache}")

                    except Exception as e:
                        logger.error(f"处理 PPT 回填异常: {e}")
            else:
                logger.warning(f"解析文档时通过VLM增强图片和表格时获取的category是未知类型，跳过处理")
    
# ═══════════════════════════════════════════════════════════════
# 级联管线辅助函数 — 从层级树中提取 VLM 修复所需全局上下文
# ═══════════════════════════════════════════════════════════════

def _get_prev_heading_stack(hierarchy, page_no: int) -> list[str]:
    """获取指定页之前最后一页的标题栈（按层级顺序）。

    遍历整个文档的标题节点，找到 page_no 前一页的最后一个标题栈。
    """
    from .hierarchy_builder import SemanticNode

    # 收集全文档标题，按页码排序
    all_headings: list[tuple[int, int, str]] = []  # (page_no, level, text)

    def _collect_headings(node: SemanticNode):
        if node.node_type == 'title' and node.text.strip():
            all_headings.append((node.page_num, node.level, node.text.strip()))
        for child in node.children:
            _collect_headings(child)

    _collect_headings(hierarchy)

    if not all_headings:
        return []

    # 找到 page_no 之前的最大页码
    prev_page = 0
    for pg, _, _ in all_headings:
        if pg < page_no and pg > prev_page:
            prev_page = pg

    if prev_page == 0:
        return []

    # 重建上一页的标题栈
    stack: list[str] = []
    for pg, lv, text in all_headings:
        if pg > prev_page:
            break
        if pg == prev_page:
            while stack and len(stack) >= lv:
                stack.pop()
            stack.append(text)

    return stack


def _get_next_first_heading(hierarchy, page_no: int) -> str:
    """获取指定页之后第一页的第一个标题文本。"""
    from .hierarchy_builder import SemanticNode

    first_text = ""

    def _find_first(node: SemanticNode):
        nonlocal first_text
        if first_text:
            return
        if node.node_type == 'title' and node.page_num > page_no and node.text.strip():
            first_text = node.text.strip()
            return
        for child in node.children:
            _find_first(child)

    _find_first(hierarchy)
    return first_text


def _get_docling_headings_for_page(hierarchy, page_no: int) -> list[dict]:
    """提取指定页 Docling 识别的标题列表（供 VLM 验证）。"""
    from .hierarchy_builder import SemanticNode

    headings: list[dict] = []

    def _collect(node: SemanticNode):
        if node.page_num == page_no and node.node_type == 'title' and node.text.strip():
            bbox_hint = "未知"
            if node.bbox:
                y_center = (node.bbox[1] + node.bbox[3]) / 2
                if y_center < 0.33:
                    bbox_hint = "页面上部"
                elif y_center < 0.66:
                    bbox_hint = "页面中部"
                else:
                    bbox_hint = "页面下部"
            headings.append({
                "text": node.text.strip(),
                "level": node.level,
                "bbox_hint": bbox_hint,
            })
        for child in node.children:
            _collect(child)

    _collect(hierarchy)
    return headings


def _do_convert(file_path: str, artifacts_path: str, do_ocr: bool, ocr_engine: str, image_scale: float) -> DoclingDocument:
    """子进程执行函数：在子进程内部初始化转换器"""
    # 延迟导入，减少主进程启动负担
    import subprocess
    from pathlib import Path
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption, PowerpointFormatOption, ExcelFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions, EasyOcrOptions, TableStructureOptions

    table_opts = TableStructureOptions(do_cell_matching=True)
    pipeline_opts = PdfPipelineOptions(
        artifacts_path=artifacts_path,
        do_ocr=do_ocr,
        do_chart_extraction=False,
        generate_table_images=True,
        generate_picture_images=True,
        generate_page_images=True, # 始终开启，方便后续视觉增强或调试
        table_structure_options=table_opts,
        images_scale=image_scale
    )

    # OCR 配置
    if ocr_engine.lower() == "tesseract":
        pipeline_opts.ocr_options = TesseractOcrOptions(lang=["chi_sim", "eng"])
    else:
        pipeline_opts.ocr_options = EasyOcrOptions(lang=["ch_sim", "en"])

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
            InputFormat.DOCX: WordFormatOption(pipeline_options=pipeline_opts),
            InputFormat.PPTX: PowerpointFormatOption(pipeline_options=pipeline_opts), # 显式添加 PPTX
            InputFormat.XLSX: ExcelFormatOption(pipeline_options=pipeline_opts), # 显式添加 EXCEL
        }
    )
    file_ext = Path(file_path).suffix.lower()
    # PPT 转换为 PDF 处理，避免渲染失败
    if file_ext in [".pptx", ".ppt"]:
        # 1. 强制转为 PDF（这是为了获得高保真的全页渲染能力）
        temp_dir = os.path.dirname(file_path)
        subprocess.run([
            'soffice', '--headless', '--convert-to', 'pdf', 
            '--outdir', temp_dir, file_path
        ], check=True)
        
        pdf_path = os.path.join(temp_dir, Path(file_path).stem + ".pdf")
        
        # 2. 让 Docling 解析这个生成的 PDF
        # 这样 Docling 就会把这个 PDF 当作源文件，自动生成 page_obj.image
        result = converter.convert(pdf_path)
    else:
        result = converter.convert(file_path)
    # 返回 DoclingDocument 对象，该对象在 Docling 2.x 中是支持 pickle 的
    return result.document


def _do_convert_render_only(
    file_path: str, artifacts_path: str, image_scale: float
) -> DoclingDocument:
    """子进程执行函数（vlm 模式）：仅渲染页面图 + 提取嵌入图片，不做文本 OCR。

    与 _do_convert 的区别：do_ocr=False，generate_table_images=False。
    """
    import subprocess
    from pathlib import Path
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_opts = PdfPipelineOptions(
        artifacts_path=artifacts_path,
        do_ocr=False,
        do_chart_extraction=False,
        generate_table_images=False,
        generate_picture_images=True,
        generate_page_images=True,
        images_scale=image_scale,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
        }
    )

    file_ext = Path(file_path).suffix.lower()
    if file_ext in [".pptx", ".ppt"]:
        temp_dir = os.path.dirname(file_path)
        subprocess.run([
            'soffice', '--headless', '--convert-to', 'pdf',
            '--outdir', temp_dir, file_path
        ], check=True)
        pdf_path = os.path.join(temp_dir, Path(file_path).stem + ".pdf")
        result = converter.convert(pdf_path)
    else:
        result = converter.convert(file_path)
    return result.document