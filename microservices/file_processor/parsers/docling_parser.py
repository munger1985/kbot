import os
import re
import asyncio
from pathlib import Path
from enum import Enum
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from typing_extensions import override

# Docling core imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    EasyOcrOptions, 
    TesseractOcrOptions,
    TableStructureOptions
)
# Docling document conversion format options
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption

# Docling Core chunking and serialization
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingSerializerProvider
from docling_core.types.doc.labels import DocItemLabel
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
    SectionHeaderItem  # Special class for header processing
)

# Project dependencies
from utils.clients import AIModelClient
from ..parser_schema import DocParserParams
from .doc_structure import SemanticLevelCorrector


class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    DOCTAGS = "doctags"
    CHUNKS = "chunks"


class VLMAnnotationPictureSerializer(MarkdownPictureSerializer):
    """Custom picture serializer: Output VLM-injected descriptions."""
    @override
    def serialize(self, *, item: PictureItem, doc: DoclingDocument, **kwargs: Any) -> tuple[SerializationResult, str]:
        text_parts = []
        
        # Get image save directory (passed from external)
        image_root = Path(kwargs.get("image_dir", "data/images"))
        image_root.mkdir(parents=True, exist_ok=True)

        # 1. Physically save image
        if item.image and item.image.pil_image:
            # Use unique ID for naming to avoid overwriting images from same-named PDFs
            image_name = f"pic_{item.self_ref.replace('/', '_')}.png"
            image_path = image_root / image_name
            item.image.pil_image.save(image_path)
            # logger.debug(f"Extracted image saved successfully: {image_path}")

        # 2. Inject VLM description as reference block
        for annotation in item.annotations:
            if isinstance(annotation, DescriptionAnnotation):
                # Place in reference block for easy RAG identification as supplementary context
                text_parts.append(f"\n> [AI Visual Description]: {annotation.text}\n")
        
        text_res = "\n".join(text_parts) if text_parts else ""
        return create_ser_result(text=text_res, span_source=item), image_name

class VLMEnabledMarkdownProvider(ChunkingSerializerProvider):
    """Markdown serializer provider with VLM image description support"""
    def get_serializer(self, doc: DoclingDocument) -> MarkdownDocSerializer:
        return MarkdownDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            picture_serializer=VLMAnnotationPictureSerializer(),
        )

class DoclingDocProcessor:
    """
    Document processing class based on Docling framework with enhanced features:
    - VLM-powered image description
    - Semantic-aware header detection
    - Hierarchical chunking for RAG/ES
    - Multi-format output support (Markdown/HTML/JSON/Chunks)
    """
    def __init__(self, local_artifacts_path: str, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.vlm_semaphore = asyncio.Semaphore(5)  # Rate limiting for VLM API calls
        self.local_artifacts_path = local_artifacts_path
        self.vlm_client = AIModelClient()
        # Universal title stack: Keep only recent levels
        self.title_hierarchy = []
        # Universal header regex (supports common patterns: 1. / 1 / 一、 / Chapter X / 1.1.1)
        self.title_re = re.compile(r'^(\d+(\.\d+)*|第[一二三四五六七八九十\d]+[章节条款项]|[一二三四五六七八九十]+)\s*[.、：]?\s*\S')
        self.max_hierarchy_depth = 3  # Header level limit (default: 3 levels)
        self.corrector = SemanticLevelCorrector()
        self.serializer = VLMAnnotationPictureSerializer()

    def _is_title_item(self, item) -> bool:
        """Check if document item is a header/title"""
        text = getattr(item, "text", "").strip()
        if len(text) < 2 or len(text) > 200:
            return False
        if isinstance(item, SectionHeaderItem):
            return True
        if isinstance(item, TextItem):
            return bool(self.title_re.match(text))
        return False
    
    def _update_hierarchy(self, item):
        """Update title hierarchy stack with new header items"""
        if not self._is_title_item(item):
            return
        text = getattr(item, "text", "").strip()

        # Extract numeric hierarchy level: 1 → level 1; 1.1 → level 2; 1.1.1→level 3; Chapter X→level 1
        # Universal rule: Split by "." to get level count
        dot_part = re.match(r'^(\d+(\.\d+)*)', text)
        if dot_part:
            level = len(dot_part.group(1).split('.'))
        else:
            level = 1  # Non-numeric headers treated as level 1

        # Stack maintenance: Pop all higher/equal levels, push current level
        while self.title_hierarchy and self.title_hierarchy[-1]["level"] >= level:
            self.title_hierarchy.pop()
        self.title_hierarchy.append({"text": text, "level": level})

    def _get_current_hierarchy(self) -> list[str]:
        """Return current complete title chain (text only)"""
        return [x["text"] for x in self.title_hierarchy[:self.max_hierarchy_depth]]
    
    def _get_page_num(self, item: Any) -> int | None:
        """
        Enhanced page number extraction: Supports PDF (actual page numbers) + Word (logical position)
        """
        # Attempt 1: Extract PDF actual page number from prov attribute
        page_num = None
        item_prov = getattr(item, "prov", [])
        if item_prov:
            for prov in item_prov:
                if hasattr(prov, "page_no") and prov.page_no is not None:
                    page_num = prov.page_no
                    break
        
        # Attempt 2: Extract from item's page_reference (Docling 2.75+ attribute)
        if page_num is None and hasattr(item, "page_reference"):
            page_ref = getattr(item, "page_reference", {})
            page_num = page_ref.get("page_no") or page_ref.get("page_index")
        
        return page_num

    def _detect_is_zh(self, doc: DoclingDocument) -> bool:
        """Detect if document is primarily Chinese content"""
        sample = "".join([t.text for t in doc.texts[:10]])
        if not sample: 
            return False
        zh_chars = len(re.findall(r'[\u4e00-\u9fff]', sample))
        return zh_chars / (len(sample) + 1) > 0.1

    async def _process_vlm_descriptions(self, doc: DoclingDocument, params: DocParserParams) -> None:
        """
        Process document images with VLM to generate descriptive annotations
        """
        if not params.use_vlm or not params.vlm_model or not doc.pictures:
            return

        # Filter valid images: check both image existence and size constraints
        # Model requires both height and width > 10 pixels
        valid_pictures = []
        for pic in doc.pictures:
            if pic.image and pic.image.pil_image:
                width, height = pic.image.pil_image.size
                if width > 10 and height > 10:
                    valid_pictures.append(pic)
                else:
                    logger.warning(f"Skipping image with invalid size {width}x{height} (must be >10x10)")

        if not valid_pictures:
            logger.info("No valid images (size >10x10) found for VLM processing")
            return

        target_prompt = params.vlm_prompt or "Please describe this image in detail. Extract key data if it's a chart/table."
        tasks = [
            self._vlm_task(self.vlm_client, params.vlm_model, target_prompt, i, pic.image.pil_image)
            for i, pic in enumerate(valid_pictures)
        ]

        if tasks:
            results = await asyncio.gather(*tasks)
            for idx, desc in results:
                if desc:
                    valid_pictures[idx].annotations.append(
                        DescriptionAnnotation(text=desc, provenance=f"vlm_{params.vlm_model}")
                    )

    async def _vlm_task(self, client, model_name, prompt, index, image_obj) -> tuple:
        """Async VLM processing task with rate limiting"""
        async with self.vlm_semaphore:
            try:
                res = await client.call_vlm_model(model_name=model_name, image=image_obj, prompt=prompt)
                return index, res
            except Exception as e:
                logger.error(f"VLM processing failed: {e}")
                return index, None
    
    def _generate_chunks(self, doc: DoclingDocument, params: DocParserParams) -> list[dict]:
        chunk_results = []
        current_chunk_num = 1
        active_path = []  # 严格的路径栈
        pending_header_context = ""

        for item, depth in doc.iterate_items():
            image_name = None
            
            raw_text = getattr(item, "text", "").strip()
            if not raw_text and not isinstance(item, (TableItem, PictureItem)):
                continue

            is_header = isinstance(item, SectionHeaderItem)
            detected_level = self.corrector.get_level(item, raw_text) if is_header else None
            
            # --- 拦截逻辑：二次判定，将“伪标题”降级 ---
            if is_header and detected_level is not None:
                if raw_text.endswith(('；', '。', '：', ':')) and len(raw_text) < 5:
                    is_header = False 
                    detected_level = None
            # --- 逻辑分支 A: 真正的有效标题 ---
            if is_header and detected_level is not None:
                if detected_level == 1:
                    active_path = []
                
                # logger.debug(f"[PathTrace] New Header: '{raw_text}' | Detected Level: {detected_level}")

                # 弹出旧路径
                while len(active_path) >= detected_level:
                    active_path.pop()
                
                # 压入新路径
                if not active_path or active_path[-1] != raw_text:
                    active_path.append(raw_text)
                
                # 标题特有状态
                pending_header_context = raw_text
                current_type = "heading"
                structure_level = detected_level

            # --- 逻辑分支 B: 普通文本、表格、图片，或被降级的伪标题 ---
            else:
                current_type = "text"
                structure_level = len(active_path) + 1
                # 注意：这里不需要清空 pending_header_context，它会被下方的 metadata 使用

            # --- 内容封装 ---
            content_list = []
            if isinstance(item, TableItem):
                current_type = "table"
                content_list.append(item.export_to_markdown(doc=doc))
            elif isinstance(item, PictureItem):
                current_type = "picture"
                ser_result, image_name = self.serializer.serialize(item=item, doc=doc, image_dir=params.image_dir)
                content_list.append(ser_result.text)
            else:
                content_list.append(raw_text)

            # --- 封装输出 ---
            for sub_idx, sub_content in enumerate(content_list, start=1):
                # 记录当前路径状态的快照
                final_path_list = list(active_path)
                
                # 2. 弱语义标题增强逻辑
                # 如果当前路径不为空，且最后一个标题太短（如 "注："），
                # 且它前面还有父级标题，我们就不做特殊处理（因为 final_path_list 已经包含全路径）。
                # 但在检索展示时，为了防止 UI 只显示末尾，我们可以在这里通过逻辑确保 context 的丰富度。
                
                current_header_ctx = pending_header_context

                # 如果当前标题是“弱语义”标题，我们需要在 metadata 中保留更完整的上下文
                if is_header and len(raw_text) <= 3:
                    if len(active_path) > 1:
                        # 将父级标题与当前弱标题合并作为 context，例如 "理赔须知 (注：)"
                        current_header_ctx = f"{active_path[-2]} ({raw_text})"

                # 清理 metadata 中的特殊字符,确保 JSON 兼容性
                node_path = str(item.self_ref) if hasattr(item, 'self_ref') else ""
                node_path = node_path.replace('\x00', '')  # 移除空字节
                node_path = node_path.strip()

                chunk_results.append({
                    "content": sub_content,
                    "path_names": final_path_list,  # 传入 List，供后续 flatten_path_names 处理
                    "structure_level": structure_level,
                    "chunk_type": current_type,
                    "metadata": {
                        "chunk_num": current_chunk_num,
                        "sub_index": sub_idx,
                        "page_num": self._get_page_num(item),
                        "node_path": node_path,
                        "image_name": image_name,
                        "header_context": current_header_ctx 
                    }
                })
                current_chunk_num += 1

        return chunk_results
    
    async def convert_document(
        self, 
        file_path: str, 
        params: DocParserParams, 
        output_format: OutputFormat = OutputFormat.MARKDOWN
    ) -> str | dict | list[dict]:
        """
        Core document parsing logic: PDF/Word -> DoclingDocument -> VLM injection -> Format conversion
        
        Args:
            file_path: Path to input document (PDF/DOCX)
            params: Document parsing parameters
            output_format: Desired output format
            
        Returns:
            Formatted output (Markdown/HTML/JSON string or chunks list)
        """
        # 1. Configuration initialization
        pipeline_opts = PdfPipelineOptions(artifacts_path=self.local_artifacts_path)
        pipeline_opts.do_ocr = params.do_ocr
        pipeline_opts.generate_picture_images = params.generate_picture_images
        pipeline_opts.images_scale = params.image_scale
        
        # Table configuration
        pipeline_opts.table_structure_options = TableStructureOptions(do_cell_matching=True)
        pipeline_opts.do_table_structure = True

        # OCR engine configuration
        engine = (params.ocr_engine or "easyocr").lower()
        if engine == "tesseract":
            pipeline_opts.ocr_options = TesseractOcrOptions(lang=["chi_sim", "eng"])
        else:
            pipeline_opts.ocr_options = EasyOcrOptions(lang=["ch_sim", "en"])

        # 2. Converter initialization
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
                InputFormat.DOCX: WordFormatOption(pipeline_options=pipeline_opts)
            }
        )
        
        # 3. Execute conversion
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, converter.convert, file_path)
        doc = result.document

        # 4. VLM injection for image descriptions
        await self._process_vlm_descriptions(doc, params)

        # 5. Generate output in desired format
        if output_format == OutputFormat.CHUNKS:
            return self._generate_chunks(doc, params)

        return self._serialize(doc, output_format)

    def _serialize(self, doc: DoclingDocument, fmt: OutputFormat) -> str | dict:
        """Serialize DoclingDocument to specified format"""
        if fmt == OutputFormat.MARKDOWN:
            return VLMEnabledMarkdownProvider().get_serializer(doc).serialize().text
        if fmt == OutputFormat.HTML: 
            return doc.export_to_html()
        if fmt == OutputFormat.JSON: 
            return doc.export_to_dict()
        if fmt == OutputFormat.DOCTAGS: 
            return doc.export_to_doctags()
        return ""