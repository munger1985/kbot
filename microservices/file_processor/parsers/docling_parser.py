import os
import re
import asyncio
from pathlib import Path
from enum import Enum
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from typing_extensions import override

# Docling 核心模块
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    EasyOcrOptions, 
    TesseractOcrOptions,
    TableStructureOptions
)
from docling.document_converter import (
    DocumentConverter, 
    PdfFormatOption, 
    WordFormatOption, 
    ExcelFormatOption
)
# Docling Core 
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
    SectionHeaderItem
)

# 项目依赖
from utils.clients import AIModelClient
from ..parser_schema import DocParserParams


class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    DOCTAGS = "doctags"
    CHUNKS = "chunks"


class VLMAnnotationPictureSerializer(MarkdownPictureSerializer):
    """自定义图片序列化器：整合 VLM 描述"""
    @override
    def serialize(self, *, item: PictureItem, doc: DoclingDocument, **kwargs: Any) -> tuple[SerializationResult, str]:
        text_parts = []
        image_root = Path(kwargs.get("image_dir", "data/images"))
        image_root.mkdir(parents=True, exist_ok=True)

        image_name = ""
        if item.image and item.image.pil_image:
            image_name = f"pic_{item.self_ref.replace('/', '_')}.png"
            image_path = image_root / image_name
            item.image.pil_image.save(image_path)

        # 提取预处理阶段注入的 VLM 描述
        for annotation in item.annotations:
            if isinstance(annotation, DescriptionAnnotation):
                vlm_text = getattr(annotation, "text", "")
                if vlm_text:
                    text_parts.append(f"\n> [AI视觉描述]: {vlm_text}\n")
        
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
    
class DoclingDocProcessor:
    """
    深度优化版：集成 VLM 视觉引导、表格重构及语义聚合缓冲
    """
    def __init__(self, local_artifacts_path: str, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.vlm_semaphore = asyncio.Semaphore(5)
        self.local_artifacts_path = local_artifacts_path
        self.vlm_client = AIModelClient()
        self.serializer = VLMAnnotationPictureSerializer()
        self.title_re = re.compile(r'^(\d+(\.\d+)*|第[一二三四五六七八九十\d]+[章节条款项]|[一二三四五六七八九十]+)\s*[.、：]?\s*\S')

    # --- 1. 核心 VLM 驱动层 ---

    async def _vlm_task(self, client, model_name, prompt, index, image_obj) -> tuple:
        """带限流的 VLM 请求"""
        async with self.vlm_semaphore:
            try:
                res = await client.call_vlm_model(model_name=model_name, image=image_obj, prompt=prompt)
                return index, res
            except Exception as e:
                logger.error(f"VLM 推理异常 (Index {index}): {e}")
                return index, None

    async def _pre_enhance_with_vlm(self, doc: DoclingDocument, params: DocParserParams):
        """预处理：并行调用 VLM 重构表格和描述图片"""
        if not params.use_vlm:
            return

        tasks = []
        # 处理表格：解决复杂 Excel 识别断裂
        for i, table in enumerate(doc.tables):
            if table.image and table.image.pil_image:
                prompt = "请将此图片中的表格还原为标准的 Markdown 格式。确保列不丢失，合并单元格请填充内容，直接输出表格，不含解释。"
                tasks.append(self._vlm_task(self.vlm_client, params.vlm_model, prompt, f"table_{i}", table.image.pil_image))

        # 处理图片描述
        for i, pic in enumerate(doc.pictures):
            if pic.image and pic.image.pil_image:
                prompt = params.vlm_prompt or "请详细描述此图片，提取图表关键数据或文字信息。"
                tasks.append(self._vlm_task(self.vlm_client, params.vlm_model, prompt, f"pic_{i}", pic.image.pil_image))

        if not tasks:
            return

        results = await asyncio.gather(*tasks)
        
        # 结果回填
        for key, content in results:
            if not content: continue
            
            idx = int(key.split('_')[1])
            if key.startswith("table"):
                doc.tables[idx].annotations.append(
                    DescriptionAnnotation(text=content, provenance="vlm_table_rebuild")
                )
            elif key.startswith("pic"):
                doc.pictures[idx].annotations.append(
                    DescriptionAnnotation(text=content, provenance=f"vlm_{params.vlm_model}")
                )

    # --- 2. 增强型分块层 (Chunking) ---

    def _get_page_num(self, item: Any) -> int:
        """获取页码，默认为1"""
        if hasattr(item, "page_reference"):
            return item.page_reference.get("page_no", 1)
        return 1

    async def _generate_chunks(self, doc: DoclingDocument, result, params: DocParserParams) -> list[dict]:
        """
        视觉驱动的分块逻辑：包含标题研判、表格重构、以及图片语义提取
        """
        chunk_results = []
        active_path = []
        buffer_content = []
        buffer_meta = None

        def flush_buffer():
            nonlocal buffer_content, buffer_meta
            if not buffer_content: 
                return
            
            # 合并内容并深度清洗：去除多余空格和换行
            combined_text = "\n".join(buffer_content).strip()
            
            # 关键过滤：如果内容为空，或者只是“暂无内容”，或者字数太少，直接丢弃
            if not combined_text or combined_text == "暂无内容" or len(combined_text) < 1:
                buffer_content = []
                buffer_meta = None
                return

            chunk_results.append({
                "content": combined_text,
                "path_names": list(active_path),
                "structure_level": len(active_path) + 1,
                "chunk_type": "text",
                "metadata": buffer_meta or {}
            })
            buffer_content = []
            buffer_meta = None

        for item, _ in doc.iterate_items():
            raw_text = getattr(item, "text", "").strip()
            # 过滤 1：跳过完全无内容的元素
            if not raw_text and not isinstance(item, (TableItem, PictureItem)):
                continue
            # 过滤 2：跳过纯标点符号或特殊不可见字符（常出现在 OCR 结果中）
            if re.match(r'^[ \t\n\r\f\v\s.，。、；;：:]+$', raw_text):
                continue

            page_no = self._get_page_num(item) or 1
            
            # --- 1. 安全提取坐标 ---
            bbox = None
            if isinstance(item, (TextItem, SectionHeaderItem, TableItem, PictureItem)):
                item_prov = getattr(item, "prov", [])
                if item_prov:
                    bbox = item_prov[0].bbox

            # --- 2. VLM 视觉研判层级 (针对文本) ---
            vlm_level = 0
            is_potential_header = isinstance(item, (SectionHeaderItem, TextItem)) and 2 <= len(raw_text) <= 80
            
            if is_potential_header and bbox and params.use_vlm:
                try:
                    crop_img = result.render.crop_bbox(bbox=bbox, page_no=page_no, padding=20)
                    prompt = (
                        f"文本: '{raw_text}'\n"
                        "请根据视觉特征（字号、加粗、位置）判断其结构层级：\n"
                        "1: 顶级标题; 2: 二级标题; 3: 三级标题; 0: 普通正文或列表项。\n"
                        "仅输出数字。"
                    )
                    _, vlm_res = await self._vlm_task(self.vlm_client, params.vlm_model, prompt, "level_check", crop_img)
                    vlm_level = int(re.sub(r'\D', '', vlm_res or "0"))
                except Exception as e:
                    logger.warning(f"标题视觉研判跳过: {e}")
                    vlm_level = 0

            # --- 3. 路径栈维护 ---
            if vlm_level > 0:
                flush_buffer()
                while len(active_path) >= vlm_level:
                    active_path.pop()
                active_path.append(raw_text)
                
                chunk_results.append({
                    "content": raw_text,
                    "path_names": list(active_path),
                    "structure_level": vlm_level,
                    "chunk_type": "heading",
                    "metadata": {"page_num": page_no, "is_vlm_verified": True}
                })
                continue

            # --- 4. 表格处理 ---
            if isinstance(item, TableItem):
                flush_buffer()
                vlm_table = None
                for ann in getattr(item, "annotations", []):
                    if isinstance(ann, DescriptionAnnotation) and getattr(ann, "provenance", "") == "vlm_table_rebuild":
                        vlm_table = getattr(ann, "text", None)
                        break
                
                # 如果是原生 Excel，直接用 export_to_markdown()，不要强求 VLM
                content = vlm_table if vlm_table else item.export_to_markdown()
                
                if content and content.strip():
                    chunk_results.append({
                        "content": content,
                        "path_names": list(active_path),
                        "structure_level": len(active_path) + 1,
                        "chunk_type": "table",
                        "metadata": {"page_num": self._get_page_num(item)}
                    })
                continue

            # --- 5. 新增：图片处理 (补全原有业务逻辑) ---
            if isinstance(item, PictureItem):
                flush_buffer()
                # 寻找预处理阶段生成的 VLM 图片描述
                img_description = None
                for ann in item.annotations:
                    if isinstance(ann, DescriptionAnnotation):
                        img_description = getattr(ann, "text", None)
                        break
                
                if img_description:
                    chunk_results.append({
                        "content": f"[图片描述]: {img_description}",
                        "path_names": list(active_path),
                        "structure_level": len(active_path) + 1,
                        "chunk_type": "image",
                        "metadata": {
                            "page_num": page_no, 
                            "node_path": getattr(item, "self_ref", ""),
                            "is_visual_desc": True
                        }
                    })
                continue

            # --- 6. 普通正文聚合 ---
            if raw_text:
                if not buffer_meta: buffer_meta = {"page_num": page_no}
                buffer_content.append(raw_text)
                if len("\n".join(buffer_content)) >= params.min_chunk_len:
                    flush_buffer()

        flush_buffer()
        return chunk_results

    # --- 3. 转换主流程 ---

    async def convert_document(
        self, 
        file_path: str, 
        params: DocParserParams, 
        output_format: OutputFormat = OutputFormat.MARKDOWN
    ) -> str | dict | list[dict]:
        
        # 配置 Pipeline
        pipeline_opts = PdfPipelineOptions(artifacts_path=self.local_artifacts_path)
        pipeline_opts.do_ocr = params.do_ocr
        pipeline_opts.do_table_structure = True
        pipeline_opts.generate_table_images = True # 必须开启以支持 VLM
        pipeline_opts.generate_picture_images = params.generate_picture_images
        
        pipeline_opts.table_structure_options = TableStructureOptions(do_cell_matching=True)
        
        engine = (params.ocr_engine or "easyocr").lower()
        if engine == "tesseract":
            pipeline_opts.ocr_options = TesseractOcrOptions(lang=["chi_sim", "eng"])
        else:
            pipeline_opts.ocr_options = EasyOcrOptions(lang=["ch_sim", "en"])

        # 执行转换
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
                InputFormat.DOCX: WordFormatOption(pipeline_options=pipeline_opts),
                InputFormat.XLSX: ExcelFormatOption(pipeline_options=pipeline_opts),
            }
        )
        
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, converter.convert, file_path)
        doc = result.document

        # VLM 视觉增强（表格重构 + 图片描述）
        await self._pre_enhance_with_vlm(doc, params)

        # 输出
        if output_format == OutputFormat.CHUNKS:
            return await self._generate_chunks(doc, result, params)

        return self._serialize(doc, output_format)
    
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