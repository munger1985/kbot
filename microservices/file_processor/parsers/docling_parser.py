import os
import re
import asyncio
from pathlib import Path
from enum import Enum
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from typing_extensions import override

# Docling 核心
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    EasyOcrOptions, 
    TesseractOcrOptions,
    TableStructureOptions
)
# Docling 文档转换格式选项
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption

# Docling Core 切分与序列化
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
    SectionHeaderItem  # 专门处理标题的类
)

# 项目依赖
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
    """自定义图片序列化器：输出 VLM 注入的描述。"""
    @override
    def serialize(self, *, item: PictureItem, doc: DoclingDocument, **kwargs: Any) -> tuple[SerializationResult, str]:
        text_parts = []
        
        # 获取图片保存目录（从外部传入）
        image_root = Path(kwargs.get("image_dir", "data/images"))
        image_root.mkdir(parents=True, exist_ok=True)

        # 1. 物理保存图片
        if item.image and item.image.pil_image:
            # 使用唯一 ID 命名，避免同名 PDF 图片覆盖
            image_name = f"pic_{item.self_ref.replace('/', '_')}.png"
            image_path = image_root / image_name
            item.image.pil_image.save(image_path)
            logger.debug(f"提取的图片保存成功: {image_path}")

        # 2. 注入 VLM 描述作为引用块
        for annotation in item.annotations:
            if isinstance(annotation, DescriptionAnnotation):
                # 这里放入引用块，方便后续 RAG 识别为背景补充信息
                text_parts.append(f"\n> [AI 视觉描述]: {annotation.text}\n")
        
        text_res = "\n".join(text_parts) if text_parts else ""
        return create_ser_result(text=text_res, span_source=item), image_name

class VLMEnabledMarkdownProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc: DoclingDocument) -> MarkdownDocSerializer:
        return MarkdownDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            picture_serializer=VLMAnnotationPictureSerializer(),
        )

class DoclingDocProcessor:
    def __init__(self, local_artifacts_path: str, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.vlm_semaphore = asyncio.Semaphore(5)
        self.local_artifacts_path = local_artifacts_path
        self.vlm_client = AIModelClient()
        # 通用标题栈：只保留最近几级标题
        self.title_hierarchy = []
        # 通用标题正则（适配所有常见标题：1. / 1 / 一、 / 第X章 / 1.1.1 等）
        self.title_re = re.compile(r'^(\d+(\.\d+)*|第[一二三四五六七八九十\d]+[章节条款项]|[一二三四五六七八九十]+)\s*[.、：]?\s*\S')
        self.max_hierarchy_depth = 3 # 标题层数限制，默认取3级标题
        self.corrector = SemanticLevelCorrector()
        self.serializer = VLMAnnotationPictureSerializer()

    def _is_title_item(self, item):
        text = getattr(item, "text", "").strip()
        if len(text) < 2 or len(text) > 200:
            return False
        if isinstance(item, SectionHeaderItem):
            return True
        if isinstance(item, TextItem):
            return bool(self.title_re.match(text))
        return False
    
    def _update_hierarchy(self, item):
        if not self._is_title_item(item):
            return
        text = getattr(item, "text", "").strip()

        # 提取标题的数字层级：1 → 1级；1.1 →2级；1.1.1→3级；第X章→1级；一、→1级
        # 通用规则：按"."分割层数
        dot_part = re.match(r'^(\d+(\.\d+)*)', text)
        if dot_part:
            level = len(dot_part.group(1).split('.'))
        else:
            level = 1  # 非数字标题统一当 1 级标题

        # 栈维护：比我级别高或相等的全部弹出，我入栈
        while self.title_hierarchy and self.title_hierarchy[-1]["level"] >= level:
            self.title_hierarchy.pop()
        self.title_hierarchy.append({"text": text, "level": level})

    def _get_current_hierarchy(self):
        # 返回当前完整标题链（只取文本）
        return [x["text"] for x in self.title_hierarchy[:self.max_hierarchy_depth]]
    
    def _get_page_num(self, item: Any) -> int | None:
        """
        增强版页码提取：适配PDF（真实页码）+ Word（逻辑位置）
        """
        # 尝试1：从prov属性提取PDF真实页码
        page_num = None
        item_prov = getattr(item, "prov", [])
        if item_prov:
            for prov in item_prov:
                if hasattr(prov, "page_no") and prov.page_no is not None:
                    page_num = prov.page_no
                    break
        
        # 尝试2：从item的page_reference提取（Docling 2.75部分版本的属性）
        if page_num is None and hasattr(item, "page_reference"):
            page_ref = getattr(item, "page_reference", {})
            page_num = page_ref.get("page_no") or page_ref.get("page_index")
        
        return page_num

    def _detect_is_zh(self, doc: DoclingDocument) -> bool:
        sample = "".join([t.text for t in doc.texts[:10]])
        if not sample: return False
        zh_chars = len(re.findall(r'[\u4e00-\u9fff]', sample))
        return zh_chars / (len(sample) + 1) > 0.1

    async def _process_vlm_descriptions(self, doc: DoclingDocument, params: DocParserParams) -> None:
        if not params.use_vlm or not params.vlm_model or not doc.pictures:
            return
        
        target_prompt = params.vlm_prompt or "请详细描述这张图片的内容，如果是图表请提取关键数据。"
        tasks = [self._vlm_task(self.vlm_client, params.vlm_model, target_prompt, i, pic.image.pil_image)
                 for i, pic in enumerate(doc.pictures) if pic.image and pic.image.pil_image]
        if tasks:
            results = await asyncio.gather(*tasks)
            for idx, desc in results:
                if desc:
                    doc.pictures[idx].annotations.append(DescriptionAnnotation(text=desc, provenance=f"vlm_{params.vlm_model}"))

    async def _vlm_task(self, client, model_name, prompt, index, image_obj) -> tuple:
        async with self.vlm_semaphore:
            try:
                res = await client.call_vlm_model(model_name=model_name, image=image_obj, prompt=prompt)
                return index, res
            except Exception as e:
                logger.error(f"VLM 失败: {e}"); return index, None
    
    def _generate_chunks(self, doc: DoclingDocument, params: DocParserParams) -> list[dict]:
        """
        优化后的分块逻辑：
        1. 引入 SemanticLevelCorrector 修正手写层级
        2. 维护 Active Path Stack 记录当前内容的完整标题路径
        3. 输出适配 ES 分层检索的格式
        """
        chunk_results = []
        current_chunk_num = 1
        corrector = SemanticLevelCorrector()
        
        # 活跃路径栈，存储当前层级的标题文字
        # ES 中通过此数组实现"路径过滤"和"上下文补全"
        active_path = [] 

        for item, depth in doc.iterate_items():
            text_content = getattr(item, "text", "").strip()
            image_name = None
            # --- 1. 动态层级维护 ---
            detected_level = corrector.get_level(item, text_content)
            
            if detected_level is not None:
                # 遇到新标题：弹出同级或更深层级的旧路径，压入新路径
                while len(active_path) >= detected_level:
                    active_path.pop()
                active_path.append(text_content)
                current_type = "heading"
                structure_level = detected_level
            else:
                current_type = "text"
                structure_level = len(active_path) + 1

            # --- 2. 内容提取 ---
            content_list = []
            if isinstance(item, TableItem):
                current_type = "table"
                content_list.append(item.export_to_markdown(doc=doc))

            elif isinstance(item, PictureItem):
                current_type = "picture"
                logger.debug(f"image_dir: {params.image_dir}")
                # 调用自定义的序列化器来获取"正确的 Markdown 内容"并执行"物理保存"
                # 注意：这里需要传入 doc 对象，因为它在 serialize 签名里
                ser_result, image_name = self.serializer.serialize(item=item, doc=doc, image_dir=params.image_dir)
                content_list.append(ser_result.text) # 这将包含 ![pic_...](path) 和 AI 描述

            elif isinstance(item, (TextItem, SectionHeaderItem)):
                # 文本切分
                if len(text_content) > params.chunk_size:
                    sentences = re.split(r'(?<=[。！？；!?;])\s*', text_content)
                    curr = ""
                    for s in sentences:
                        if len(curr) + len(s) <= params.chunk_size: curr += s
                        else:
                            if curr: content_list.append(curr.strip())
                            curr = s
                    if curr: content_list.append(curr.strip())
                else:
                    content_list.append(text_content)

            # --- 3. 封装 ES Chunk (升维存储) ---
            page_num = self._get_page_num(item)
            
            for sub_idx, sub_content in enumerate(content_list, start=1):
                if not sub_content or len(sub_content) < params.min_chunk_len:
                    continue

                chunk_results.append({
                    "content": sub_content,
                    # 全路径数组，直接进 ES 做混合检索
                    "path_names": list(active_path), 
                    "structure_level": structure_level,
                    "chunk_type": current_type,
                    "metadata": {
                        "chunk_num": current_chunk_num,
                        "sub_index": sub_idx,
                        "page_num": page_num,
                        "node_path": item.self_ref, # 保留 Docling 原始引用
                        "image_name": image_name
                    }
                })
                current_chunk_num += 1

        return chunk_results
    
    async def convert_document(self, file_path: str, params: DocParserParams, output_format: OutputFormat = OutputFormat.MARKDOWN) -> str | dict | list[dict]:
        """
        核心解析逻辑：处理 PDF/Word -> DoclingDocument -> 注入 VLM
        """
        # 1. 配置初始化
        pipeline_opts = PdfPipelineOptions(artifacts_path=self.local_artifacts_path)
        pipeline_opts.do_ocr = params.do_ocr
        pipeline_opts.generate_picture_images = params.generate_picture_images
        pipeline_opts.images_scale = params.image_scale
        
        # 表格配置
        pipeline_opts.table_structure_options = TableStructureOptions(do_cell_matching=True)
        pipeline_opts.do_table_structure = True

        # OCR 引擎配置
        engine = (params.ocr_engine or "easyocr").lower()
        if engine == "tesseract":
            pipeline_opts.ocr_options = TesseractOcrOptions(lang=["chi_sim", "eng"])
        else:
            pipeline_opts.ocr_options = EasyOcrOptions(lang=["ch_sim", "en"])

        # 2. 转换器初始化
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
                InputFormat.DOCX: WordFormatOption(pipeline_options=pipeline_opts)
            }
        )
        
        # 3. 执行转换
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, converter.convert, file_path)
        doc = result.document

        # 4. VLM 注入
        await self._process_vlm_descriptions(doc, params)

        # 5. 根据期望返回的格式，生成 chunk 或者其他输出格式
        if output_format == OutputFormat.CHUNKS:
            return self._generate_chunks(doc, params)

        return self._serialize(doc, output_format)

    def _serialize(self, doc: DoclingDocument, fmt: OutputFormat) -> str | dict:
        if fmt == OutputFormat.MARKDOWN:
            return VLMEnabledMarkdownProvider().get_serializer(doc).serialize().text
        if fmt == OutputFormat.HTML: return doc.export_to_html()
        if fmt == OutputFormat.JSON: return doc.export_to_dict()
        if fmt == OutputFormat.DOCTAGS: return doc.export_to_doctags()
        return ""