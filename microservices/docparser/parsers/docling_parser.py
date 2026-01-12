import os
import re
import asyncio
from enum import Enum
from typing import Any, List, Dict, Union, Optional
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from pydantic import BaseModel, Field, model_validator
from typing_extensions import override
from transformers import AutoTokenizer

# Docling 核心
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    EasyOcrOptions, 
    TesseractOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# Docling Core 切分与序列化
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
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
)

# 项目依赖
from utils.model_client import CallModel
from ..parser_schema import ParserParams

class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    DOCTAGS = "doctags"
    CHUNKS = "chunks"



class VLMAnnotationPictureSerializer(MarkdownPictureSerializer):
    """自定义图片序列化器：输出 VLM 注入的描述。"""
    @override
    def serialize(self, *, item: PictureItem, doc: DoclingDocument, **kwargs: Any) -> SerializationResult:
        text_parts = []
        for annotation in item.annotations:
            if isinstance(annotation, DescriptionAnnotation):
                text_parts.append(f"\n> [图片内容描述: {annotation.text}]\n")
        
        text_res = "\n".join(text_parts) if text_parts else ""
        return create_ser_result(text=text_res, span_source=item)

class VLMEnabledMarkdownProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc: DoclingDocument) -> MarkdownDocSerializer:
        return MarkdownDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            picture_serializer=VLMAnnotationPictureSerializer(),
        )

class DoclingDocProcessor:
    def __init__(self, en_tokenizer_path: str, zh_tokenizer_path: str):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.vlm_semaphore = asyncio.Semaphore(5)
        self.en_tk = AutoTokenizer.from_pretrained(en_tokenizer_path)
        self.zh_tk = AutoTokenizer.from_pretrained(zh_tokenizer_path)

    def _detect_is_zh(self, doc: DoclingDocument) -> bool:
        sample = "".join([t.text for t in doc.texts[:10]])
        if not sample: return False
        zh_chars = len(re.findall(r'[\u4e00-\u9fff]', sample))
        return zh_chars / (len(sample) + 1) > 0.1

    async def _process_vlm_descriptions(self, doc: DoclingDocument, params: ParserParams) -> None:
        if not params.use_vlm or not params.vlm_model or not doc.pictures:
            return
        vlm_client = CallModel()
        target_prompt = params.vlm_prompt or "请详细描述这张图片的内容，如果是图表请提取关键数据。"
        tasks = [self._vlm_task(vlm_client, params.vlm_model, target_prompt, i, pic.image.pil_image)
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

    def _generate_chunks(self, doc: DoclingDocument, params: ParserParams) -> list[dict]:
        is_zh = self._detect_is_zh(doc)
        tk = self.zh_tk if is_zh else self.en_tk
        
        chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer(tokenizer=tk, max_tokens=params.chunk_size),
            serializer_provider=VLMEnabledMarkdownProvider()
        )
        
        chunk_results = []
        for chunk in chunker.chunk(doc):
            content = chunker.contextualize(chunk).strip()
            if len(content) < params.min_chunk_len:
                continue

            # 修复 Pylance 警告：使用 .items 访问元数据关联项
            doc_items = [item_ref.item for item_ref in getattr(chunk.meta, "items", [])]
            
            # 类型判定
            pics = [item for item in doc_items if isinstance(item, PictureItem)]
            has_vlm = any(isinstance(ann, DescriptionAnnotation) for p in pics for ann in p.annotations)
            
            # 逻辑控制：G=True, VLM=False 且是纯图片块时跳过
            if pics and not has_vlm and params.generate_picture_images:
                if all(isinstance(item, PictureItem) for item in doc_items):
                    continue
            
            # 页码提取
            page_num = next((item.prov[0].page_no for item in doc_items if item.prov), None)

            chunk_results.append({
                "text": content,
                "metadata": {
                    "chunk_type": "picture" if pics else "text",
                    "page_num": page_num,
                    "has_vlm": has_vlm
                }
            })
        return chunk_results

    async def convert_document(self, params: ParserParams) -> Union[str, Dict, List[Dict]]:
        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.do_ocr = params.do_ocr
        pipeline_opts.generate_picture_images = params.generate_picture_images
        pipeline_opts.images_scale = params.images_scale
        
        engine = (params.ocr_engine or "easyocr").lower()
        pipeline_opts.ocr_options = TesseractOcrOptions() if engine == "tesseract" else EasyOcrOptions()
        pipeline_opts.ocr_options.lang = ["ch_sim", "en"]

        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)})
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, converter.convert, params.file_path)
        doc = result.document

        await self._process_vlm_descriptions(doc, params)

        if params.output_format == OutputFormat.CHUNKS:
            return self._generate_chunks(doc, params)
        
        return self._serialize(doc, OutputFormat(params.output_format))

    def _serialize(self, doc: DoclingDocument, fmt: OutputFormat) -> str | dict:
        if fmt == OutputFormat.MARKDOWN:
            return VLMEnabledMarkdownProvider().get_serializer(doc).serialize().text
        if fmt == OutputFormat.HTML: return doc.export_to_html()
        if fmt == OutputFormat.JSON: return doc.export_to_dict()
        if fmt == OutputFormat.DOCTAGS: return doc.export_to_doctags()
        return ""