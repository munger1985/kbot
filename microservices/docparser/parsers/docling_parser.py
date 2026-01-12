"""Docling 文档处理模块。

本模块提供基于 Docling 的文档解析能力，集成 VLM 进行图片描述增强，
支持自定义 VLM Prompt，并提供灵活的切分与格式导出功能。
"""

import os
import re
import asyncio
import tempfile
from enum import Enum
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from typing_extensions import override
from transformers import AutoTokenizer

# Docling 核心
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
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
    """支持的输出格式枚举。"""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    DOCTAGS = "doctags"
    CHUNKS = "chunks"


class VLMAnnotationPictureSerializer(MarkdownPictureSerializer):
    """自定义图片序列化器：输出 VLM 注入的描述。"""
    @override
    def serialize(self, *, item: PictureItem, doc_serializer: Any, doc: DoclingDocument, **kwargs: Any) -> SerializationResult:
        text_parts: list[str] = []
        for annotation in item.annotations:
            if isinstance(annotation, DescriptionAnnotation):
                text_parts.append(f"\n> [图片内容描述: {annotation.text}]\n")
        
        text_res = "\n".join(text_parts) if text_parts else ""
        return create_ser_result(text=text_res, span_source=item)


class VLMEnabledMarkdownProvider(ChunkingSerializerProvider):
    """Markdown 序列化提供者。"""
    def get_serializer(self, doc: DoclingDocument) -> MarkdownDocSerializer:
        return MarkdownDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            picture_serializer=VLMAnnotationPictureSerializer(),
        )


class DoclingDocProcessor:
    """Docling 文档处理器。"""

    def __init__(self, en_tokenizer_path: str, zh_tokenizer_path: str):
        """初始化处理器。

        Args:
            en_tokenizer_path: 英文 Tokenizer 模型路径。
            zh_tokenizer_path: 中文 Tokenizer 模型路径。
        """
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.vlm_semaphore = asyncio.Semaphore(5)
        self.en_tk = AutoTokenizer.from_pretrained(en_tokenizer_path)
        self.zh_tk = AutoTokenizer.from_pretrained(zh_tokenizer_path)

    def _detect_is_zh(self, doc: DoclingDocument) -> bool:
        """检测文档是否主要为中文。"""
        sample = "".join([t.text for t in doc.texts[:10]])
        if not sample: return False
        zh_chars = len(re.findall(r'[\u4e00-\u9fff]', sample))
        return zh_chars / (len(sample) + 1) > 0.1

    async def _process_vlm_descriptions(
        self, 
        doc: DoclingDocument, 
        vlm_model_id: int | None,
        vlm_prompt: str | None = None
    ) -> None:
        """遍历图片并调用 VLM，使用内存对象直传图片，支持自定义 Prompt。

        Args:
            doc: 文档对象。
            vlm_model_id: 模型ID。
            vlm_prompt: 发送给 VLM 的提示词。
        """
        if not vlm_model_id or not doc.pictures:
            return
            
        vlm_client = CallModel()
        # 默认 Prompt（如果用户未提供）
        default_prompt = "请详细描述这张图片的内容，如果是图表请提取关键数据。"
        target_prompt = vlm_prompt or default_prompt

        tasks = []
        for i, pic in enumerate(doc.pictures):
            # 直接检查 PIL 对象是否存在
            if not pic.image or not pic.image.pil_image:
                continue
            
            # 直接传递 pic.image.pil_image (内存对象)
            tasks.append(
                self._vlm_task(
                    client=vlm_client, 
                    model_id=vlm_model_id, 
                    prompt=target_prompt, 
                    index=i, 
                    image_obj=pic.image.pil_image  # 传入对象而非路径
                )
            )
        
        if tasks:
            results = await asyncio.gather(*tasks)
            for idx, desc in results:
                if desc:
                    doc.pictures[idx].annotations.append(
                        DescriptionAnnotation(
                            text=desc, 
                            provenance=f"vlm_{vlm_model_id}"
                        )
                    )

    async def _vlm_task(
        self, 
        client: CallModel, 
        model_id: int, 
        prompt: str, 
        index: int, 
        image_obj: Any
    ) -> tuple[int, str | None]:
        """执行单个 VLM 推理任务（内存直传）。"""
        async with self.vlm_semaphore:
            try:
                # client 的 call_vlm_model_for_parsing_picture 已经支持 PIL 对象
                res = await client.call_vlm_model(
                    model_id=model_id, 
                    image=image_obj,  # 直接传 PIL 对象
                    prompt=prompt
                )
                return index, res
            except Exception as e:
                logger.error(f"VLM 推理失败 (Index {index}): {e}")
                return index, None

    async def convert_document(
        self, 
        file_path: str, 
        params: ParserParams,
        vlm_model: int | None = None,
        vlm_prompt: str | None = None,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
    ) -> str | dict | list[str]:
        """转换文档。

        Args:
            file_path: 文件路径。
            params: 解析参数。
            vlm_model: VLM 模型ID。
            vlm_prompt: VLM 提示词。
            output_format: 输出格式。
        """
        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.do_ocr = params.do_ocr
        pipeline_opts.generate_picture_images = params.generate_picture_images
        
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
        )
        
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, converter.convert, file_path)
        doc = result.document

        # 注入自定义 Prompt 处理
        await self._process_vlm_descriptions(doc, vlm_model, vlm_prompt)

        if output_format == OutputFormat.CHUNKS:
            return self._generate_chunks(doc, params)
        return self._serialize(doc, output_format)

    def _generate_chunks(self, doc: DoclingDocument, params: ParserParams) -> list[str]:
        """生成文档分块。"""
        is_zh = self._detect_is_zh(doc)
        tk = self.zh_tk if is_zh else self.en_tk
        
        chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer(tokenizer=tk, max_tokens=params.chunk_size),
            serializer_provider=VLMEnabledMarkdownProvider()
        )
        
        return [chunker.contextualize(c).strip() for c in chunker.chunk(doc) 
                if len(chunker.contextualize(c).strip()) >= params.min_chunk_len]

    def _serialize(self, doc: DoclingDocument, fmt: OutputFormat) -> str | dict:
        """序列化输出。"""
        match fmt:
            case OutputFormat.MARKDOWN:
                provider = VLMEnabledMarkdownProvider()
                return provider.get_serializer(doc).serialize().text 
            case OutputFormat.HTML:
                return doc.export_to_html()
            case OutputFormat.JSON:
                return doc.export_to_dict()
            case OutputFormat.DOCTAGS:
                return doc.export_to_doctags()
            case _:
                logger.warning(f"未知的输出格式请求: {fmt}")
                return ""