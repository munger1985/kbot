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
from utils.clients import AIModelClient
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
        # 开启并行
        loop = asyncio.get_running_loop()
        try:
            # 修正并行调用：只传递基础数据类型
            doc = await loop.run_in_executor(
                self.executor,
                _do_convert,
                str(file_path),
                self.artifacts_path,
                params.do_ocr,
                params.ocr_engine or "tesseract",
                params.image_scale
            )
        except Exception as e:
            logger.error(f"子进程转换异常: {file_path}, 详情: {e}")
            raise

        # 判断后续流程
        if output_format == OutputFormat.CHUNKS:
            # --- Stage 1: 调用VLM处理复杂表格和图片 ---
            file_ext = Path(file_path).suffix.lower()
            # 只有在参数开启了 VLM 且不是纯文本时才触发
            if params.use_vlm:
                await self._enhance_document_content(doc, params, file_ext, file_id)
            # 局部导入避免循环依赖
            from .chunk_generator import ChunkerGenerator
            chunker = ChunkerGenerator(params)
            vlm_data: dict = self._vlm_enhancement_cache.get(file_id, {})
            # logger.debug(f"向generate chunk方法传递的vlm描述：{vlm_data}")
            return await chunker.generate_chunks(doc, file_ext, vlm_data)

        # 其他格式序列化
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
    
    async def _enhance_document_content(self, doc: DoclingDocument, params: DocParserParams, file_ext: str, file_id: str) -> None:
        """
        视觉增强：复杂 Excel/PDF 表格截图调用 VLM 重构。
        """
        if not params.use_vlm or not params.vlm_model:
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
                    self.model_client, params.vlm_model, 
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
                    # 动态组装 Prompt
                    vlm_final_prompt = default_prompt.format(
                        params.img2txt_prompt, 
                        current_header=pic_context
                    )

                    tasks.append(self.model_task.vlm_task(
                        self.model_client, params.vlm_model, 
                        vlm_final_prompt, f"pic:hash:{img_hash}", raw_img
                    ))
                else:
                    hash_to_pic_indices[img_hash].append(i)
            
            logger.debug(f"构建完成 Hash 映射表，共 {len(hash_to_pic_indices)} 组唯一图片")

            # --- 3. 处理表格 (VLM) ---
            for i, table in enumerate(doc.tables):
                # 路径：有图（复杂 Excel 渲染或 PDF 表格）-> VLM 视觉重构
                if table.image and table.image.pil_image:
                    table_context = item_id_to_header.get(id(table), "未知章节")

                    table_prompt = f"""你是一个专业的文档解析专家。当前表格处于文本上下文：【{table_context}】中。
请将图片中的表格解析为 JSON 格式。
必须严格遵守以下约束：
1. 返回格式必须为：{{"header": "Markdown格式表头", "rows": ["数据行1", "数据行2", ...]}}
2. rows 数组中的每一项必须是单行 Markdown。
3. 严禁输出任何 JSON 以外的文字。"""
                
                    tasks.append(self.model_task.vlm_task(
                        self.model_client, params.vlm_model, 
                        table_prompt, f"table:index:{i}", table.image.pil_image
                    ))

            if not tasks:
                return
            
        # --- D. 并发执行与智能回填 ---
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, (BaseException, type(None))) or not isinstance(res, tuple):
                if isinstance(res, BaseException):
                    logger.error(f"VLM 任务执行崩溃: {res}")
                continue
                
            key, content = res 
            if not content: continue
                
            parts = key.split(':')
            if len(parts) < 3: continue

            category, sub_type, identifier = parts[0], parts[1], parts[2]

            # 1. 处理表格 (table:index:0)
            if category == "table":
                if identifier.isnumeric():
                    idx = int(identifier)
                    doc.tables[idx].annotations.append(
                        DescriptionAnnotation(text=content, provenance="vlm_table_rebuild")
                    )

            # 2. 处理图片 (pic:hash:abcd...)
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

            # 3. 处理标题/层级 (heading:index:0)
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

            # 4. 处理PPT slide
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