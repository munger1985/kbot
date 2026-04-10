import json
import re
import asyncio
import imagehash
from pathlib import Path
from PIL import Image
from enum import Enum
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from typing_extensions import override

# Docling 核心模块导入
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    PipelineOptions,
    EasyOcrOptions, 
    TesseractOcrOptions,
    TableStructureOptions
)
# Docling 文档转换格式选项
from docling.document_converter import(
    DocumentConverter, 
    PdfFormatOption, 
    WordFormatOption,
    ExcelFormatOption,
    PowerpointFormatOption,
    HTMLFormatOption,
    MarkdownFormatOption
) 

# Docling Core 分块与序列化模块
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
    SectionHeaderItem  # 用于标题处理的专用类
)

# 项目依赖
from utils.clients import AIModelClient
from ..parser_schema import DocParserParams


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

class DoclingDocProcessor:
    """基于Docling框架的文档处理类，增强功能包括：
    - VLM驱动的图片描述生成
    - 语义感知的标题检测
    - 适用于RAG/ES的层级分块
    - 多格式输出支持（Markdown/HTML/JSON/Chunks）
    
    Attributes:
        executor: 线程池执行器，用于同步任务的异步执行
        vlm_semaphore: VLM API调用的限流信号量
        local_artifacts_path: 本地工件存储路径
        vlm_client: AI模型客户端，用于调用VLM服务
        title_hierarchy: 标题层级栈，维护当前文档位置的标题路径
        title_re: 标题匹配正则表达式，支持常见标题格式
        max_hierarchy_depth: 标题层级最大深度限制
        corrector: 语义层级校正器，用于标题层级的智能判断
        serializer: VLM图片序列化器实例
    """
    def __init__(self, local_artifacts_path: str, max_workers: int = 4):
        """初始化文档处理器
        
        Args:
            local_artifacts_path: 本地工件存储路径，用于保存处理过程中的临时文件
            max_workers: 线程池最大工作线程数，默认4
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.vlm_semaphore = asyncio.Semaphore(2)  # VLM API调用限流（最多2个并发）
        self.llm_semaphore = asyncio.Semaphore(5)  # LLM API调用限流（最多5个并发）
        self.local_artifacts_path = local_artifacts_path
        self.model_client = AIModelClient()
        # 通用标题栈：仅保留最近的层级
        self.serializer = VLMAnnotationPictureSerializer()
        # [新增] VLM 描述缓存，Key 为图片指纹(hash)，Value 为描述文本
        self._vlm_cache = {}

    async def _vlm_task(self, client, model_name, prompt, index, image_obj) -> tuple:
        async with self.vlm_semaphore:
            try:
                if image_obj:
                    # --- 智能压缩逻辑开始 ---
                    # 1. 设定最大长边限制（QwenVL 建议在 1000 左右平衡度最好）
                    max_size = 1024 
                    w, h = image_obj.size
                    
                    if max(w, h) > max_size:
                        scale = max_size / max(w, h)
                        new_size = (int(w * scale), int(h * scale))
                        # 使用 LANCZOS 算法保证缩放后的边缘平滑，利于文字识别
                        image_obj = image_obj.resize(new_size, Image.Resampling.LANCZOS)
                        logger.debug(f"VLM 图片缩放: {w}x{h} -> {new_size}")

                    # 2. 如果是识别标题层级，可以转为 RGB 减少调色板干扰（Docling 有时返回 P 模式）
                    if image_obj.mode != "RGB":
                        image_obj = image_obj.convert("RGB")
                    # --- 智能压缩逻辑结束 ---

                res = await client.call_vlm_model(model_name=model_name, image=image_obj, prompt=prompt)
                return index, res
            except Exception as e:
                logger.error(f"VLM处理失败 (Index {index}): {e}")
                return index, None
            
    async def _llm_task(self, client: AIModelClient, model_name: str, prompt: str, index: str) -> tuple:
        """抽象出的 LLM 任务，用于处理纯文本逻辑（如标题层级、表格修复）"""
        try:
            async with self.llm_semaphore:
                raw_res = await client.get_llm_answer(
                    model_name=model_name, 
                    prompt=prompt,
                    stream=True  # 即使是内部聚合，开启 stream 也能更早释放资源
                )
                
                if "level" in str(index).lower():
                    # 仅匹配字符串中出现的第一个数字字符
                    # 防止 LLM 返回 "1." 或 "层级为 1" 导致的解析失败
                    match = re.search(r'\d', str(raw_res or "0"))
                    if match:
                        val = int(match.group())
                        # 约束：层级只允许在 0-7 之间，超过则视为无效
                        return index, (val if 0 <= val <= 7 else 0)
                    return index, 0

                # 第三步：非层级任务（如表格修复）保持原样返回字符串
                return index, (raw_res if raw_res else None)

        except Exception as e:
            logger.error(f"LLM 任务执行失败 (Index {index}): {str(e)}")
            return index, (0 if "level" in str(index).lower() else None)
            
    async def _enhance_document_content(self, doc: DoclingDocument, params: DocParserParams) -> None:
        """
        NexusCube 增强引擎：
        1. 视觉增强：复杂 Excel/PDF 表格截图调用 VLM 重构。
        2. 结构增强：利用 LLM 判定 TextItem 的真实标题层级（替换不稳定的正则）。
        """
        if not params.use_vlm or not params.vlm_model:
            return
        
        tasks = []
        # [新增] 哈希映射表，用于存储 hash -> [item_indices] 的关系
        hash_to_pic_indices = {}

        # --- A. 结构增强：文档标题层级判定 (LLM) ---
        # 遍历所有文本项，寻找可能是标题的项进行语义判定
        for i, item in enumerate(doc.texts):
            # 过滤逻辑：长度适中（避免正文误判），或者 Docling 初步认为它是标题的
            text_content = item.text.strip()
            if 0 < len(text_content) < 200: 
                # 针对 Word/PPT/PDF，判定其在文档大纲中的逻辑层级
                prompt = f"""分析以下文本在文档中的大纲层级：
文本内容："{text_content}"
要求：若是顶级标题返1，二级标题返2，三级标题返3，若只是普通正文或无关信息返0。只返回数字，不要任何解释。"""
                tasks.append(self._llm_task(
                    self.model_client, params.llm_model, prompt, f"heading_llm_{i}"
                ))

        # --- B. 视觉增强：处理图片与复杂表格 (VLM) ---
        for i, pic in enumerate(doc.pictures):
            if pic.image and pic.image.pil_image:
                img = pic.image.pil_image
            
            # 1. 物理过滤：太小的图（如 60x60 以下）通常是装饰性图标，直接标记并跳过 VLM
            if img.width < 60 or img.height < 60:
                pic.annotations.append(DescriptionAnnotation(text="装饰性图标/Logo", provenance="size_filter"))
                continue

            # 2. 计算指纹
            img_hash = str(imagehash.dhash(img))
            
            # 3. 检查全局或内存缓存
            if img_hash in self._vlm_cache:
                pic.annotations.append(DescriptionAnnotation(text=self._vlm_cache[img_hash], provenance="vlm_cache_hit"))
                continue
            
            # 4. 任务合并：同一个文档内相同的图只跑一次
            if img_hash not in hash_to_pic_indices:
                hash_to_pic_indices[img_hash] = []
                tasks.append(self._vlm_task(
                    self.model_client, params.vlm_model, 
                    params.vlm_prompt or "描述图片内容", f"pic_hash_{img_hash}", img
                ))
            hash_to_pic_indices[img_hash].append(i)

        # --- C. 处理表格 (VLM) ---
        for i, table in enumerate(doc.tables):
            # 路径：有图（复杂 Excel 渲染或 PDF 表格）-> VLM 视觉重构
            if table.image and table.image.pil_image:
                prompt = """你是一个专业的文档解析专家。请将图片中的表格解析为 JSON 格式。
必须严格遵守以下约束：

返回格式必须为：{"header": "Markdown格式表头", "rows": ["数据行1", "数据行2", ...]}

rows 数组中的每一项必须是单行 Markdown。

严禁输出任何 JSON 以外的文字。

如果表格跨行，请完整保留每一行。"""
                tasks.append(self._vlm_task(
                    self.model_client, params.vlm_model, prompt, f"table_vlm_{i}", table.image.pil_image
                ))

        if not tasks:
            return
        
        # --- D. 并发执行与智能回填 ---
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, (BaseException, type(None))) or not isinstance(res, tuple):
                continue
                
            key, content = res 
            if not content: continue
                
            parts = key.split('_')
            category, engine, idx = parts[0], parts[1], int(parts[2])

            # 1. 回填层级信息
            if category == "heading":
                target_item = doc.texts[idx]
                # 使用 getattr 获取列表，如果不存在则返回 None
                annos = getattr(target_item, "annotations", None)
                
                if isinstance(annos, list):
                    try:
                        level_val = int(re.sub(r'\D', '', str(content)))
                        annos.append(
                            DescriptionAnnotation(
                                text=str(level_val), 
                                provenance="llm_structure_level"
                            )
                        )
                    except: pass

            # 2. 回填表格描述/重构
            elif category == "table":
                doc.tables[idx].annotations.append(
                    DescriptionAnnotation(text=content, provenance="vlm_table_rebuild")
                )

            # 3. 回填图片描述
            elif category == "pic":
                if engine == "hash":
                    img_hash = idx
                    self._vlm_cache[img_hash] = content # 存入缓存
                    # 将结果回填给所有拥有此 hash 的图片
                    for p_idx in hash_to_pic_indices.get(img_hash, []):
                        doc.pictures[p_idx].annotations.append(
                            DescriptionAnnotation(text=content, provenance=f"vlm_deduplicated_{params.vlm_model}")
                        )
    
    def _get_page_num(self, item: Any) -> int:
        """获取项所在的页码，增强对 PDF 结构的兼容性"""
        try:
            # 1. 尝试标准 Docling 路径：item -> prov (Provenance) -> page_no
            # 很多时候 page_no 存在于 item.prov[0].page_no 中
            if hasattr(item, "prov") and item.prov:
                # 取第一个来源证明中的页码
                return getattr(item.prov[0], "page_no", 1)
            
            # 2. 尝试旧版或特定格式的 page_reference 对象
            page_ref = getattr(item, "page_reference", None)
            if page_ref:
                # 如果是字典
                if isinstance(page_ref, dict):
                    return page_ref.get("page_no", 1)
                # 如果是 Pydantic 对象
                return getattr(page_ref, "page_no", 1)
                
        except Exception as e:
            logger.debug(f"页码获取失败，回退至默认值 1: {e}")
            
        return 1
    
    async def _process_table_vlm(self, item: TableItem, doc: DoclingDocument, params: DocParserParams, active_path: list, header_context: str) -> list[dict]:
        """专门处理表格的分块逻辑：VLM解析 + JSON切片 + 物理硬切防溢出"""
        table_chunks = []
        TABLE_ROW_STEP = 40 
        MAX_CHAR_LIMIT = 15000
        current_header = ""
        
        # 提取 VLM 标注
        vlm_res = next((ann.text for ann in getattr(item, "annotations", []) 
                    if getattr(ann, "provenance", "") == "vlm_table_rebuild"), None)

        # 准备待处理文本列表
        parts_to_verify = []

        if vlm_res:
            try:
                # 尝试 JSON 解析并按行切分
                clean_json = re.sub(r'```json\s*|\s*```', '', vlm_res).strip()
                table_data = json.loads(clean_json)
                header = table_data.get("header", "").strip()
                current_header = header 
                rows = table_data.get("rows", [])
                
                if rows:
                    current_rows, current_chars = [], len(header)
                    for row in rows:
                        row_str = str(row)
                        if (len(current_rows) >= TABLE_ROW_STEP) or (current_chars + len(row_str) > MAX_CHAR_LIMIT):
                            if current_rows:
                                parts_to_verify.append(f"{header}\n" + "\n".join(current_rows))
                            current_rows, current_chars = [row_str], len(header) + len(row_str)
                        else:
                            current_rows.append(row_str)
                            current_chars += len(row_str)
                    if current_rows:
                        parts_to_verify.append(f"{header}\n" + "\n".join(current_rows))
                else:
                    parts_to_verify.append(vlm_res)
            except:
                parts_to_verify.append(vlm_res)
        else:
            # 无 VLM 场景使用默认导出
            parts_to_verify.append(item.export_to_markdown(doc=doc))

        # 二次物理防线：处理单块依然超长的情况
        for part in parts_to_verify:
            if len(part) > MAX_CHAR_LIMIT:
                lines = part.split('\n')
                final_h = current_header if current_header else "\n".join(lines[:2])
                start_l = 0 if current_header else 2
                for i in range(start_l, len(lines), TABLE_ROW_STEP):
                    sub = "\n".join(lines[i : i + TABLE_ROW_STEP])
                    if sub.strip():
                        table_chunks.append({"content": f"{final_h}\n{sub}", "type": "table"})
            else:
                table_chunks.append({"content": part, "type": "table"})

        # 映射为标准业务输出格式
        return [{
            "content": tc["content"],
            "path_names": list(active_path),
            "structure_level": len(active_path) + 1,
            "chunk_type": "table",
            "metadata": {
                "chunk_num": 0, # 在主循环中统一编号
                "page_num": self._get_page_num(item),
                "header_context": header_context,
                "image_name": None
            }
        } for tc in table_chunks]

    async def _generate_chunks(self, doc: DoclingDocument, result, params: DocParserParams) -> list[dict]:
        """
        分块逻辑：
        1. 使用vlm识别复杂Excel表格，并且如果表格超长，按照参数设定的行数进行切分，子表格保留相同表头。
        2. 针对 Word/PPT 自动使用纯文本 LLM 判定。
        """
        chunk_results = []
        current_chunk_num = 1
        active_path = []
        seen_visual_contents = set()
        # 分块参数
        MIN_CHUNK_LEN = params.min_chunk_len or 200 # 最小合并字符数，低于此值会尝试与下一段合并
        MAX_CHUNK_LEN = params.chunk_size or 600 # 达到此长度强制刷出

        # --- Stage 1: 语义聚合 (Pass 1) ---
        # 目标：将 Docling 的碎 Item 聚合成语义块，消除目录、页码、断开的标题
        semantic_units = []
        staging_prefix = ""

        for item, _ in doc.iterate_items():
            raw_text = getattr(item, "text", "").strip()
            # 1. 物理过滤：跳过纯目录虚线项 (例如: 内容 .......... 29)
            if re.search(r'[\.\s…]{5,}\s*\d+$', raw_text):
                continue
            
            # 2. 媒体项处理
            if isinstance(item, (TableItem, PictureItem)):
                staging_prefix = ""
                semantic_units.append({"type": "media", "item": item})
                continue

            if not raw_text: continue

            # 3. 碎片前缀识别：识别可能是标题编号的短文本 (如 "5", "3.1.2", "第一章")
            # 如果长度很短且符合编号特征，先攒着
            is_short_num = len(raw_text) < 10 and (raw_text.isdigit() or re.match(r'^[\d\.、]+$', raw_text) or raw_text.startswith("第"))
            
            if is_short_num:
                staging_prefix = raw_text
                continue

            # 4. 语义缝合
            if staging_prefix:
                # 将前缀与当前内容合并，例如 "5" + "可靠性分析" -> "5 可靠性分析"
                combined_text = f"{staging_prefix} {raw_text}"
                staging_prefix = ""
            else:
                combined_text = raw_text
            
            # 5. 判定聚合后的文本身份
            is_header = isinstance(item, SectionHeaderItem) or (len(combined_text) < 60 and re.match(r'^[一二三四\d].*?[\s\.]', combined_text))

            if is_header:
                semantic_units.append({"type": "header", "text": combined_text, "item": item})
            else:
                # 正文合并：如果上一个也是 text，就粘在一起，减少 chunk 数量
                if semantic_units and semantic_units[-1]["type"] == "text":
                    semantic_units[-1]["text"] += f"\n{combined_text}"
                else:
                    semantic_units.append({"type": "text", "text": combined_text, "item": item})

        # --- Stage 2: 逻辑分块输出 ---
        text_buffer = []
        buffer_len = 0
        pending_header_context = ""

        def flush(item_ref, c_type="text", custom_content=None, img_name=None):
            nonlocal current_chunk_num, text_buffer, buffer_len
            content = custom_content if custom_content else "\n".join(text_buffer).strip()
            if not content: return
            
            chunk_results.append({
                "content": content,
                "path_names": list(active_path),
                "structure_level": len(active_path),
                "chunk_type": c_type,
                "metadata": {
                    "chunk_num": current_chunk_num,
                    "page_num": self._get_page_num(item_ref),
                    "header_context": pending_header_context,
                    "image_name": img_name
                }
            })
            current_chunk_num += 1
            if not custom_content:
                text_buffer, buffer_len = [], 0

        for unit in semantic_units:
            if unit["type"] == "header":
                # 只有内容够多才切片，否则只更新路径并合并
                if buffer_len > MIN_CHUNK_LEN:
                    flush(unit["item"])
                
                # 更新 path_names (简化逻辑：只保留最近的 3 级标题)
                header_text = unit["text"]
                if len(active_path) >= 3: active_path.pop(0)
                active_path.append(header_text)
                pending_header_context = header_text
                
                text_buffer.append(f"{header_text}")
                buffer_len += len(header_text)

            elif unit["type"] == "text":
                text_buffer.append(unit["text"])
                buffer_len += len(unit["text"])
                if buffer_len > MAX_CHUNK_LEN:
                    flush(unit["item"])

            elif unit["type"] == "media":
                # 媒体必须先清空前面的文字 buffer
                flush(unit["item"])
                item = unit["item"]
                
                if isinstance(item, TableItem):
                    # 引用之前讨论的严谨 Excel VLM 处理逻辑
                    table_chunks = await self._process_table_vlm(item, doc, params, active_path, pending_header_context)
                    for t_chunk in table_chunks:
                        t_chunk["metadata"]["chunk_num"] = current_chunk_num
                        chunk_results.append(t_chunk)
                        current_chunk_num += 1
                
                elif isinstance(item, PictureItem):
                    vlm_text = next((ann.text for ann in getattr(item, "annotations", []) 
                                if "vlm" in getattr(ann, "provenance", "")), None)
                    # 如果该图片有 VLM 描述，且该描述已经输出过，则直接跳过，不生成 Chunk
                    if vlm_text:
                        if vlm_text in seen_visual_contents:
                            logger.debug(f"跳过重复图片 Chunk: {vlm_text[:20]}...")
                            continue
                        seen_visual_contents.add(vlm_text)
                    # 序列化并输出
                    ser_result, img_name = self.serializer.serialize(item=item, doc=doc, image_dir=params.image_dir)
                    # flush(item, c_type="picture", custom_content=ser_result.text, img_name=img_name)
                    # 如果序列化结果为空（例如被标记为装饰性图标且没描述），则不生成 Chunk
                    if ser_result.text.strip():
                        flush(item, c_type="picture", custom_content=ser_result.text, img_name=img_name)

        # 循环结束收尾
        if text_buffer:
            flush(semantic_units[-1]["item"] if semantic_units else None)

        return chunk_results
    
    async def convert_document(self, file_path: str, params: DocParserParams, output_format: OutputFormat = OutputFormat.MARKDOWN):
        # 1. 显式实例化 TableStructureOptions (Docling 2.7.x 必须整体赋值)
        # do_cell_matching=True 用于精确对齐表格单元格，对后续 VLM 处理至关重要
        table_opts = TableStructureOptions(do_cell_matching=True)

        # 2. 统一使用 PdfPipelineOptions
        # 理由：规避 DOCX/XLSX 在 SimplePipeline 初始化时对 do_chart_extraction 的硬编码访问
        shared_opts = PdfPipelineOptions(
            artifacts_path=self.local_artifacts_path,
            do_ocr=params.do_ocr,
            do_chart_extraction=False,     # 显式关闭以防干扰
            generate_table_images=True,    # 开启表格截图，供 VLM 模块使用
            generate_page_images=False,
            table_structure_options=table_opts # 整体赋值，符合 Pydantic 验证要求
        )

        # 3. 配置 OCR 引擎
        engine = (params.ocr_engine or "easyocr").lower()
        shared_opts.ocr_options = TesseractOcrOptions(lang=["chi_sim", "eng"]) if engine == "tesseract" else EasyOcrOptions(lang=["ch_sim", "en"])

        # 4. 转换器初始化
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=shared_opts),
                InputFormat.DOCX: WordFormatOption(pipeline_options=shared_opts),
                InputFormat.XLSX: ExcelFormatOption(pipeline_options=shared_opts),
                InputFormat.PPTX: PowerpointFormatOption(pipeline_options=shared_opts),
            }
        )
        
        loop = asyncio.get_running_loop()
        try:
            # 使用 self.executor 确保在 RHEL 8 的多核环境下不阻塞主事件循环
            result = await loop.run_in_executor(self.executor, converter.convert, file_path)
        except Exception as e:
            # 记录详细的异常追踪，方便在 NexusCube 日志中排查具体文件问题
            logger.error(f"Docling 转换失败: {file_path}, Error: {str(e)}")
            raise
        doc = result.document

        # 提前送去 VLM/LLM 重构复杂内容
        await self._enhance_document_content(doc, params)

        # 在分块阶段使用 VLM 进行标题定级
        if output_format == OutputFormat.CHUNKS:
            return await self._generate_chunks(doc, result, params) # 注意传了 result 进去拿渲染器

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