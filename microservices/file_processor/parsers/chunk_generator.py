import re
import json
from pathlib import Path
from typing import Any
from loguru import logger
from docling_core.types.doc.document import (
    DoclingDocument,
    DescriptionAnnotation,
    PictureItem,
    TableItem,
    TextItem, 
    TitleItem,
    SectionHeaderItem  # 用于标题处理的专用类
)
from .utils import ParserToolLib, ModelTask
from .engine import VLMAnnotationPictureSerializer
from ..parser_schema import DocParserParams, ChunkMetadata, ChunkResult
from utils.clients import AIModelClient
from services.default_prompt import PromptManager


class ChunkerGenerator:
    """分块生成类"""
    
    def __init__(self, params: DocParserParams):
        self.params = params
        self.min_len = params.min_chunk_len or 200
        self.max_len = params.chunk_size or 600
        self.chunk_count = 1
        self.model_task = ModelTask()
        self.model_client = AIModelClient()
        self.prompt_mgr = PromptManager()
        self.serializer = VLMAnnotationPictureSerializer()

    async def generate_chunks(self, doc: DoclingDocument, file_ext: str, vlm_enhancement: dict) -> list[ChunkResult]:
        """
        分块生成逻辑：
        1. Stage 1: 调用VLM处理复杂表格和图片
        2. Stage 2: 全局感知 - 提取文档全局摘要
        3. Stage 3: 语义聚合 - 聚合并判定 Header/Text
        4. Stage 4: 逻辑输出 - 注入全局背景
        
        PPT：  按照每一页一个chunk
        Excel：复杂Excel给VLM进行识别，每个table作为单独的chunk，
               除非chunk超过Embedding的max tokens，则按照每40行切分为子表，每个子表补充相同的表头。
        """
        chunk_results: list[ChunkResult] = []
        current_chunk_num = 1
        is_ppt = file_ext in [".pptx", ".ppt"]
        seen_image_hashes = set() # 用于过滤重复图片

        # --- Stage 1: 全局感知 (Global Context) ---
        # 提取前 3000 字用于生成全局摘要
        full_text_snapshot = ""
        max_summary_chars = 3000
        
        for item, _ in doc.iterate_items():
            if isinstance(item, TextItem):
                full_text_snapshot += item.text + "\n"
                if len(full_text_snapshot) > max_summary_chars: break
        
        global_summary = "未知主题文档"
        if full_text_snapshot:
            summary_prompt = f"请用一句话（50字以内）概括以下文档的核心主题（含标准号或项目名）。文档片段：\n{full_text_snapshot[:max_summary_chars]}"
            
            # 调用LLM进行背景总结
            llm_res = await self.model_task.llm_task(
                self.model_client, 
                self.params.llm_model,
                summary_prompt
            )
            if llm_res:
                global_summary = str(llm_res).replace("\n", " ").strip()
                logger.debug(f"文档全局摘要提取成功: {global_summary}")
            else:
                logger.warning("文档全局摘要提取失败，使用默认值")

        # --- Stage 2: add_to_results 闭包，处理 chunk 结果生成 ---
        async def add_to_results(content, item_ref, c_type="text", img_name=None) -> None:
            nonlocal current_chunk_num
            if not content.strip(): return
            
            # 使用LLM生成动态提取虚拟标题，如果失败则使用从内容提取的默认值
            lines = [l for l in content.split('\n') if l.strip()]
            virtual_header = lines[0].replace('#', '').strip()[:50] if lines else "正文详情"
            # 使用LLM生成virtual_header
            header_prompt = f"阅读以下文本片段并生成5-10个字的标题，文本片段：{content}"
            
            # 调用LLM进行背景总结
            llm_res = await self.model_task.llm_task(
                self.model_client, 
                self.params.llm_model,
                header_prompt
            )
            if llm_res:
                virtual_header = str(llm_res).replace("\n", " ").strip()
                logger.debug(f"文档片段标题生成成功: {virtual_header}")
            else:
                logger.warning("文档片段标题生成失败，使用默认值")
            
            # 构造 search_helper：[全局背景] > [虚拟标题] > [内容前缀]
            search_helper = f"{global_summary} > {virtual_header} > {content[:120].replace('\n', ' ')}"

            metadata = ChunkMetadata(
                page_num=self._get_page_num(item_ref) if item_ref else 1,
                image_name=img_name
            )
            
            chunk = ChunkResult.create(
                content=content,
                summary=global_summary,
                header=virtual_header,  # 传入提取的虚拟标题
                search_helper=search_helper, # 显式传入增强检索字段
                chunk_num=current_chunk_num,
                chunk_type=c_type,
                metadata=metadata
            )
            
            chunk_results.append(chunk) 
            current_chunk_num += 1

        # --- Stage 3: 解析路由：PPT单独处理，其他文档进行语义聚合 ---

        if is_ppt:
            logger.debug(f"文档类型为 PPT，以 PPT 模式开始解析")
            # PPT 模式：按页物理聚合
            page_buckets = {}
            page_titles = {}

            # 1. 建立基础内容 Bucket
            for item, _ in doc.iterate_items():
                p_no = self._get_page_num(item)
                if p_no not in page_buckets: 
                    page_buckets[p_no] = []
                    page_titles[p_no] = ""
                
                # 提取标题：用于生成 Chunk 的元数据
                if isinstance(item, (SectionHeaderItem, TitleItem)):
                    txt = item.text.strip()
                    if txt and not page_titles[p_no]:
                        page_titles[p_no] = txt
                
                # 收集文本：保留原始文本用于关键词检索 (Keyword Hit)
                if isinstance(item, TextItem):
                    page_buckets[p_no].append(item.text)
                
                # 注意：PPT 模式下，不再单独处理 TableItem 和 PictureItem
                # 因为它们的内容已经包含在“整页视觉总结”中了

            # 2. 物理页排序并注入 VLM 视觉总结
            for p_no in sorted(page_buckets.keys()):
                # 获取该页的 Page 对象
                page_obj = doc.pages.get(p_no)
                if not page_obj:
                    logger.warning(f"PPT 页面 {p_no} 不存在")
                    continue
                
                # --- 核心：提取预处理回填的视觉总结 ---
                page_info = vlm_enhancement.get(p_no, {})
                slide_vlm_desc = page_info.get("description", None)
                slide_img_name = page_info.get("image_name", None)
                
                logger.debug(f"slide_img_name: {slide_img_name}, slide No.{p_no}")

                # 3. 构造最终的 Chunk 内容
                bucket_content = page_buckets[p_no]
                raw_text = "\n".join(bucket_content).strip()
                
                # 组合内容：视觉总结置顶（语义核心）+ 原始文本（检索辅助）
                final_parts = []
                if slide_vlm_desc:
                    final_parts.append(f"【页面视觉总结】\n{slide_vlm_desc}")
                
                if raw_text:
                    final_parts.append(f"【原始文本内容】\n{raw_text}")
                
                combined_text = "\n\n".join(final_parts)
                if not combined_text: continue

                # 5. 保存整页截图并在前端展示
                await add_to_results(combined_text, None, c_type="slide", img_name=slide_img_name)
        else:
            logger.debug("使用常规模式：线性追踪解析文档")
            # 常规模式：线性追踪
            text_buffer = []
            current_len = 0
            MIN_CHUNK_LEN = self.params.min_chunk_len or 200 # 最小合并字符数，低于此值会尝试与下一段合并
            MAX_CHUNK_LEN = self.params.chunk_size or 1000 # 达到此长度强制刷出

            for item, _ in doc.iterate_items():
                # 1. 标题逻辑：作为内容存入buffer
                if isinstance(item, SectionHeaderItem):
                    header_line = f"## {item.text.strip()}"
                    text_buffer.append(header_line)
                    current_len += len(header_line)
                    continue

                # 2. 表格逻辑
                if isinstance(item, TableItem):
                    if text_buffer:
                        await add_to_results("\n".join(text_buffer), item)
                        text_buffer, current_len = [], 0
                    
                    # 处理表格
                    table_chunks = await self._process_table_vlm(item, doc, self.params, "表格")
                    for tc in table_chunks:
                        raw_content = tc.get("content", "").strip()
                        final_table_md = ParserToolLib.ensure_markdown_table_integrity(raw_content)
                        await add_to_results(final_table_md, item, c_type="table")
                    continue

                # 3. 图片逻辑
                if isinstance(item, PictureItem):
                    if text_buffer:
                        await add_to_results("\n".join(text_buffer), item)
                        text_buffer, current_len = [], 0
                    
                    skip, vlm_desc = await self._should_skip_image(item, seen_image_hashes)
                    logger.debug(f"图片处理诊断: skip={skip}, desc_len={len(vlm_desc) if vlm_desc else 0}, hash_seen={len(seen_image_hashes)}")
                    if not skip:
                        try:
                            # 执行序列化逻辑，获取验证过的 img_name
                            _, img_name = self.serializer.serialize(item=item, doc=doc, image_dir=self.params.image_dir)
                            if img_name:
                                logger.debug(f"图片序列化成功: {img_name}")
                                await add_to_results(
                                    content=f"[图片内容描述]: {vlm_desc}", 
                                    item_ref=item, 
                                    c_type="picture", 
                                    img_name=img_name
                                )
                            else:
                                logger.error("图片序列化失败：返回的 img_name 为空")
                        except Exception as e:
                            logger.exception(f"序列化图片时发生异常: {e}")
                    continue

                # 4. 正文逻辑
                if isinstance(item, TextItem):
                    txt = item.text.strip()
                    if not txt: continue
                    text_buffer.append(txt)
                    current_len += len(txt)
                    # 只有当 buffer 加上新内容会“过度溢出”时，才刷出旧内容
                    # 这样可以保证 text_buffer 总是尽量接近 MAX_CHUNK_LEN
                    if current_len >= MAX_CHUNK_LEN:
                        await add_to_results("\n".join(text_buffer), item)
                        text_buffer, current_len = [], 0

            # 5. 收尾工作与合并
            if text_buffer:
                final_content = "\n".join(text_buffer)
                merged = False
                # 只有内容短于阈值时才尝试合并
                if len(final_content) < MIN_CHUNK_LEN and chunk_results:
                    # 从后往前找第一个类型为 "text" 的 chunk
                    for i in range(len(chunk_results) - 1, -1, -1):
                        target_chunk = chunk_results[i]
                        
                        # 限制 1：必须是正文类型
                        # 限制 2：合并后不能超过 MAX_CHUNK_LEN 太远 (比如 1.2 倍)
                        if target_chunk.chunk_type == "text":
                            new_content = target_chunk.content + "\n" + final_content
                            if len(new_content) < MAX_CHUNK_LEN * 1.2:
                                target_chunk.content = new_content
                                # 合并后需要重新生成该块的虚拟标题和检索路径
                                new_lines = [l for l in new_content.split('\n') if l.strip()]
                                new_v_header = new_lines[0].replace('#','').strip()[:50] if new_lines else "正文片段"
                                target_chunk.header = new_v_header
                                target_chunk.search_helper = f"{global_summary} > {new_v_header} > {new_content[:120].replace('\n',' ')}"
                                merged = True
                                break
                
                # 如果不符合合并条件或没找到合适的 text chunk，则独立成块
                if not merged:
                    await add_to_results(final_content, None)

        return chunk_results
    
    def _get_page_num(self, item: Any) -> int:
        try:
            # 1. 标准路径 (PPT 元素通常走这里)
            if hasattr(item, "prov") and item.prov:
                # 增加对 page_no 的防御性检查
                p_no = getattr(item.prov[0], "page_no", None)
                if p_no is not None:
                    return p_no

            # 2. 针对某些版本的 Docling，可能会把位置信息放在属性里
            # 有些 Item 具有 origin 属性，里面包含 page_index (从0开始)
            origin = getattr(item, "origin", None)
            if origin:
                # PPT 解析中 page_index 往往就是 Slide 索引
                return getattr(origin, "page_index", 0) + 1 

            # 3. 兼容 page_reference
            page_ref = getattr(item, "page_reference", None)
            if page_ref:
                return getattr(page_ref, "page_no", 1) if not isinstance(page_ref, dict) else page_ref.get("page_no", 1)

        except Exception as e:
            logger.debug(f"页码获取异常: {e}")
        return 1

    async def _process_table_vlm(self, item: TableItem, doc: DoclingDocument, params: DocParserParams, current_header: str) -> list[dict]:
        """专门处理表格的分块逻辑：VLM解析 + JSON切片 + 物理硬切防溢出"""
        table_chunks = []
        img_name = None
        TABLE_ROW_STEP = 40 
        MAX_CHAR_LIMIT = 15000
        
        if item.image: # 如果表格有关联图片（PDF中的表格通常有截图）
            _, img_name = self.serializer.serialize(item=item, doc=doc, image_dir=self.params.image_dir) # type: ignore

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
            "current_header": current_header,
            "chunk_type": "table",
            "metadata": {
                "page_num": self._get_page_num(item),
                "is_sub_table": len(parts_to_verify) > 1,
                "image_name": img_name
            }
        } for tc in table_chunks]
    
    async def _should_skip_image(self, item: PictureItem, seen_hashes: set) -> tuple[bool, str]:
        """
        重构后的过滤器：仅依赖 _enhance_document_content 的预处理结果
        """
        # 1. 获取物理层 Hash (直接从 item 引用中获取，如果之前没存，这里再取一次 MD5)
        # 注意：为了性能，建议在 _enhance_document_content 里把 hash 塞进 metadata，这里直接取
        img_hash = None
        vlm_desc = ""

        # --- 从预处理注入的 annotations 中提取信息 ---
        for anno in item.annotations:
            if not isinstance(anno, DescriptionAnnotation):
                continue
            
            # 提取之前存入的 Hash
            if anno.provenance == "hash_marker":
                img_hash = anno.text
            # 检查是否被尺寸过滤器拦截
            elif anno.provenance == "vlm_inference":
                vlm_desc = anno.text


        # 2. 逻辑判定
        # 过滤规则 A: 重复图过滤 (基于 Hash)
        if img_hash:
            if img_hash in seen_hashes:
                return True, "" # 已经处理过同指纹图片，跳过输出

        # 过滤规则 B: 无语义图过滤 (基于尺寸或 VLM 返回)
        if not vlm_desc or vlm_desc.strip() in ["", "[NONE]", "[VLM 无法生成描述]"]:
            return True, ""
        
        # 3. 记录已输出的 Hash，并返回描述
        if img_hash:
            seen_hashes.add(img_hash)

        return False, vlm_desc