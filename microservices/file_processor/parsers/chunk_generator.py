import re
from typing import Any
from loguru import logger
from docling_core.types.doc.document import (
    DoclingDocument,
    TableItem,
    TextItem,
)
from .utils import ModelTask
from ..parser_schema import DocParserParams, ChunkMetadata, ChunkResult
from utils.clients import AIModelClient


class ChunkerGenerator:
    """分块生成类"""

    def __init__(self, params: DocParserParams):
        self.params = params
        self.min_len = params.min_chunk_len or 200
        self.max_len = params.chunk_size or 600
        self.chunk_count = 1
        self.model_task = ModelTask()
        self.model_client = AIModelClient()
        from .engine import VLMAnnotationPictureSerializer
        self.serializer = VLMAnnotationPictureSerializer()

    async def generate_chunks(self, doc: DoclingDocument, file_ext: str,
                              vlm_enhancement: dict,
                              prebuilt_hierarchy=None) -> list[ChunkResult]:
        """分块入口（统一走智能分块）"""
        return await self.generate_chunks_v2(doc, file_ext, vlm_enhancement,
                                             prebuilt_hierarchy=prebuilt_hierarchy)

    # ═══════════════════════════════════════════════════════════════
    # 智能分块 (v2) — 层级感知 + 语义边界

    async def generate_chunks_v2(self, doc: DoclingDocument, file_ext: str,
                                  vlm_enhancement: dict,
                                  prebuilt_hierarchy=None) -> list[ChunkResult]:
        """智能分块：层级感知 + 语义边界，无 per-chunk LLM 调用。

        流程: 层级树构建 → 多栏检测 → 跨页缝合 → 智能分块 → 全局摘要 → 转换输出

        Args:
            prebuilt_hierarchy: 可选，已经构建好的 SemanticNode 层级树。
                               如果提供，跳过 Step 1-3 直接进入分块。
                               用于级联管线中 QualityGate→VLM Repair→Merge 后传入。
        """
        from .hierarchy_builder import HierarchyBuilder
        from .layout_cluster import LayoutClusterer
        from .page_span_stitcher import PageSpanStitcher

        # 特殊格式：无标题层级，走专用处理器
        is_ppt = file_ext in [".pptx", ".ppt"]
        is_spreadsheet = file_ext in [".xlsx", ".xls", ".csv"]

        if is_ppt:
            return await self._generate_chunks_ppt(doc, vlm_enhancement)

        if is_spreadsheet:
            return await self._generate_chunks_spreadsheet(doc)

        kb_id = getattr(self.params, 'kb_id', None) or "unknown"
        file_id = getattr(self.params, 'file_id', None) or "unknown"

        # Step 0: 图片预序列化（保存到磁盘，记录文件名）
        pic_image_map: dict[int, str] = {}
        if self.params.generate_picture_images and self.params.image_dir:
            for pic in doc.pictures:
                try:
                    _, img_name = self.serializer.serialize(
                        item=pic, doc=doc, image_dir=self.params.image_dir
                    )
                    if img_name:
                        pic_image_map[id(pic)] = img_name
                except Exception as e:
                    logger.warning(f"图片序列化失败: {e}")

        # Step 1-3: 语义重构 (如果外部已提供 hierarchy 则跳过)
        if prebuilt_hierarchy is not None:
            hierarchy = prebuilt_hierarchy
            builder = HierarchyBuilder(doc)
        else:
            builder = HierarchyBuilder(doc)
            hierarchy = builder.build()
            clusterer = LayoutClusterer()
            hierarchy = clusterer.correct_reading_order(hierarchy)
            stitcher = PageSpanStitcher()
            hierarchy = stitcher.stitch(hierarchy)

        # Step 4: 智能分块
        smart = SmartChunker(
            min_len=self.params.min_chunk_len or 100,
            max_len=self.params.chunk_size or 800,
            hierarchy_builder=builder,
            kb_id=kb_id,
            file_id=file_id,
        )
        candidates = smart.chunk(hierarchy)

        # Step 5: 全局摘要（1 次 LLM 调用）
        global_summary = await self._get_global_summary(doc)

        # Step 6: 转换为 ChunkResult（纯规则，无 LLM！）
        results: list[ChunkResult] = []
        for i, cand in enumerate(candidates):
            result = self._candidate_to_result(cand, global_summary, i + 1, pic_image_map)
            results.append(result)

        return results

    async def _get_global_summary(self, doc: DoclingDocument) -> str:
        """生成文档全局摘要（1 次 LLM 调用，50 字以内）"""
        full_text_snapshot = ""
        max_chars = 3000
        for item, _ in doc.iterate_items():
            if isinstance(item, TextItem):
                full_text_snapshot += item.text + "\n"
                if len(full_text_snapshot) > max_chars:
                    break

        if not full_text_snapshot:
            return "未知主题文档"

        summary_prompt = (
            f"请用一句话（50字以内）概括以下文档的核心主题（含标准号或项目名）。"
            f"文档片段：\n{full_text_snapshot[:max_chars]}"
        )
        try:
            llm_res = await self.model_task.llm_task(
                self.model_client,
                self.params.llm_model,
                summary_prompt
            )
            if llm_res:
                summary = str(llm_res).replace("\n", " ").strip()
                logger.debug(f"文档全局摘要提取成功: {summary}")
                return summary
        except Exception as e:
            logger.warning(f"文档全局摘要提取失败: {e}")

        return "未知主题文档"

    async def _generate_chunks_spreadsheet(self, doc: DoclingDocument) -> list[ChunkResult]:
        """Excel/CSV 模式：每张表独立成块，大表按行拆分并用 section_id 关联"""
        import uuid

        global_summary = await self._get_global_summary(doc)
        results: list[ChunkResult] = []
        chunk_num = 1
        ROWS_PER_CHUNK = 40

        for item, _ in doc.iterate_items():
            if not isinstance(item, TableItem):
                continue

            try:
                md_table = item.export_to_markdown(doc=doc)
            except Exception:
                continue

            if not md_table or not md_table.strip():
                continue

            page_no = 1
            if hasattr(item, 'prov') and item.prov:
                page_no = item.prov[0].page_no

            lines = md_table.split('\n')
            section_id = f"tbl-{uuid.uuid4().hex[:12]}"

            if len(lines) <= ROWS_PER_CHUNK + 2:  # 表头2行 + 数据行
                results.append(ChunkResult.create(
                    content=md_table,
                    summary=global_summary,
                    header=f"Sheet{page_no} 数据表",
                    search_helper=f"{global_summary}\nSheet{page_no} 数据表\n{md_table[:300]}",
                    chunk_num=chunk_num,
                    chunk_type="table",
                    metadata=ChunkMetadata(page_num=page_no, is_sub_table=False),
                    section_id=section_id,
                ))
                chunk_num += 1
                continue

            # 大表拆分：找表头分隔行
            header_idx = 0
            for idx, line in enumerate(lines[:5]):
                if '---|---' in line.replace(' ', ''):
                    header_idx = idx
                    break

            table_header = "\n".join(lines[:header_idx + 1])
            data_lines = lines[header_idx + 1:]

            for i in range(0, len(data_lines), ROWS_PER_CHUNK):
                chunk_data = data_lines[i:i + ROWS_PER_CHUNK]
                combined = table_header + "\n" + "\n".join(chunk_data)
                results.append(ChunkResult.create(
                    content=combined,
                    summary=global_summary,
                    header=f"Sheet{page_no} 数据表 (行{i + 1}-{min(i + ROWS_PER_CHUNK, len(data_lines))})",
                    search_helper=f"{global_summary}\nSheet{page_no} 数据表\n{combined[:300]}",
                    chunk_num=chunk_num,
                    chunk_type="table",
                    metadata=ChunkMetadata(page_num=page_no, is_sub_table=True),
                    section_id=section_id,
                ))
                chunk_num += 1

        logger.info(f"Spreadsheet chunking: {chunk_num - 1} chunks from {len(results)} items")
        return results

    async def _generate_chunks_ppt(self, doc: DoclingDocument,
                                    vlm_enhancement: dict) -> list[ChunkResult]:
        """PPT 模式：每页一个 chunk + VLM 视觉总结"""
        from .hierarchy_builder import HierarchyBuilder
        builder = HierarchyBuilder(doc)
        hierarchy = builder.build()

        global_summary = await self._get_global_summary(doc)
        results: list[ChunkResult] = []
        chunk_num = 1

        for page_no in sorted(doc.pages.keys()):
            page_obj = doc.pages.get(page_no)
            if not page_obj:
                continue

            page_info = vlm_enhancement.get(page_no, {})
            slide_vlm_desc = page_info.get("description", "")
            slide_img_name = page_info.get("image_name", "")

            # 收集该页文本
            page_texts: list[tuple[str, int]] = []
            for child in hierarchy.children:
                for p_node in self._iter_descendants(child):
                    if p_node.page_num == page_no and p_node.text.strip():
                        page_texts.append((p_node.text, p_node.page_num))

            raw_text = "\n".join([t for t, _ in page_texts])

            final_parts = []
            if slide_vlm_desc:
                final_parts.append(f"【页面视觉总结】\n{slide_vlm_desc}")
            if raw_text:
                final_parts.append(f"【原始文本内容】\n{raw_text}")
            combined = "\n\n".join(final_parts)
            if not combined:
                continue

            result = ChunkResult.create(
                content=combined,
                summary=global_summary,
                header=f"幻灯片 {page_no}",
                search_helper=f"{global_summary}\n幻灯片 {page_no}\n{combined[:300]}",
                chunk_num=chunk_num,
                chunk_type="slide",
                metadata=ChunkMetadata(
                    page_num=page_no,
                    image_name=slide_img_name,
                    bbox=None,
                ),
            )
            results.append(result)
            chunk_num += 1

        return results

    def _candidate_to_result(self, cand, global_summary: str,
                              chunk_num: int,
                              pic_image_map: dict[int, str] | None = None) -> ChunkResult:
        """将 ChunkCandidate 转为 ChunkResult — 纯规则，不调用 LLM"""
        hierarchy_str = " > ".join(cand.hierarchy_path)
        last_heading = cand.hierarchy_path[-1] if cand.hierarchy_path else ""
        first_sentence = self._extract_first_sentence(cand.content)
        virtual_header = f"{last_heading} - {first_sentence}"[:200]

        search_helper = (
            f"文档: {global_summary}\n"
            f"章节: {hierarchy_str}\n"
            f"段落: {virtual_header}\n"
            f"内容: {cand.content[:500]}"
        )

        # 查找图片文件名（picture/slide 类型）
        image_name = None
        if cand.content_type in ("picture", "slide") and pic_image_map:
            for node in getattr(cand, "nodes", []):
                item = getattr(node, "item", None)
                if item is not None:
                    img_name = pic_image_map.get(id(item))
                    if img_name:
                        image_name = img_name
                        break

        return ChunkResult.create(
            content=cand.content,
            summary=global_summary,
            header=virtual_header,
            search_helper=search_helper,
            chunk_num=chunk_num,
            chunk_type=cand.content_type,
            metadata=ChunkMetadata(
                page_num=cand.page_range[0] if cand.page_range else 1,
                image_name=image_name,
                bbox=cand.bbox,
            ),
            hierarchy_path=cand.hierarchy_path,
            hierarchy_depth=cand.hierarchy_depth,
            heading_level=cand.heading_level,
            parent_chunk_id=cand.parent_chunk_id,
            section_id=cand.section_id,
        )

    @staticmethod
    def _extract_first_sentence(text: str) -> str:
        """从文本中提取第一句（最多 50 字）"""
        text = text.strip()
        for sep in ('。', '？', '！', '\n', '.', '?', '!'):
            idx = text.find(sep)
            if idx > 0:
                return text[:idx + 1].strip()[:50]
        return text[:50]

    @staticmethod
    def _iter_descendants(node) -> list:
        """递归遍历所有子孙节点"""
        result = []
        for child in node.children:
            result.append(child)
            result.extend(SmartChunker._iter_descendants_static(child))
        return result


class SmartChunker:
    """以语义边界（标题层级）为主，长度阈值为辅的智能分块引擎。

    原则:
    1. 标题作为块边界标记，不作为独立块
    2. 同一标题下的连续段落合并（不超过 max_len）
    3. 表格/图片独立成块，携带层级路径
    4. 超长段落（> max_len * 1.2）在句子边界切分
    5. 同一 section 下的所有 chunk 共享 section_id
    """

    def __init__(self, min_len: int, max_len: int,
                 hierarchy_builder, kb_id: str, file_id: str):
        self.min_len = min_len
        self.max_len = max_len
        self.builder = hierarchy_builder
        self.kb_id = kb_id
        self.file_id = file_id

    def chunk(self, hierarchy) -> list:
        """递归遍历层级树生成块候选"""
        chunks = []
        for section in hierarchy.children:
            if section.node_type == 'title':
                chunks.extend(self._chunk_section(section))
        chunks = self._merge_short_chunks(chunks)
        chunks = self._split_long_chunks(chunks)
        return chunks

    def _chunk_section(self, section) -> list:
        path = self.builder.get_breadcrumb_path(section)
        # path 的第一项是 section 自己的标题
        if section.text.strip():
            path = path + [section.text.strip()] if path else [section.text.strip()]

        section_id = self.builder.make_section_id(
            self.kb_id, self.file_id, path
        )
        chunks = []
        buffer_nodes, buffer_len = [], 0

        for child in section.children:
            if child.node_type in ('paragraph',):
                text = child.text.strip()
                if not text:
                    continue
                if buffer_len + len(text) > self.max_len and buffer_len >= self.min_len:
                    chunks.append(self._make_chunk(buffer_nodes, path, section_id))
                    buffer_nodes, buffer_len = [child], len(text)
                else:
                    buffer_nodes.append(child)
                    buffer_len += len(text)

            elif child.node_type in ('table', 'picture'):
                if buffer_nodes:
                    chunks.append(self._make_chunk(buffer_nodes, path, section_id))
                    buffer_nodes, buffer_len = [], 0
                chunks.append(self._make_chunk([child], path, section_id))

        if buffer_nodes:
            chunks.append(self._make_chunk(buffer_nodes, path, section_id))

        # 设置相邻 chunk 的 parent_chunk_id
        for i in range(1, len(chunks)):
            chunks[i].parent_chunk_id = chunks[i - 1].section_id
        return chunks

    def _make_chunk(self, nodes: list, hierarchy_path: list[str],
                    section_id: str) -> "ChunkCandidate":
        content = "\n\n".join([n.text.strip() for n in nodes if n.text.strip()])
        pages = sorted({n.page_num for n in nodes if n.page_num})
        bbox = nodes[0].bbox if nodes else None
        ct = 'mixed'
        if len(nodes) == 1:
            ct = nodes[0].node_type
            if ct == 'paragraph':
                ct = 'text'

        depth = len(hierarchy_path)
        heading_level = 0
        for n in nodes:
            if n.node_type == 'title' and n.level > 0:
                heading_level = n.level
                break

        return ChunkCandidate(
            nodes=nodes, hierarchy_path=hierarchy_path,
            content=content, content_type=ct,
            page_range=pages, bbox=bbox,
            hierarchy_depth=depth, heading_level=heading_level,
            section_id=section_id,
        )

    def _merge_short_chunks(self, chunks: list) -> list:
        """合并过短的块（< min_len）到前一个块"""
        if len(chunks) < 2:
            return chunks
        merged = []
        for c in chunks:
            if merged and len(c.content) < self.min_len:
                prev = merged[-1]
                if c.hierarchy_path == prev.hierarchy_path:
                    prev.content = prev.content + "\n" + c.content
                    prev.nodes.extend(c.nodes)
                    prev.page_range = sorted(set(prev.page_range + c.page_range))
                    prev.content_type = 'mixed'
                    continue
            merged.append(c)
        return merged

    def _split_long_chunks(self, chunks: list) -> list:
        """拆分过长的块（> max_len * 1.2），在句子边界切分"""
        result = []
        for c in chunks:
            if len(c.content) <= self.max_len * 1.2:
                result.append(c)
                continue
            # 在句子边界切分
            sub_content = ""
            for sentence in self._split_sentences(c.content):
                if len(sub_content) + len(sentence) > self.max_len and sub_content:
                    result.append(ChunkCandidate(
                        nodes=c.nodes[:1], hierarchy_path=c.hierarchy_path,
                        content=sub_content.strip(), content_type=c.content_type,
                        page_range=c.page_range, bbox=c.bbox,
                        hierarchy_depth=c.hierarchy_depth,
                        heading_level=c.heading_level,
                        section_id=c.section_id,
                    ))
                    sub_content = sentence
                else:
                    sub_content += " " + sentence if sub_content else sentence
            if sub_content.strip():
                result.append(ChunkCandidate(
                    nodes=c.nodes[-1:], hierarchy_path=c.hierarchy_path,
                    content=sub_content.strip(), content_type=c.content_type,
                    page_range=c.page_range, bbox=c.bbox,
                    hierarchy_depth=c.hierarchy_depth,
                    heading_level=c.heading_level,
                    section_id=c.section_id,
                ))
        return result

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """按中英文标点拆分句子"""
        import re
        return [s.strip() for s in re.split(r'(?<=[。！？.!?；;])\s*', text) if s.strip()]

    @staticmethod
    def _iter_descendants_static(node) -> list:
        result = []
        for child in node.children:
            result.append(child)
            result.extend(SmartChunker._iter_descendants_static(child))
        return result


from dataclasses import dataclass, field


@dataclass
class ChunkCandidate:
    """分块候选"""
    nodes: list = field(default_factory=list)
    hierarchy_path: list[str] = field(default_factory=list)
    content: str = ""
    content_type: str = "text"
    page_range: list[int] = field(default_factory=list)
    bbox: tuple | None = None
    hierarchy_depth: int = 0
    heading_level: int = 0
    parent_chunk_id: str | None = None
    section_id: str | None = None