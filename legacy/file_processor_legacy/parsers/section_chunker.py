"""精准引擎 — Markdown → ChunkResult 转换器。

按 Markdown 标题（# ## ###）切分，每个 section 生成一个 chunk。
通过 <!-- page:N --> 标记跟踪页码来源。
"""

import re
import uuid

from loguru import logger

from ..parser_schema import ChunkResult, ChunkMetadata
from .precision_analyzer import PageMarkdown


# 标题行正则: 可选空白 + 1-6 个 # + 至少1个空白 + 标题文字
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# 页面标记正则
PAGE_MARKER_RE = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->")

# [IMAGE: 描述] 标记正则
IMAGE_MARKER_RE = re.compile(r'\[IMAGE:\s*(.+?)\]')


class MarkdownSectionChunker:
    """Markdown → ChunkResult 转换器"""

    def convert(
        self,
        pages: list[PageMarkdown],
        global_summary: str,
        file_id: str = "",
    ) -> list[ChunkResult]:
        """将多页 Markdown 合并后按标题切分，生成 chunks。

        Args:
            pages: 按页码排序的 PageMarkdown 列表
            global_summary: 文档全局摘要
            file_id: 文件 ID

        Returns:
            ChunkResult 列表
        """
        # Step 1: 合并所有有意义的页，用 page 标记分隔
        full_md_parts = []
        for p in pages:
            if p.meaningful and p.markdown.strip():
                full_md_parts.append(f"<!-- page:{p.page} -->\n{p.markdown.strip()}")

        if not full_md_parts:
            logger.warning("[MarkdownSectionChunker] 无有意义内容")
            return []

        full_md = "\n\n".join(full_md_parts)

        # Step 2: 按标题切分 + 跟踪页码
        sections = self._split_by_headings(full_md)

        # Step 3: 每个 section → ChunkResult
        results: list[ChunkResult] = []
        hierarchy_stack: list[str] = []

        for sec in sections:
            content = sec["content"].strip()
            if not content:
                continue

            # 过滤掉纯 page 标记行
            content = self._clean_page_markers(content)
            if not content:
                continue

            level = sec["level"]
            title = sec["title"]
            page_no = sec.get("page_no", 1)

            # 更新层级路径
            while hierarchy_stack and len(hierarchy_stack) >= level:
                hierarchy_stack.pop()
            if title:
                hierarchy_stack.append(title)

            hierarchy_path = list(hierarchy_stack)
            header = " > ".join(hierarchy_path) if hierarchy_path else "正文"

            search_helper = (
                f"文档: {global_summary}\n"
                f"章节: {header}\n"
                f"内容: {content[:500]}"
            )

            chunk_num = len(results) + 1
            section_id = f"vlm-{file_id[:8]}-s{chunk_num:04d}"

            # 检测 [IMAGE:xxx] 标记 → picture 类型
            image_markers = IMAGE_MARKER_RE.findall(content)
            if image_markers:
                chunk_type = "picture"
                image_name = f"page_{page_no}.png"
            elif sec.get("is_table"):
                chunk_type = "table"
                image_name = None
            else:
                chunk_type = "text"
                image_name = None

            results.append(ChunkResult.create(
                content=content,
                summary=global_summary,
                header=header,
                search_helper=search_helper,
                chunk_num=chunk_num,
                chunk_type=chunk_type,
                metadata=ChunkMetadata(page_num=page_no, image_name=image_name),
                hierarchy_path=hierarchy_path,
                hierarchy_depth=len(hierarchy_path),
                heading_level=level,
                section_id=section_id,
            ))

        logger.info(
            f"[MarkdownSectionChunker] {len(sections)} markdown sections → "
            f"{len(results)} chunks"
        )
        return results

    # ── Markdown 切分 ─────────────────────────────────────────

    @staticmethod
    def _split_by_headings(markdown: str) -> list[dict]:
        """按标题切分 Markdown 为 sections，跟踪页码来源。

        Returns:
            [{"level": 1, "title": "1 范围", "content": "...",
              "is_table": False, "page_no": 3}, ...]
        """
        lines = markdown.split("\n")
        sections: list[dict] = []
        current_lines: list[str] = []
        current_title = ""
        current_level = 0
        current_page = 1

        for line in lines:
            # 检测页面标记
            pm = PAGE_MARKER_RE.match(line)
            if pm:
                current_page = int(pm.group(1))
                continue

            m = HEADING_RE.match(line)
            if m:
                if current_lines:
                    sections.append({
                        "level": current_level,
                        "title": current_title,
                        "content": "\n".join(current_lines).strip(),
                        "is_table": _has_markdown_table(current_lines),
                        "page_no": current_page,
                    })

                hashes = m.group(1)
                current_level = len(hashes)
                current_title = m.group(2).strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        # 最后一个 section
        if current_lines:
            sections.append({
                "level": current_level,
                "title": current_title,
                "content": "\n".join(current_lines).strip(),
                "is_table": _has_markdown_table(current_lines),
                "page_no": current_page,
            })

        return sections

    @staticmethod
    def _clean_page_markers(content: str) -> str:
        """移除内容中的 page 标记行"""
        lines = content.split("\n")
        cleaned = [l for l in lines if not PAGE_MARKER_RE.match(l)]
        return "\n".join(cleaned).strip()


def _has_markdown_table(lines: list[str]) -> bool:
    """检测是否包含 Markdown 表格（|---| 分隔行）"""
    for line in lines:
        if "---|---" in line.replace(" ", ""):
            return True
    return False
