"""跨页段落缝合器 — 检测并缝合跨页的段落。

判断条件(AND):
1. 上一页最后段落 bbox 延伸到页面底部(bbox.b > 0.85)
2. 下一页第一个段落从页面顶部开始(bbox.t < 0.15)
3. 上一段不以句号/问号/感叹号结尾
4. 下一页段落不以大写字母/数字序号开头（排除天然新段落）
"""
from loguru import logger
from .hierarchy_builder import SemanticNode


_SENTENCE_ENDS = frozenset({'。', '？', '！', '.', '?', '!', '；', ';'})


class PageSpanStitcher:
    """检测并合并跨页段落"""

    def stitch(self, hierarchy: SemanticNode) -> SemanticNode:
        """遍历层级树，检测并合并跨页段落"""
        # 收集所有段落，按页码分组
        paragraphs: list[SemanticNode] = []
        self._collect_paragraphs(hierarchy, paragraphs)
        if not paragraphs:
            return hierarchy

        # 按页码分组
        pages: dict[int, list[SemanticNode]] = {}
        for p in paragraphs:
            pages.setdefault(p.page_num, []).append(p)

        sorted_pages = sorted(pages.keys())
        for i in range(len(sorted_pages) - 1):
            page_a = sorted_pages[i]
            page_b = sorted_pages[i + 1]
            last_of_a = self._last_paragraph(pages[page_a])
            first_of_b = self._first_paragraph(pages[page_b])

            if self._should_stitch(last_of_a, first_of_b):
                logger.debug(
                    f"[PageSpanStitch] 缝合跨页段落: "
                    f"p{page_a}#{last_of_a.text[:30]}... ↔ p{page_b}#{first_of_b.text[:30]}..."
                )
                self._merge_nodes(last_of_a, first_of_b)
                # 从 pages[page_b] 移除已合并节点
                if first_of_b in pages[page_b]:
                    pages[page_b].remove(first_of_b)

        return hierarchy

    def _collect_paragraphs(self, node: SemanticNode,
                            result: list[SemanticNode]):
        if node.node_type == 'paragraph' and node.text.strip():
            result.append(node)
        for child in node.children:
            self._collect_paragraphs(child, result)

    @staticmethod
    def _last_paragraph(items: list[SemanticNode]) -> SemanticNode | None:
        return items[-1] if items else None

    @staticmethod
    def _first_paragraph(items: list[SemanticNode]) -> SemanticNode | None:
        return items[0] if items else None

    def _should_stitch(self, node_a: SemanticNode | None,
                       node_b: SemanticNode | None) -> bool:
        if node_a is None or node_b is None:
            return False
        if node_b.node_type == 'title':
            return False

        # 条件 1 & 2: 位置检测
        if not self._is_page_bottom(node_a):
            return False
        if not self._is_page_top(node_b):
            return False

        # 条件 3: node_a 不以结束标点结尾
        text_a = node_a.text.strip()
        if text_a and text_a[-1] in _SENTENCE_ENDS:
            return False

        # 条件 4: node_b 不以大写/数字开头(中文)
        text_b = node_b.text.strip()
        if text_b:
            first_char = text_b[0]
            if first_char.isupper() or first_char.isdigit():
                return False

        return True

    @staticmethod
    def _is_page_bottom(node: SemanticNode) -> bool:
        if not node.bbox:
            return False
        return node.bbox[3] > 0.85

    @staticmethod
    def _is_page_top(node: SemanticNode) -> bool:
        if not node.bbox:
            return False
        return node.bbox[1] < 0.15

    @staticmethod
    def _merge_nodes(node_a: SemanticNode, node_b: SemanticNode):
        """将 node_b 的文本合并到 node_a，标记跨页"""
        node_a.text = node_a.text + " " + node_b.text
        node_a.is_page_span = True
        if node_a.page_num not in node_a.span_pages:
            node_a.span_pages.append(node_a.page_num)
        if node_b.page_num not in node_a.span_pages:
            node_a.span_pages.append(node_b.page_num)
        # 从父节点中移除 node_b
        parent = node_b.parent
        if parent and node_b in parent.children:
            parent.children.remove(node_b)
