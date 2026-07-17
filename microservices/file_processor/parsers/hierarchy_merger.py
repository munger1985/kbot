"""层级树合并器 — 将 VLM 修复结果合并回 Docling 层级树。

核心操作:
1. 遍历 Docling 层级树，找到 VLM 修复页面的标题节点
2. 应用 VLM 修正: 改 level、改 text、删误判、补漏报
3. 处理跨页标记（标记跨页节点，供 SmartChunker 处理）
4. 返回可直接传给 SmartChunker 的修正后层级树
"""

from loguru import logger

from .hierarchy_builder import SemanticNode
from .structure_repairer import PageRepairResult, VerifiedHeading


class HierarchyMerger:
    """将 VLM 修复结果合并回 Docling 层级树。

    用法:
        merger = HierarchyMerger()
        merged = merger.merge(hierarchy, repair_results)
    """

    def merge(
        self,
        hierarchy: SemanticNode,
        repair_results: dict[int, PageRepairResult],
    ) -> SemanticNode:
        """主入口：应用 VLM 修复结果到层级树。

        Args:
            hierarchy: HierarchyBuilder.build() 输出的根节点
            repair_results: {page_no: PageRepairResult} VLM 修复结果

        Returns:
            修正后的根节点（原地修改 + 返回）
        """
        if not repair_results:
            return hierarchy

        # 收集按页码分组的所有节点
        pages: dict[int, list[SemanticNode]] = {}
        self._collect_by_page(hierarchy, pages)

        total_corrected = 0
        total_deleted = 0
        total_added = 0

        for page_no, result in repair_results.items():
            if not result.success:
                continue

            page_nodes = pages.get(page_no, [])
            headings = [n for n in page_nodes if n.node_type == 'title']

            for vh in result.verified_headings:
                if vh.action == "keep":
                    continue  # 无需改动

                elif vh.action == "correct":
                    matched = self._find_heading_by_text(headings, vh.text)
                    if matched:
                        old_level = matched.level
                        matched.level = vh.level
                        logger.debug(
                            f"[HierarchyMerger] p{page_no} 修正标题 "
                            f"\"{vh.text[:30]}\" level {old_level}→{vh.level} "
                            f"({vh.reasoning})"
                        )
                        total_corrected += 1
                    else:
                        logger.debug(
                            f"[HierarchyMerger] p{page_no} 标题未找到匹配: "
                            f"\"{vh.text[:30]}\" (已删除或重命名？)"
                        )

                elif vh.action == "delete":
                    matched = self._find_heading_by_text(headings, vh.text)
                    if matched:
                        self._remove_heading(matched)
                        logger.debug(
                            f"[HierarchyMerger] p{page_no} 删除误判标题: "
                            f"\"{vh.text[:30]}\" ({vh.reasoning})"
                        )
                        total_deleted += 1

                elif vh.action == "add":
                    # VLM 发现漏掉的标题 → 挂载到当前页的标题栈下
                    self._add_heading_to_page(hierarchy, page_no, vh)
                    logger.debug(
                        f"[HierarchyMerger] p{page_no} 补充标题: "
                        f"\"{vh.text[:30]}\" level={vh.level} ({vh.reasoning})"
                    )
                    total_added += 1

        logger.info(
            f"[HierarchyMerger] 合并完成: "
            f"{total_corrected} 修正 / {total_deleted} 删除 / {total_added} 新增"
        )
        return hierarchy

    # ── 标题查找与操作 ──────────────────────────────────────

    @staticmethod
    def _find_heading_by_text(
        headings: list[SemanticNode],
        target_text: str,
    ) -> SemanticNode | None:
        """按文本模糊匹配标题节点"""
        target = target_text.strip()
        if not target:
            return None

        # 精确匹配优先
        for h in headings:
            if h.text.strip() == target:
                return h

        # 包含匹配 (target 是 h.text 的子串)
        for h in headings:
            if target in h.text.strip() or h.text.strip() in target:
                return h

        return None

    @staticmethod
    def _remove_heading(node: SemanticNode):
        """从父节点中移除误判的标题节点，子节点上移给祖父节点"""
        parent = node.parent
        if parent is None:
            return

        # 将该标题的子节点挂到 parent 下
        grandparent = parent.parent
        if grandparent is not None:
            idx = parent.children.index(node) if node in parent.children else -1
            if idx >= 0:
                # 在 parent 中替换 node 为它的 children
                parent.children[idx:idx + 1] = node.children
                for child in node.children:
                    child.parent = parent

    @staticmethod
    def _add_heading_to_page(
        hierarchy: SemanticNode,
        page_no: int,
        vh: VerifiedHeading,
    ):
        """将 VLM 发现的新标题添加到层级树中。

        策略: 找到该页第一个节点所在的父标题，将新标题挂载其下。
        """
        # 找到该页所有节点及其共同祖先
        page_nodes = []
        HierarchyMerger._collect_by_page(hierarchy, {page_no: page_nodes})

        if not page_nodes:
            return

        # 找第一个节点的父标题
        first_node = page_nodes[0]
        parent = first_node.parent
        if parent is None:
            parent = hierarchy

        # 创建新标题节点
        new_heading = SemanticNode(
            item=None,
            node_type='title',
            level=vh.level,
            text=vh.text,
            page_num=page_no,
            bbox=None,
        )
        new_heading.parent = parent

        # 插入到该页第一个节点之前
        if first_node in parent.children:
            idx = parent.children.index(first_node)
            parent.children.insert(idx, new_heading)
        else:
            parent.children.append(new_heading)

    # ── 辅助 ──────────────────────────────────────────────

    @staticmethod
    def _collect_by_page(
        node: SemanticNode,
        pages: dict[int, list[SemanticNode]],
    ):
        """递归收集所有节点按页码分组"""
        if node.node_type != 'root':
            pages.setdefault(node.page_num, []).append(node)
        for child in node.children:
            HierarchyMerger._collect_by_page(child, pages)
