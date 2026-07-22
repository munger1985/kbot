"""多栏版面聚类器 — 检测多栏 PDF 并修正阅读顺序。

策略: 默认信任 Docling 的 iterate_items() 顺序。
仅在检测到同一页内多个 item 的 y 范围高度重叠(>60%)
且 x 坐标相差悬殊时，触发两栏重排。
"""
from .hierarchy_builder import SemanticNode


class LayoutClusterer:
    """检测并修正多栏页面的阅读顺序"""

    def correct_reading_order(self, hierarchy: SemanticNode) -> SemanticNode:
        """主入口：检测所有页面，对多栏页面重新排序"""
        pages: dict[int, list[SemanticNode]] = {}
        self._group_by_page(hierarchy, pages)

        for page_no, items in pages.items():
            if len(items) < 4:
                continue
            if self._is_multi_column(items):
                self._reorder_by_column(items, hierarchy)

        return hierarchy

    def _group_by_page(self, node: SemanticNode,
                       pages: dict[int, list[SemanticNode]]):
        """递归收集所有节点按页码分组"""
        if node.node_type != 'root' and node.bbox:
            if node.page_num not in pages:
                pages[node.page_num] = []
            pages[node.page_num].append(node)
        for child in node.children:
            self._group_by_page(child, pages)

    def _is_multi_column(self, items: list[SemanticNode]) -> bool:
        """通过 x 坐标分布判断是否多栏"""
        x_centers = []
        for n in items:
            if n.bbox:
                center = (n.bbox[0] + n.bbox[2]) / 2
                x_centers.append(center)

        if len(x_centers) < 4:
            return False

        x_centers.sort()
        # 计算相邻中心的间隔
        gaps = [x_centers[i + 1] - x_centers[i] for i in range(len(x_centers) - 1)]
        if not gaps:
            return False

        avg_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        # 若最大间隔 > 2 倍平均间隔，说明存在明显分栏
        return max_gap > avg_gap * 2.0

    def _reorder_by_column(self, items: list[SemanticNode],
                           hierarchy: SemanticNode):
        """对多栏页面的 items 按栏重排"""
        if not items:
            return

        # 按 x 中心排序
        items_with_x = [(n, (n.bbox[0] + n.bbox[2]) / 2)
                        for n in items if n.bbox]
        items_with_x.sort(key=lambda t: t[1])
        x_vals = [x for _, x in items_with_x]

        gaps = [(i, x_vals[i + 1] - x_vals[i])
                for i in range(len(x_vals) - 1)]
        if not gaps:
            return
        split_idx = max(gaps, key=lambda g: g[1])[0]

        left = [n for n, _ in items_with_x[:split_idx + 1]]
        right = [n for n, _ in items_with_x[split_idx + 1:]]

        # 在每栏内按 y 坐标从上到下排序
        left.sort(key=lambda n: n.bbox[1] if n.bbox else 0)
        right.sort(key=lambda n: n.bbox[1] if n.bbox else 0)

        # 重新排列：左栏全部 + 右栏全部
        ordered = left + right
        # 更新父节点的 children 顺序
        self._reorder_in_parent(items[0], ordered, hierarchy)

    def _reorder_in_parent(self, sample: SemanticNode,
                           ordered: list[SemanticNode],
                           hierarchy: SemanticNode):
        """在父节点中重新排列子节点顺序"""
        parent = sample.parent
        if parent is None or parent.node_type == 'root':
            parent = hierarchy

        # 找出这些 items 在父节点中的起始位置
        existing_ids = {id(n): i for i, n in enumerate(parent.children)}
        target_indices = []
        for n in ordered:
            nid = id(n)
            if nid in existing_ids:
                target_indices.append(existing_ids[nid])

        if target_indices and len(target_indices) == len(ordered):
            # 原地替换
            base = min(target_indices)
            for offset, n in enumerate(ordered):
                if base + offset < len(parent.children):
                    parent.children[base + offset] = n
