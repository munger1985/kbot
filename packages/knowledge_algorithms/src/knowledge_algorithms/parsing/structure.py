"""提取 KC 的分栏/标题栈基线，参数版本化且不按空间距离强连图片。"""

from dataclasses import replace
import re

from ..contracts.content import ContentKind, ContentNode
from ..contracts.errors import require
from .contracts import LayoutProfile


def order_and_structure(
    nodes: tuple[ContentNode, ...], profile: LayoutProfile,
) -> tuple[tuple[ContentNode, ...], tuple[str, ...]]:
    require(len({n.key for n in nodes}) == len(nodes), "结构节点身份重复")
    by_page: dict[int, list[ContentNode]] = {}
    for node in nodes:
        by_page.setdefault(node.locators[0].page_no or 0, []).append(node)
    ordered = []
    for page_no in sorted(by_page):
        page = by_page[page_no]
        if any(node.locators[0].bbox is None for node in page):
            ordered.extend(sorted(page, key=lambda n: (n.reading_order, n.native_ref)))
            continue
        wide = sorted((node for node in page if
                       node.locators[0].bbox[2] - node.locators[0].bbox[0] >= profile.full_width_ratio),
                      key=lambda n: (n.locators[0].bbox[1], n.reading_order))
        narrow = [n for n in page if n not in wide]

        def band_order(band: list[ContentNode]) -> list[ContentNode]:
            vertical = lambda n: (n.locators[0].bbox[1], n.locators[0].bbox[0], n.reading_order, n.native_ref)
            centers = sorted((n.locators[0].bbox[0] + n.locators[0].bbox[2]) / 2 for n in band)
            gaps = [(b - a, (a + b) / 2) for a, b in zip(centers, centers[1:])]
            if not gaps or max(gaps)[0] < profile.column_gap:
                return sorted(band, key=vertical)
            split = max(gaps)[1]
            return sorted(band, key=lambda n: (
                (n.locators[0].bbox[0] + n.locators[0].bbox[2]) / 2 > split, vertical(n)))

        for boundary in wide:
            before = [n for n in narrow if n.locators[0].bbox[1] < boundary.locators[0].bbox[1]]
            ordered.extend(band_order(before))
            narrow = [n for n in narrow if n not in before]
            ordered.append(boundary)
        ordered.extend(band_order(narrow))
    stack: list[tuple[int, str]] = []
    result = []
    warnings = []
    available_keys = {n.key.object_key for n in nodes}
    for ordinal, node in enumerate(ordered):
        parent = node.parent_key
        if parent is not None:
            require(parent in available_keys and parent != node.key.object_key, "显式结构父节点无效")
        if node.kind == ContentKind.SECTION:
            level = node.heading_level
            if level is None:
                match = re.match(r"^\s*(\d+(?:\.\d+)*)(?:[\s、.)]|$)", node.text)
                level = match.group(1).count(".") + 1 if match else 1
                warnings.append("INFERRED_HEADING_LEVEL")
            require(level <= 9, "标题层级超过合同上限")
            while stack and stack[-1][0] >= level:
                stack.pop()
            if level > (stack[-1][0] if stack else 0) + 1:
                warnings.append("HEADING_LEVEL_GAP")
            parent = parent or (stack[-1][1] if stack else None)
            stack.append((level, node.key.object_key))
            node = replace(node, heading_level=level)
        elif parent is None and stack:
            parent = stack[-1][1]
        result.append(replace(node, reading_order=ordinal, parent_key=parent))
    parents = {n.key.object_key: n.parent_key for n in result}
    for origin in parents:
        visited = set()
        current = origin
        while current is not None:
            require(current not in visited, "结构父节点存在环")
            visited.add(current)
            current = parents[current]
    return tuple(result), tuple(dict.fromkeys(warnings))
