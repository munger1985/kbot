"""层级树构建器 — 从 DoclingDocument 的扁平 items 构建带层级关系的语义树。

核心功能:
1. 识别 SectionHeaderItem / TitleItem 建立标题栈
2. 栈顶标题作为后继内容节点（段落/表格/图片）的父节点
3. 新标题根据 level 弹出旧标题，维护层级关系
4. 对 DS OCR 注入的 item（annotations 中有 dsocr_type），按元素类型分类挂载
5. 提供 get_breadcrumb_path(node) 返回从根到当前节点的完整路径
6. 为每个 section 生成唯一的 section_id
"""
import hashlib
from dataclasses import dataclass, field
from typing import Any
from loguru import logger

from docling_core.types.doc.document import (
    DoclingDocument,
    TitleItem,
    SectionHeaderItem,
    TableItem,
    PictureItem,
    TextItem,
    DescriptionAnnotation,
)


@dataclass
class SemanticNode:
    """语义节点：代表文档中的一个语义单元"""
    item: Any = None
    node_type: str = ""              # root / title / paragraph / table / picture / list
    level: int = 0                   # heading 级别 0-6
    text: str = ""
    children: list["SemanticNode"] = field(default_factory=list)
    parent: "SemanticNode | None" = None
    page_num: int = 1
    bbox: tuple | None = None        # (l, t, r, b) 归一化坐标
    is_page_span: bool = False
    span_pages: list[int] = field(default_factory=list)
    self_ref: str = ""               # Docling item.self_ref


class HierarchyBuilder:
    """从 DoclingDocument 构建语义层级树"""

    def __init__(self, doc: DoclingDocument):
        self.doc = doc
        self.root = SemanticNode(node_type="root", text="文档根节点")
        self._section_counter: dict[str, int] = {}

    def build(self) -> SemanticNode:
        """主入口：构建完整层级树"""
        title_stack: list[SemanticNode] = []

        for item, _ in self.doc.iterate_items():
            # 1. 处理标题
            if isinstance(item, (TitleItem, SectionHeaderItem)):
                level = getattr(item, 'level', 1)
                text = getattr(item, 'text', '') or ''
                node = self._create_node(item, 'title', level, text)

                while title_stack and title_stack[-1].level >= level:
                    title_stack.pop()

                if title_stack:
                    parent = title_stack[-1]
                else:
                    parent = self.root
                parent.children.append(node)
                node.parent = parent
                title_stack.append(node)
                continue

            # 2. 对 DS OCR 注入的 item 按类型路由
            dsocr_type = self._get_dsocr_type(item)

            if dsocr_type == 'table':
                parent = self._find_parent(title_stack)
                node = self._create_node(item, 'table', 0,
                                         getattr(item, 'text', '') or '')
                parent.children.append(node)
                node.parent = parent
                continue

            if dsocr_type == 'picture':
                parent = self._find_parent(title_stack)
                node = self._create_node(item, 'picture', 0,
                                         getattr(item, 'text', '') or '')
                parent.children.append(node)
                node.parent = parent
                continue

            # 3. 处理表格
            if isinstance(item, TableItem):
                parent = self._find_parent(title_stack)
                text = getattr(item, 'text', '') or ''
                node = self._create_node(item, 'table', 0, text)
                parent.children.append(node)
                node.parent = parent
                continue

            # 4. 处理图片 — 读取 DS OCR / VLM 注入的增强文本
            if isinstance(item, PictureItem):
                parent = self._find_parent(title_stack)
                ocr_text = ""
                vlm_text = ""
                for anno in getattr(item, "annotations", []):
                    if isinstance(anno, DescriptionAnnotation):
                        if anno.provenance == "ocr_inference" and anno.text:
                            ocr_text = anno.text
                        elif anno.provenance == "vlm_inference" and anno.text:
                            vlm_text = anno.text
                # DS OCR 文字优先；无 OCR 则用 VLM 语义描述；都没有则兜底 item.text
                if ocr_text and vlm_text:
                    text = f"{ocr_text}\n\n[图片描述]: {vlm_text}"
                elif ocr_text:
                    text = ocr_text
                elif vlm_text:
                    text = vlm_text
                else:
                    text = getattr(item, 'text', '') or ''
                node = self._create_node(item, 'picture', 0, text)
                parent.children.append(node)
                node.parent = parent
                continue

            # 5. 处理正文
            if isinstance(item, TextItem):
                parent = self._find_parent(title_stack)
                text = getattr(item, 'text', '').strip()
                if not text:
                    continue
                node = self._create_node(item, 'paragraph', 0, text)
                parent.children.append(node)
                node.parent = parent
                continue

        return self.root

    def _create_node(self, item: Any, node_type: str, level: int,
                     text: str) -> SemanticNode:
        page_num = self._get_page_num(item)
        bbox = self._get_bbox(item)
        self_ref = getattr(item, 'self_ref', '') or ''

        return SemanticNode(
            item=item, node_type=node_type, level=level,
            text=text, page_num=page_num, bbox=bbox,
            self_ref=self_ref,
        )

    def _find_parent(self, title_stack: list[SemanticNode]) -> SemanticNode:
        return title_stack[-1] if title_stack else self.root

    def get_breadcrumb_path(self, node: SemanticNode) -> list[str]:
        """获取从根到当前节点的层级路径"""
        path: list[str] = []
        current = node.parent
        while current and current.node_type != 'root':
            if current.text.strip():
                path.insert(0, current.text.strip())
            current = current.parent
        return path

    def make_section_id(self, kb_id: str, file_id: str,
                        hierarchy_path: list[str]) -> str:
        """为 section 生成稳定的唯一标识"""
        raw = f"{kb_id}|{file_id}|{'/'.join(hierarchy_path)}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _get_page_num(item: Any) -> int:
        try:
            if hasattr(item, "prov") and item.prov:
                return getattr(item.prov[0], "page_no", 1)
        except Exception:
            pass
        return 1

    @staticmethod
    def _get_bbox(item: Any) -> tuple | None:
        try:
            if hasattr(item, "prov") and item.prov:
                b = item.prov[0].bbox
                if b:
                    return (b.l, b.t, b.r, b.b)
        except Exception:
            pass
        return None

    @staticmethod
    def _get_dsocr_type(item: Any) -> str | None:
        """提取 DS OCR 注入的元素类型标注"""
        for anno in getattr(item, "annotations", []):
            if isinstance(anno, DescriptionAnnotation) and anno.provenance == "dsocr_type":
                return anno.text
        return None
