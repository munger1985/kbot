"""Markdown 导入规则：仅将无属性的换行标签识别为排版。"""

from html import unescape
import re

from marko import Markdown
from marko.block import CodeBlock, FencedCode, HTMLBlock
from marko.ext.gfm.elements import Table, TableCell, TableRow
from marko.helpers import MarkoExtension
from marko.inline import CodeSpan, Image, InlineHTML, RawText

from knowledge_algorithms.contracts.errors import require


_BREAK_TAG = r"<br[ \t]*/?>"
_BREAK = re.compile(_BREAK_TAG, re.IGNORECASE | re.ASCII)
_BREAK_LINE = re.compile(rf"[ \t]*(?:{_BREAK_TAG}[ \t]*)+\n?", re.IGNORECASE | re.ASCII)
_EMBEDDED_IMAGE = re.compile(r"data:image/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}")


class _GuardedHTMLBlock(HTMLBlock):
    override = True

    @classmethod
    def match(cls, source):
        line = source.next_line()
        # 仅有换行标签的一行交给行内解析，不让 HTML 块吞掉后续正文。
        return False if line and _BREAK_LINE.fullmatch(line) else super().match(source)


class _GuardedInlineHTML(InlineHTML):
    override = True
    parse_group = 0
    pattern = re.compile(rf"(?ai:{_BREAK_TAG})|{InlineHTML.pattern.pattern}")


class _GuardedImage(Image):
    override = True

    def __init__(self, match):
        super().__init__(match)
        # 引用式图片的目标来自定义，不占行内图片的匹配区间。
        require(match.start(2) < match.end(2) and _EMBEDDED_IMAGE.fullmatch(self.dest) is not None,
                "Markdown 图片必须是已登记的内嵌资源", stage="convert")


def _is_break(element):
    return isinstance(element, InlineHTML) and _BREAK.fullmatch(element.children) is not None


def _parse_markdown(text):
    parser = Markdown()
    parser.use(MarkoExtension(elements=[_GuardedHTMLBlock, _GuardedInlineHTML, _GuardedImage,
                                       Table, TableRow, TableCell]))
    document = parser.parse(text)
    pending = [iter((document,))]
    while pending:
        element = next(pending[-1], None)
        if element is None:
            pending.pop()
            continue
        require(not isinstance(element, (HTMLBlock, InlineHTML)) or _is_break(element),
                "Markdown 包含不允许的原始 HTML", stage="convert")
        if isinstance(element, (CodeBlock, FencedCode, CodeSpan)):
            continue
        if type(element) is RawText:
            require("![" not in element.children, "Markdown 包含未登记的图片语法", stage="convert")
        children = getattr(element, "children", None)
        if isinstance(children, (list, tuple)) and children:
            pending.append(iter(children))
    return document


def validate_markdown(raw):
    _parse_markdown(raw.decode("utf-8"))


def _cell_text(cell, images):
    parts = []
    pending = [iter(cell.children)]
    while pending:
        child = next(pending[-1], None)
        if child is None:
            pending.pop()
            continue
        if _is_break(child):
            parts.append("\n")
        else:
            if isinstance(child, Image):
                images.append(child)
            children = getattr(child, "children", None)
            if isinstance(children, str):
                parts.append(unescape(children) if isinstance(child, RawText) else children)
            elif isinstance(children, (list, tuple)) and children:
                pending.append(iter(children))
    return "".join(parts)


def guarded_markdown_backend():
    from docling.backend.md_backend import MarkdownDocumentBackend
    from docling_core.types.doc import DoclingDocument, DocumentOrigin, TableCell as DoclingCell, TableData

    class GuardedMarkdownBackend(MarkdownDocumentBackend):
        def convert(self):
            require(self.is_valid(), "Markdown 转换器未成功读取输入", stage="convert")
            element = _parse_markdown(self.markdown)
            origin = DocumentOrigin(filename=self.file.name or "file", mimetype="text/markdown",
                                    binary_hash=self.document_hash)
            doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
            # 只遍历通过校验的语法树，不走 Docling 的原始 HTML 分支。
            self._iterate_elements(element=element, depth=0, doc=doc, visited=set(), creation_stack=[],
                                   list_ordered_flag_by_ref={}, list_last_item_by_ref={})
            self._close_table(doc)
            return doc

        def _iterate_elements(self, *, element, depth, doc, parent_item=None, **state):
            if isinstance(element, Table):
                self._close_table(doc)
                images = []
                cells = [
                    DoclingCell(text=_cell_text(cell, images), row_span=1, col_span=1,
                                start_row_offset_idx=row_index, end_row_offset_idx=row_index + 1,
                                start_col_offset_idx=column, end_col_offset_idx=column + 1,
                                column_header=row_index == 0)
                    for row_index, row in enumerate(element.children)
                    for column, cell in enumerate(row.children)
                ]
                table = doc.add_table(data=TableData(num_rows=len(element.children),
                    num_cols=element.num_of_cols, table_cells=cells), parent=parent_item)
                for image in images:
                    super()._iterate_elements(element=image, depth=depth + 1, doc=doc, parent_item=table, **state)
                return
            if _is_break(element):
                # 正文两侧由 Docling 分别生成文本；不把标签写进正文，也不生成空节点。
                return
            super()._iterate_elements(element=element, depth=depth, doc=doc, parent_item=parent_item, **state)

    return GuardedMarkdownBackend
