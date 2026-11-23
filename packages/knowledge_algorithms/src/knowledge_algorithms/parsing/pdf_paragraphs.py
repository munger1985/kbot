"""依据 PDF 版面证据还原折行，保留原页上的全部定位框。"""

from dataclasses import dataclass, replace
import re

from ..contracts.content import ContentKind, ContentNode, Page, ProvenanceKind


REVISION = "pdf-paragraphs/v2"
CLAUSE_START = re.compile(r"^\s*(第\s*[零〇一二三四五六七八九十百千万\d]+\s*条)(?=\s|[^条款]|$)")
LIST_START = re.compile(r"^\s*([（(][零〇一二三四五六七八九十百\d]+[）)]|[一二三四五六七八九十]+、)")
STRUCTURE_START = re.compile(
    r"^\s*(?:第\s*[零〇一二三四五六七八九十百千万\d]+\s*[章节条款]"
    r"|[（(][一二三四五六七八九十\d]+[）)]|\d+(?:\.\d+)*(?:[.、．)]|\s)|[一二三四五六七八九十]+、)"
)
SENTENCE_END = re.compile(r"[。！？；.!?;：:][”’）)】」』\s]*$")
_ASCII_WORD_END = re.compile(r"[A-Za-z0-9,]$")
_ASCII_WORD_START = re.compile(r"^[A-Za-z0-9]")
_MAX_LINES = 8
_MAX_BOXES_PER_LINE = 16
PDF_TEXT_KINDS = frozenset({ContentKind.PARAGRAPH, ContentKind.LIST_ITEM})


@dataclass(frozen=True, slots=True)
class PdfTextBox:
    page_no: int
    left: float
    top: float
    right: float
    bottom: float

    @property
    def height(self) -> float:
        return self.bottom - self.top


def pdf_text_box(locators) -> PdfTextBox | None:
    if not 1 <= len(locators) <= _MAX_BOXES_PER_LINE:
        return None
    page_no = locators[0].page_no
    if page_no is None or any(loc.page_no != page_no or loc.bbox is None for loc in locators):
        return None
    boxes = tuple(loc.bbox for loc in locators)
    return PdfTextBox(page_no, min(b[0] for b in boxes), min(b[1] for b in boxes),
                      max(b[2] for b in boxes), max(b[3] for b in boxes))


def continues_pdf_text(head, previous, current, *, aspect: float = 1.5) -> bool:
    """可为误标成列表项的末行续接，亦接受前一个节点已覆盖两三行。"""
    if (any(node.kind not in PDF_TEXT_KINDS for node in (head, previous, current))
            or previous.provenance != current.provenance
            or current.provenance.kind not in {ProvenanceKind.NATIVE, ProvenanceKind.OCR}
            or len(current.text.strip().splitlines()) != 1 or not previous.text.strip()
            or SENTENCE_END.search(previous.text) or STRUCTURE_START.match(current.text)):
        return False
    first, left, right = (pdf_text_box(node.locators) for node in (head, previous, current))
    if (first is None or left is None or right is None or left.page_no != right.page_no
            or first.page_no != left.page_no):
        return False
    line_height = min(left.height, right.height)
    numbered = bool(STRUCTURE_START.match(head.text))
    if (line_height <= 0 or right.top <= left.top or right.height > 1.65 * line_height
            or left.height > (7 if numbered else 1.65) * line_height
            or not -0.25 * line_height <= right.top - left.bottom <= 1.25 * line_height):
        return False
    # 续行可以很短；以左边缘、真实重叠和上一行接近整栏宽度为准。
    em = line_height * aspect
    previous_width = left.right - left.left
    current_width = right.right - right.left
    overlap = min(left.right, right.right) - max(left.left, right.left)
    indent = left.left - right.left
    overlap_ratio = 0.55 if numbered and head is previous and indent > 0 else 0.75
    if (previous_width < 8 * em or overlap < overlap_ratio * min(previous_width, current_width)
            or left.right < first.right - 2 * em or right.right > first.right + 2 * em):
        return False
    if head is previous:
        minimum_indent = -0.65 if numbered else 0.75
        return minimum_indent * em <= indent <= 3 * em
    return abs(indent) <= 0.65 * em


def join_pdf_lines(parts) -> str:
    """去掉排版折行；相邻英文单词之间保留词边界。"""
    joined = ""
    for part in parts:
        text = part.strip()
        if _ASCII_WORD_END.search(joined) and _ASCII_WORD_START.match(text):
            joined += " "
        joined += text
    return joined


def restore_pdf_paragraphs(nodes: tuple[ContentNode, ...], pages: tuple[Page, ...], *,
                           protected_keys: frozenset[str] = frozenset()) -> tuple[ContentNode, ...]:
    """仅合并有证据的相邻续行；被显式引用的节点不得作为续行移除。"""
    page_by_number = {page.page_no: page for page in pages}
    parent_keys = {node.parent_key for node in nodes} | protected_keys
    paragraphs = []
    position = 0
    while position < len(nodes):
        node = nodes[position]
        head = pdf_text_box(node.locators)
        position += 1
        if head is None or head.page_no not in page_by_number or node.key.object_key in protected_keys:
            paragraphs.append(node)
            continue
        page = page_by_number[head.page_no]
        lines = [node]
        while position < len(nodes) and len(lines) < _MAX_LINES:
            following = nodes[position]
            if (following.key.object_key in parent_keys or following.parent_key != node.parent_key
                    or following.numbering_scope != node.numbering_scope):
                break
            if not continues_pdf_text(node, lines[-1], following, aspect=page.height / page.width):
                break
            lines.append(following)
            position += 1
        if len(lines) > 1:
            node = replace(node, text=join_pdf_lines(line.text for line in lines),
                           locators=tuple(loc for line in lines for loc in line.locators))
        paragraphs.append(node)
    return tuple(replace(node, reading_order=order) if node.reading_order != order else node
                 for order, node in enumerate(paragraphs))
