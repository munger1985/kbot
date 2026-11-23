"""按 PDF 原页字形间距修复英文词边界，不改动坐标或补写正文。"""

from bisect import bisect_left, bisect_right
from contextlib import closing
from dataclasses import dataclass, replace
import math
import re
from statistics import median
import unicodedata

from knowledge_algorithms.contracts.content import ContentKind, ProvenanceKind


REVISION = "pdf-word-spacing/v1"
_TEXT_KINDS = frozenset({ContentKind.SECTION, ContentKind.PARAGRAPH, ContentKind.LIST_ITEM,
                         ContentKind.CAPTION, ContentKind.FOOTNOTE, ContentKind.HEADER, ContentKind.FOOTER})
_LATIN = re.compile(r"[A-Za-z]+")
_DASHES = str.maketrans({chr(code): "-" for code in range(0x2010, 0x2016)})
_MAX_PAGE_GLYPHS = 60_000
_MAX_NODE_GLYPHS = 8_000
_MAX_LOCATORS = 16


@dataclass(frozen=True, slots=True)
class Glyph:
    index: int
    char: str
    left: float
    bottom: float
    right: float
    top: float
    em: float
    explicit_space: bool

    @property
    def y(self):
        return (self.bottom + self.top) * 0.5


def _fragmented(text):
    if not text or len(text) > _MAX_NODE_GLYPHS:
        return False
    tokens = _LATIN.findall(text)
    singles = sum(len(word) == 1 for word in tokens)
    return (singles >= 6 and singles * 2 >= len(tokens) and sum(map(len, tokens)) >= 12
            and all(char.isascii() or not char.isalpha() for char in text))


def _signature(text):
    # 比对时允许两种提取器把原页横线写作不同 Unicode 横线；发布时仍使用原页字符。
    return "".join(unicodedata.normalize("NFKC", text).translate(_DASHES).split())


def _line_break(left, right):
    return abs(right.y - left.y) > 0.58 * min(left.em, right.em)


def _gap(left, right):
    return (right.left - left.right) / min(left.em, right.em)


def _restore_text(original, glyphs):
    if not 2 <= len(glyphs) <= _MAX_NODE_GLYPHS or _signature(original) != _signature("".join(g.char for g in glyphs)):
        return None
    pairs = tuple(zip(glyphs, glyphs[1:]))
    same_line = tuple((left, right) for left, right in pairs if not _line_break(left, right))
    # 字形顺序倒退一般意味着多栏或竖排。不能仅从包围框猜测该类正文的词边界。
    if any(right.left < left.left - 0.25 * left.em for left, right in same_line):
        return None
    gaps = [_gap(left, right) for left, right in same_line if not right.explicit_space
            and left.char.isascii() and right.char.isascii() and left.char.isalpha() and right.char.isalpha()]
    if len(gaps) < 8:
        return None
    # 使用点数归一化的字形间距，并要求有明确的小字距/大字距两组证据。
    threshold = max(0.18, median(gaps) + 0.12)
    if sum(gap >= threshold for gap in gaps) < 2:
        return None
    parts = [glyphs[0].char]
    for left, right in pairs:
        if right.explicit_space or _line_break(left, right) or _gap(left, right) >= threshold:
            parts.append(" ")
        parts.append(right.char)
    repaired = "".join(parts)
    words = _LATIN.findall(repaired)
    if _fragmented(repaired) or sum(len(word) >= 2 for word in words) < 3:
        return None
    return repaired


def _page_glyphs(pdf, page_no):
    # PDFium 已是 Docling 的 PDF 运行依赖；这里只读取字符和坐标，不渲染、不访问网络。
    from pypdfium2 import PdfiumError
    from pypdfium2 import raw

    try:
        with closing(pdf[page_no - 1]) as page:
            if page.get_rotation() != 0:
                return None
            bounds = page.get_bbox()
            with closing(page.get_textpage()) as text:
                count = text.count_chars()
                if not 0 < count <= _MAX_PAGE_GLYPHS:
                    return None
                glyphs = []
                explicit_space = False
                for index in range(count):
                    point = raw.FPDFText_GetUnicode(text.raw, index)
                    if not 0 < point <= 0x10ffff:
                        continue
                    char = chr(point)
                    if char.isspace():
                        if char not in "\r\n" and raw.FPDFText_IsGenerated(text.raw, index) == 0:
                            explicit_space = True
                        continue
                    if not char.isprintable():
                        continue
                    box = text.get_charbox(index)
                    em = raw.FPDFText_GetFontSize(text.raw, index)
                    if em > 0 and math.isfinite(em) and all(math.isfinite(coord) for coord in box):
                        glyphs.append(Glyph(index, char, *box, em, explicit_space))
                    explicit_space = False
    except PdfiumError:
        # 二次提取是保守校验；缺乏可核对证据时沿用原生转换结果。
        return None
    glyphs.sort(key=lambda glyph: glyph.y)
    return bounds, glyphs, [glyph.y for glyph in glyphs]


def _in_regions(page, locators):
    (left, bottom, right, top), glyphs, heights = page
    width, height = right - left, top - bottom
    boxes = tuple((left + l * width, top - b * height, left + r * width, top - t * height)
                  for l, t, r, b in (locator.bbox for locator in locators))
    selected = {}
    for l, b, r, t in boxes:
        for offset in range(bisect_left(heights, b), bisect_right(heights, t)):
            glyph = glyphs[offset]
            if l <= (glyph.left + glyph.right) * 0.5 <= r:
                selected[glyph.index] = glyph
                if len(selected) > _MAX_NODE_GLYPHS:
                    return ()
    return tuple(selected[index] for index in sorted(selected))


def restore_pdf_word_spacing(pdf_bytes, regions):
    """单页内用原生字形核验坏空格；保留文档结构、原框、图片与其他格式文本。"""
    candidates = {}
    for position, region in enumerate(regions):
        if len(region.nodes) != 1:
            continue
        node = region.nodes[0]
        if (node.kind not in _TEXT_KINDS or node.provenance.kind != ProvenanceKind.NATIVE
                or not _fragmented(node.text) or not 1 <= len(node.locators) <= _MAX_LOCATORS):
            continue
        page_no = node.locators[0].page_no
        if page_no is not None and all(loc.page_no == page_no and loc.bbox is not None for loc in node.locators):
            candidates.setdefault(page_no, []).append((position, node))
    if not candidates:
        return regions

    from pypdfium2 import PdfDocument, PdfiumError

    try:
        pdf = PdfDocument(pdf_bytes)
    except PdfiumError:
        return regions
    with pdf:
        for page_no, nodes in candidates.items():
            if not 1 <= page_no <= len(pdf):
                continue
            page = _page_glyphs(pdf, page_no)
            if page is None:
                continue
            for position, node in nodes:
                repaired = _restore_text(node.text, _in_regions(page, node.locators))
                if repaired is not None:
                    regions[position] = replace(regions[position], nodes=(replace(node, text=repaired),))
    return regions
