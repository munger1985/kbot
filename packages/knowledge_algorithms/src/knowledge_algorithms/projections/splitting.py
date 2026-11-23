"""不改写原文的有界切片规划；所有返回区间都是原对象的字符偏移。"""

import re
import unicodedata

from ..contracts.content import ContentKind
from ..contracts.errors import AlgorithmError, ErrorCode, require
from .tables import markdown_cell_spans, markdown_table_spans


REVISION = "source-bound-split/v2"

_BREAKS = tuple(re.compile(value) for value in (
    r"\n[ \t]*\n+", r"[。！？!?]+|(?<!\d)\.(?:\s+|$)|\n",
    r"[，、；：;:]+|,(?!\d)", r"[ \t]+",
))


def _prefix_end(start, stop, fits):
    """指数探测后缩小区间，不反复复制尚未处理的整段文本。"""
    step = 1
    good = start
    while True:
        candidate = min(start + step, stop)
        if not fits(start, candidate):
            bad = candidate
            break
        good = candidate
        if candidate == stop:
            return stop
        step *= 2
    while good + 1 < bad:
        mid = (good + bad) // 2
        if fits(start, mid):
            good = mid
        else:
            bad = mid
    return good


def _cluster_boundary(text, pos):
    return 0 < pos < len(text) and (
        unicodedata.category(text[pos]).startswith("M")
        or text[pos] == "\u200d" or text[pos - 1] == "\u200d"
        or "\U0001f3fb" <= text[pos] <= "\U0001f3ff"
        or (text[pos - 1] == "\r" and text[pos] == "\n")
    )


def text_spans(text, start, stop, fits, *, overlap=True):
    """先找语言边界，最后按字符切；少量重叠必须给新内容让出空间。"""
    cursor = start
    while cursor < stop:
        limit = _prefix_end(cursor, stop, fits)
        if limit == cursor:
            raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "单个字符超过索引输入上限", stage="split")
        end = limit
        natural = False
        if limit < stop:
            window_start = cursor + max(1, (limit - cursor) // 3)
            for pattern in _BREAKS:
                last = None
                for match in pattern.finditer(text, window_start, limit):
                    last = match.end()
                if last is not None and fits(cursor, last):
                    end, natural = last, True
                    break
            # 极端组合字符可能比预算更大；仍以可解码的 Unicode 字符保证前进。
            while end > cursor + 1 and _cluster_boundary(text, end):
                end -= 1
        yield cursor, end
        if end == stop:
            break
        next_start = end
        if overlap and not natural:
            back = min(16, (end - cursor) // 8)
            candidate = end - back
            while candidate < end and _cluster_boundary(text, candidate):
                candidate += 1
            if candidate > cursor and fits(candidate, end + 1):
                next_start = candidate
        cursor = next_start


def plan_chunks(content, budget, tokenizer):
    """正文和附带的标题分开计量；索引片段只引用连续的原文。"""
    text = content.text

    def cost(value):
        count = tokenizer.count(value)
        require(type(count) is int and count >= 0, "tokenizer 返回无效成本")
        return count

    def context(*labels):
        value = " / ".join(filter(None, labels))
        return value if cost(value + "\n") <= budget // 4 else ""

    heading = ""
    for label in reversed(content.heading_path):
        heading = context(label, heading) or heading

    def fits(ctx, a, b):
        return cost("\n".join(filter(None, (ctx, text[a:b])))) <= budget

    def fragment(a, b, ctx):
        if fits(ctx, a, b):
            yield a, b, ctx
            return
        # 不能为了保留标题使可容纳的正文再切一次；需要时才为上下文预留空间。
        chosen = ctx if ctx and fits(ctx, a, min(a + 1, b)) else ""
        if fits("", a, b):
            yield a, b, ""
            return
        for left, right in text_spans(text, a, b, lambda i, j: fits("", i, j)):
            yield left, right, chosen if fits(chosen, left, right) else ""

    if not text:
        if content.content_handle:
            yield 0, 0, heading
        return
    if fits("", 0, len(text)):
        yield from fragment(0, len(text), heading)
        return
    if content.kind in {ContentKind.CODE, ContentKind.FORMULA}:
        raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "不可拆分的代码或公式超过索引输入上限", stage="split")
    if content.kind != ContentKind.TABLE:
        yield from fragment(0, len(text), heading)
        return

    header, rows = (content.table_header, content.table_rows) if content.table_rows else markdown_table_spans(text)
    if not header or not rows:
        yield from fragment(0, len(text), heading)
        return
    line_end = text.find("\n", *header)
    labels = markdown_cell_spans(text, (header[0], line_end)) if line_end > header[0] else ()
    def label_text(spans):
        return " / ".join(text[a:b].removeprefix("|").strip() for a, b in spans)
    table_context = context(heading, label_text(labels)) or heading
    yield from fragment(0, rows[0][0], heading)
    group = rows[0][0]
    for index, row in enumerate(rows):
        start = row[0]
        tail = rows[index + 1][0] if index + 1 < len(rows) else len(text)
        if fits(table_context, group, tail):
            continue
        if group < start:
            yield group, start, table_context
        group = start
        if fits("", start, tail):
            if not fits(table_context, start, tail):
                yield from fragment(start, tail, heading)
                group = tail
            continue
        cells = markdown_cell_spans(text, row)
        if not labels or len(cells) != len(labels):
            yield from fragment(start, tail, heading)
        else:
            edges = (start, *(cell[0] for cell in cells[1:]), tail)
            first = 0
            for col in range(len(cells)):
                a, b = edges[col], edges[col + 1]
                if fits("", edges[first], b):
                    continue
                if first < col:
                    yield from fragment(edges[first], a, context(heading, label_text(labels[first:col])) or heading)
                first = col
                if not fits("", a, b):
                    yield from fragment(a, b, context(heading, label_text(labels[col:col + 1])) or heading)
                    first = col + 1
            if first < len(cells):
                yield from fragment(edges[first], tail, context(heading, label_text(labels[first:])) or heading)
        group = tail
    if group < len(text):
        yield group, len(text), table_context
