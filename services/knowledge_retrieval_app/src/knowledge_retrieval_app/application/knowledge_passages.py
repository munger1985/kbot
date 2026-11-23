"""用已发布允许视图把检索片段映射回原段落，按规范阅读顺序展开。"""

import json
import re

from knowledge_algorithms.contracts.content import ContentKind
from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.wire import decode, encode
from knowledge_algorithms.parsing.pdf_paragraphs import (
    CLAUSE_START, LIST_START, PDF_TEXT_KINDS, SENTENCE_END, continues_pdf_text, join_pdf_lines, pdf_text_box,
)
from knowledge_algorithms.projections.contracts import AllowedContent


CONTEXT_KINDS = {ContentKind.SECTION, ContentKind.PARAGRAPH, ContentKind.LIST_ITEM, ContentKind.TABLE,
                 ContentKind.CODE, ContentKind.FORMULA, ContentKind.FOOTNOTE, ContentKind.CAPTION}
MAX_NEIGHBORS = 10
MAX_SCAN_PER_SIDE = 64
MAX_CLAUSE_NODES = 8
PDF_WINDOW = 24
MAX_DISPLAY_NODE_BYTES = 65536


def continues_clause(left, right, *, head=None, reordered=False):
    head = head or left
    return (reordered or head.heading_path == right.heading_path) and continues_pdf_text(head, left, right)


def paragraph_marker(text):
    return CLAUSE_START.match(text) or LIST_START.match(text)


def source_keys(item):
    if not item.mime_type.startswith("text/"):
        return ()
    spans = {span.node_key for span in item.source_spans}
    return tuple(sorted((key for key in item.source_keys if not spans or key.object_key in spans),
                        key=lambda key: key.stable_key))


def marked_text(content, hits, query_text=""):
    """把规范源字符跨度映射到允许文本；不高亮被过滤掉的字符。"""
    spans = [(span.char_start, span.char_end) for hit in hits for span in hit.source_spans
             if span.node_key == content.key.object_key]
    whole = any(hit.key == content.key for hit in hits)
    ranges = []
    for source in content.sources:
        if source.source_key != content.key:
            continue
        for start, end in spans:
            begin, finish = max(start, source.source_start), min(end, source.source_end)
            if begin < finish:
                ranges.append((source.unit_start + begin - source.source_start,
                               source.unit_start + finish - source.source_start))
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    if whole:
        merged = [(0, len(content.text))]
    # 查询原句确实出现在命中范围时高亮原句；语义召回则标出有证据的来源跨度。
    if query_text:
        pattern = re.compile(re.escape(query_text), re.IGNORECASE)
        exact = [match.span() for start, end in merged for match in pattern.finditer(content.text, start, end)]
        if exact:
            merged = exact
    result, offset = [], 0
    for start, end in merged:
        if offset < start:
            result.append({"text": content.text[offset:start], "matched": False})
        result.append({"text": content.text[start:end], "matched": True})
        offset = end
    if offset < len(content.text):
        result.append({"text": content.text[offset:], "matched": False})
    return result


class PassageDocument:
    def __init__(self, *, store, version, manifest, max_bytes):
        self.store, self.version, self.max_bytes = store, version, max_bytes
        require(manifest["view_hash"] == version["view_hash"], "段落允许视图与发布快照不一致")
        self.entries = manifest["items"]
        self.positions = {entry["key"]: index for index, entry in enumerate(self.entries)}
        self.view_ref = manifest["view_ref"]
        self._display_cache = {}
        self._node_cache = {}

    @classmethod
    async def open(cls, store, version, max_bytes):
        raw = await store.read(version["manifest"], max_bytes=max_bytes)
        return cls(store=store, version=version, manifest=json.loads(raw), max_bytes=max_bytes)

    async def read(self, index):
        entry = self.entries[index]
        raw = await self.store.read(entry["handle"], max_bytes=min(entry["max_bytes"], self.max_bytes))
        value = decode(AllowedContent, json.loads(raw))
        require(value.key.stable_key == entry["key"] and value.key.document_version_ref == self.version["version_id"]
                and value.key.parse_run_ref == self.version["parse_run_id"], "段落来源与发布版本不一致")
        return value

    async def display_node(self, index):
        if not 0 <= index < len(self.entries) or self.entries[index]["max_bytes"] > MAX_DISPLAY_NODE_BYTES:
            return None
        if index not in self._node_cache:
            self._node_cache[index] = await self.read(index)
        return self._node_cache[index]

    async def pdf_window(self, index, box):
        """只在已发布允许视图的有限同页范围内按版面找回被错排的续行。"""
        candidates = []
        for position in range(max(0, index - PDF_WINDOW), min(len(self.entries), index + PDF_WINDOW + 1)):
            node = await self.display_node(position)
            located = pdf_text_box(node.locators) if node is not None else None
            if located is not None and located.page_no == box.page_no:
                # 不丢弃标题、图表等结构断点。同页异栏先保守地视为不可合并。
                candidates.append((located.top, located.left, position, node))
        return [(position, node) for _, _, position, node in sorted(candidates)]

    async def reading_span(self, origin):
        """以明确的条编号为锚，从允许视图中查找有版面证据的续行。"""
        index = self.positions.get(origin.stable_key)
        if index is None:
            return ()
        if index in self._display_cache:
            return self._display_cache[index]
        first = await self.display_node(index)
        if first is None:
            return ()
        span = ((index, first),)
        if self.version["mime_type"] == "application/pdf" and first.kind in PDF_TEXT_KINDS:
            box = pdf_text_box(first.locators)
            if box is not None and not (paragraph_marker(first.text) and SENTENCE_END.search(first.text)):
                window = await self.pdf_window(index, box)
                target = next(i for i, (position, _) in enumerate(window) if position == index)
                for start in range(target, max(-1, target - MAX_CLAUSE_NODES), -1):
                    ordinal, head = window[start]
                    if head.kind not in PDF_TEXT_KINDS or not paragraph_marker(head.text):
                        continue
                    group = [(ordinal, head)]
                    for position, following in window[start + 1:start + MAX_CLAUSE_NODES]:
                        previous_position, previous = group[-1]
                        # 旧版把左侧短行错排到前面的章节。只有物理相邻且顺序确实倒置才允许归回正文。
                        if not continues_clause(previous, following, head=head, reordered=position < previous_position):
                            break
                        group.append((position, following))
                    if any(position == index for position, _ in group):
                        span = tuple(group)
                        break
        for position, _ in span:
            self._display_cache[position] = span
        return span

    async def display(self, hit, origins):
        if len(origins) != 1 or hit.allowed_view_ref != self.view_ref or hit.key.parse_run_ref != self.version["parse_run_id"]:
            return None, None
        span = await self.reading_span(origins[0])
        if not span:
            return None, None
        order, first = span[0]
        title = paragraph_marker(first.text) if first.kind in PDF_TEXT_KINDS else None
        parts = [content for _, content in span]
        text = join_pdf_lines(content.text for content in parts) if first.kind in {
            ContentKind.PARAGRAPH, ContentKind.LIST_ITEM} else None
        if title and text:
            text = text[len(title[0].strip()):].lstrip()
        return first.key.stable_key, {"order": order, "kind": first.kind.value,
            "title": re.sub(r"\s+", "", title[1]) if title else None, "text": text[:4000] if text else None,
            "pages": sorted({loc.page_no for content in parts for loc in content.locators if loc.page_no})}

    async def neighbors(self, boundary, step, count):
        segments = []
        index = boundary + step
        for _ in range(MAX_SCAN_PER_SIDE):
            if not 0 <= index < len(self.entries):
                break
            value = await self.read(index)
            if value.text.strip() and value.kind in CONTEXT_KINDS:
                if len(segments) == count:
                    return segments, True
                segments.append((index, value))
            index += step
        return segments, False

    async def context(self, hit, other_hits, *, before, after, query_text=""):
        if hit.allowed_view_ref != self.view_ref or hit.key.parse_run_ref != self.version["parse_run_id"]:
            return None
        anchor_keys = source_keys(hit)
        if not anchor_keys or any(key.stable_key not in self.positions for key in anchor_keys):
            return None
        indices = sorted(self.positions[key.stable_key] for key in anchor_keys)
        # 极端跨度只展示首尾之间已证实的来源，不扩读整份文档。
        require(len(indices) <= MAX_NEIGHBORS, "命中段落超过上下文读取预算")
        span = await self.reading_span(anchor_keys[0]) if len(anchor_keys) == 1 else ()
        anchors = list(span) or [(index, await self.read(index)) for index in indices]
        anchors = [(index, content) for index, content in anchors if content.text.strip()]
        if not anchors:
            return None
        if len(span) > 1:
            window = await self.pdf_window(anchors[0][0], pdf_text_box(anchors[0][1].locators))
            positions = {position: offset for offset, (position, _) in enumerate(window)}
            if all(position in positions for position, _ in anchors):
                start_at, end_at = positions[anchors[0][0]], positions[anchors[-1][0]]
                preceding = [(i, node) for i, node in window[:start_at] if node.text.strip() and node.kind in CONTEXT_KINDS]
                following = [(i, node) for i, node in window[end_at + 1:] if node.text.strip() and node.kind in CONTEXT_KINDS]
                more_before, more_after = len(preceding) > before, len(following) > after
                blocks = (preceding[-before:] if before else []) + anchors + following[:after]
            else:
                blocks, more_before, more_after = anchors, False, False
        else:
            start, more_before = await self.neighbors(anchors[0][0], -1, before)
            end, more_after = await self.neighbors(anchors[-1][0], 1, after)
            blocks = sorted(start + anchors + end, key=lambda pair: pair[0])
        selected = {index for index, _ in anchors}
        selected_keys = {content.key for _, content in anchors}
        hits = [item for item in other_hits if item.allowed_view_ref == self.view_ref
                and selected_keys.intersection(source_keys(item))]
        size = sum(len(content.text.encode("utf-8")) for _, content in blocks)
        require(size <= self.max_bytes, "段落上下文超过读取预算")
        page = next((loc.page_no for loc in hit.locators if loc.page_no), None)
        return {"blocks": [{
            "key": content.key.stable_key, "kind": content.kind.value, "anchor": index in selected,
            "locators": encode(content.locators),
            "segments": marked_text(content, hits if index in selected else (), query_text),
        } for index, content in blocks],
            "has_before": before < MAX_NEIGHBORS and more_before,
            "has_after": after < MAX_NEIGHBORS and more_after,
            "pdf_page": page if self.version["mime_type"] == "application/pdf" else None}
