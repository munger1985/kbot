"""PDF 同栏续行、索引父段落及误合并边界的隔离回归。"""

from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from knowledge_algorithm_fixtures import (AllowedReader, CharacterTokenizer, allowed, node,
    parse_profile, projection_profile, provenance, run_for, source)
from knowledge_algorithms.contracts.content import ContentKind, Page, ProvenanceKind
from knowledge_algorithms.contracts.identity import Locator
from knowledge_algorithms.parsing.contracts import ExtractedRegion, ExtractionBatch
from knowledge_algorithms.parsing.engine import ParseEngine
from knowledge_algorithms.parsing.pdf_paragraphs import restore_pdf_paragraphs
from knowledge_algorithms.parsing.structure import order_and_structure
from knowledge_algorithms.projections.builder import ProjectionBuilder


PAGE = Page("page-2", 2, 595.3, 841.9, 0)
HEAD = "第九条 内部审计配备专职审计人员，审计人员应熟悉公司的经营业务和内"
TAIL = "部控制规范，具备开展审计活动所需的相关专业知识、经验和技能。"
HEAD_BOX = (0.192286, 0.245189, 0.846, 0.258645)
TAIL_BOX = (0.152212, 0.272899, 0.743, 0.286355)


def line(name, text, box, **kwargs):
    return node(name, text, **({"locators": (Locator(2, box),)} | kwargs))


def test_wrap_uses_realistic_indent_and_line_spacing_and_preserves_both_boxes():
    first = line("head", HEAD, HEAD_BOX)
    second = line("tail", TAIL, TAIL_BOX)
    third = line("next", "第十条 独立的条款。", (0.192286, 0.309844, 0.793597, 0.323357))
    restored, next_clause = restore_pdf_paragraphs((first, second, third), (PAGE,))
    assert restored.key == first.key
    assert restored.text == HEAD + TAIL
    assert restored.locators == first.locators + second.locators
    assert next_clause == replace(third, reading_order=1)


def test_plain_paragraph_needs_indentation_but_does_not_need_a_clause_number():
    first, second = line("head", HEAD.removeprefix("第九条 "), HEAD_BOX), line("tail", TAIL, TAIL_BOX)
    assert restore_pdf_paragraphs((first, second), (PAGE,))[0].text == first.text + TAIL
    aligned = replace(first, locators=(Locator(2, (TAIL_BOX[0], *HEAD_BOX[1:])),))
    assert len(restore_pdf_paragraphs((aligned, second), (PAGE,))) == 2


def test_same_line_with_several_source_boxes_remains_locatable():
    first = line("head", HEAD, HEAD_BOX, locators=(
        Locator(2, (0.192286, 0.245189, 0.250, 0.258645)),
        Locator(2, (0.255, 0.245189, 0.846, 0.258645)),
    ))
    second = line("tail", TAIL, TAIL_BOX)
    restored, = restore_pdf_paragraphs((first, second), (PAGE,))
    assert restored.locators == first.locators + second.locators


def test_short_list_end_does_not_create_a_phantom_column_or_keep_the_wrong_chapter():
    before = line("old-heading", "第二章 原章节", (0.25, 0.21, 0.75, 0.225),
                  kind=ContentKind.SECTION, heading_level=1)
    chapter = line("heading", "第三章 职责及权限", (0.414, 0.349, 0.586, 0.361),
                   kind=ContentKind.SECTION, heading_level=1)
    head = line("item-four", "（四） 对相关会计资料及经济活动的合法性、合规性和完整性进行",
                (0.1512, 0.5593, 0.8488, 0.6011), kind=ContentKind.LIST_ITEM)
    tail = line("item-four-end", "审计；", (0.1512, 0.6147, 0.2117, 0.6289), kind=ContentKind.LIST_ITEM)
    next_item = line("item-five", "（五） 督促问题的整改落实；",
                     (0.1915, 0.6517, 0.8466, 0.6659), kind=ContentKind.LIST_ITEM)
    structured, _ = order_and_structure((before, chapter, head, tail, next_item), parse_profile().layout)
    restored = restore_pdf_paragraphs(structured, (PAGE,))
    assert [item.key for item in restored] == [before.key, chapter.key, head.key, next_item.key]
    assert restored[2].text == head.text + tail.text
    assert restored[2].locators == head.locators + tail.locators
    assert restored[2].parent_key == chapter.key.object_key


def test_numbered_paragraph_accepts_an_indented_tail_mislabeled_as_list_item():
    first = line("head", HEAD, HEAD_BOX, kind=ContentKind.LIST_ITEM)
    second = line("tail", TAIL, TAIL_BOX, kind=ContentKind.LIST_ITEM)
    restored, = restore_pdf_paragraphs((first, second), (PAGE,))
    assert restored.text == HEAD + TAIL


@pytest.mark.parametrize("case", [
    "next_clause", "numbered_list", "different_column", "farther_down", "different_page",
    "different_parent", "different_provenance", "missing_box", "sentence_finished", "crossed_box", "protected",
])
def test_adjacent_nodes_are_not_sufficient_to_join(case):
    first, second = line("head", HEAD, HEAD_BOX), line("tail", TAIL, TAIL_BOX)
    pages, protected = (PAGE,), frozenset()
    if case == "next_clause":
        second = replace(second, text="第十条 其他规定。")
    elif case == "numbered_list":
        second = replace(second, text="（一）单独列项。")
    elif case == "different_column":
        second = replace(second, locators=(Locator(2, (.86, TAIL_BOX[1], .99, TAIL_BOX[3])),))
    elif case == "farther_down":
        second = replace(second, locators=(Locator(2, (.15, .31, .75, .325)),))
    elif case == "different_page":
        second = replace(second, locators=(Locator(3, TAIL_BOX),))
        pages += (replace(PAGE, local_key="page-3", page_no=3),)
    elif case == "different_parent":
        second = replace(second, parent_key="another-section")
    elif case == "different_provenance":
        second = replace(second, provenance=provenance(ProvenanceKind.OCR))
    elif case == "missing_box":
        second = replace(second, locators=(Locator(2),))
    elif case == "sentence_finished":
        first = replace(first, text=HEAD + "。")
    elif case == "crossed_box":
        second = replace(second, locators=(Locator(2, TAIL_BOX), Locator(2, (.15, .32, .75, .34))))
    elif case == "protected":
        protected = frozenset({second.key.object_key})
    assert len(restore_pdf_paragraphs((first, second), pages, protected_keys=protected)) == 2


@pytest.mark.asyncio
async def test_parse_engine_repairs_before_projection_and_keyword_crosses_the_wrap():
    reader = source("测试 PDF 提取事件")
    reader.descriptor = replace(reader.descriptor, mime_type="application/pdf")
    profile = parse_profile()
    header = line("section", "第二章 内部审计机构和内部审计人员",
                  (0.25, 0.1, 0.8, 0.116), kind=ContentKind.SECTION, heading_level=1)
    first, second = line("head", HEAD, HEAD_BOX), line("tail", TAIL, TAIL_BOX)

    class Adapter:
        revision = "fixture-native/v1"
        mime_types = frozenset({"application/pdf"})

        async def extract(self, stream, profile, run):
            async for _ in stream.chunks():
                pass
            yield ExtractionBatch((PAGE,), tuple(ExtractedRegion(n.native_ref, (n,), 1.0)
                                                  for n in (header, first, second)), ())

    document = AllowedReader(())
    parsed = await ParseEngine(document.binding, (Adapter(),)).parse(reader, profile, run_for(reader, profile))
    assert [n.kind for n in parsed.nodes] == [ContentKind.SECTION, ContentKind.PARAGRAPH]
    body = parsed.nodes[1]
    assert body.parent_key == header.key.object_key
    assert body.text == HEAD + TAIL
    document = AllowedReader(tuple(allowed(n) for n in parsed.nodes))
    batches = [batch async for batch in ProjectionBuilder(document.binding, CharacterTokenizer(), None).build(
        document, projection_profile())]
    units = [projected.unit for batch in batches for projected in batch.units if projected.unit.kind == ContentKind.PARAGRAPH]
    paragraph, = units
    assert "内部控制规范" in paragraph.text
    assert paragraph.source_key == body.key and paragraph.sources[0].source_key == body.key
    assert paragraph.locators == first.locators + second.locators
