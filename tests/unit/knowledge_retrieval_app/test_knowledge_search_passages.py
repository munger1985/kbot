from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from knowledge_algorithms.contracts.content import ContentKind, Provenance, ProvenanceKind
from knowledge_algorithms.contracts.identity import EvidenceKey, Locator, SourceSpan
from knowledge_algorithms.projections.contracts import AllowedContent, UnitSource
from knowledge_retrieval_app.application.knowledge_library import KnowledgeLibraryService
from knowledge_retrieval_app.application.knowledge_passages import PassageDocument, continues_clause


def node(index, text, *, y=0.1, x=0.1, page=2, heading=("第二章",), kind=ContentKind.PARAGRAPH, version="version"):
    key = EvidenceKey("owner", version, "parse", str(index))
    return AllowedContent(key, kind, text, (UnitSource(0, len(text), key, 0, len(text)),),
                          (Locator(page, (x, y, x + 0.3, y + 0.02)),),
                          Provenance(ProvenanceKind.NATIVE, "fixture", "revision", "sha"), heading)


class FixtureDocument(PassageDocument):
    def __init__(self, nodes):
        self.nodes = nodes
        super().__init__(store=None, max_bytes=65536, version={
            "view_hash": "hash", "mime_type": "application/pdf", "parse_run_id": "parse"},
            manifest={"view_hash": "hash", "view_ref": "view",
                      "items": [{"key": item.key.stable_key, "max_bytes": 100} for item in nodes]})

    async def read(self, index):
        return self.nodes[index]


def hit(item):
    return SimpleNamespace(key=replace(item.key, projection_ref="search"), mime_type="text/plain",
                           source_keys=(item.key,), source_spans=(SourceSpan(item.key.object_key, 0, len(item.text)),),
                           locators=item.locators, allowed_view_ref="view")


@pytest.mark.asyncio
@pytest.mark.parametrize("tail_first", [False, True])
async def test_misordered_short_list_end_displays_one_complete_item_and_keeps_its_match(tail_first):
    chapter = node(1, "第三章 职责及权限", y=.349, heading=("第三章",), kind=ContentKind.SECTION)
    head = node(2, "（四） 对相关会计资料及经济活动的合法性、合规性和完整性进行",
                heading=("第三章",), kind=ContentKind.LIST_ITEM)
    head = replace(head, locators=(Locator(2, (.1512, .5593, .8488, .6011)),))
    tail = node(3, "审计；", heading=("第二章",), kind=ContentKind.LIST_ITEM)
    tail = replace(tail, locators=(Locator(2, (.1512, .6147, .2117, .6289)),))
    next_item = node(4, "（五） 督促问题的整改落实；", y=.6517, heading=("第三章",),
                     kind=ContentKind.LIST_ITEM)
    document = FixtureDocument([tail, chapter, head, next_item])
    hit_head, hit_tail = hit(head), hit(tail)
    selected, other = (hit_tail, hit_head) if tail_first else (hit_head, hit_tail)
    key, display = await document.display(selected, selected.source_keys)
    assert (key, display) == await document.display(other, other.source_keys)
    assert key == head.key.stable_key
    assert display["title"] == "（四）" and display["text"].endswith("进行审计；")
    context = await document.context(hit_tail, [hit_tail], before=1, after=1)
    assert [block["key"] for block in context["blocks"]] == [
        chapter.key.stable_key, head.key.stable_key, tail.key.stable_key, next_item.key.stable_key,
    ]
    assert [part["text"] for block in context["blocks"] for part in block["segments"] if part["matched"]] == ["审计；"]


@pytest.mark.asyncio
async def test_present_uses_complete_paragraph_key_and_chapter_for_tail_only_hit(monkeypatch):
    version_id = str(uuid4())
    old = node(1, "第二章 原章节", y=.07, kind=ContentKind.SECTION, version=version_id)
    chapter = node(2, "第三章 职责及权限", y=.349, heading=("第三章",), kind=ContentKind.SECTION, version=version_id)
    head = node(3, "（四） 对相关会计资料及经济活动的合法性、合规性和完整性进行",
                heading=("第三章",), kind=ContentKind.LIST_ITEM, version=version_id)
    head = replace(head, locators=(Locator(2, (.1512, .5593, .8488, .6011)),))
    tail = node(4, "审计；", kind=ContentKind.LIST_ITEM, version=version_id)
    tail = replace(tail, locators=(Locator(2, (.1512, .6147, .2117, .6289)),))
    document = FixtureDocument([old, tail, chapter, head])
    document.version.update(version_id=version_id, document_id="document", title="制度")

    async def open_document(*_):
        return document

    async def versions(_access, refs):
        assert {str(ref) for ref in refs} == {version_id}
        return {version_id: document.version}

    async def headings(_access, keys):
        assert keys == {head.key}
        return {(head.key.parse_run_ref, head.key.object_key): chapter.key.object_key}

    async def previews(_access, _receipt, keys):
        assert keys == {item.key, image.key}
        return {item.key: tail.text}

    @asynccontextmanager
    async def uow():
        yield SimpleNamespace(knowledge_retrieval=SimpleNamespace(passage_versions=versions, passage_headings=headings))

    monkeypatch.setattr(PassageDocument, "open", staticmethod(open_document))
    service = SimpleNamespace(uow_factory=uow, retrieval=SimpleNamespace(
        store=None, limits=SimpleNamespace(max_object_bytes=65536), presentation_previews=previews))
    item = hit(tail)
    item.kind, item.byte_count = ContentKind.LIST_ITEM, len(tail.text.encode())
    image = hit(node(6, "图示", kind=ContentKind.FIGURE, version=version_id))
    image.kind, image.mime_type, image.byte_count = ContentKind.FIGURE, "image/png", 64
    linked, candidate = hit(head), hit(node(5, "未展示候选", version=str(uuid4())))
    linked.kind, linked.byte_count = ContentKind.LIST_ITEM, len(head.text.encode())
    linked_edge = SimpleNamespace(relation_type=SimpleNamespace(value="PART_OF"),
                                  source_key=item.key, target_key=linked.key)
    group = SimpleNamespace(anchor_key=item.key, linked_keys=(linked.key,), manifest_status="COMPLETE",
                            continuation=None, incomplete_reason=None, relations=(linked_edge,))
    receipt = SimpleNamespace(retrieval_ref="fixture", evidence_set=SimpleNamespace(
        primary_hits=[SimpleNamespace(evidence_key=item.key, rank=1), SimpleNamespace(evidence_key=image.key, rank=2)],
        evidence=[item, linked, candidate, image],
        status="PARTIAL", reason=None, scope_exhausted=False, mode=SimpleNamespace(value="RELEVANCE"),
        groups=[group, SimpleNamespace(anchor_key=candidate.key)], channel_status=[]))
    result = await KnowledgeLibraryService.present(service, object(), receipt)
    passage, related, picture = result["items"]
    assert passage["passage_key"] == head.key.stable_key
    assert passage["chapter"] == chapter.text and passage["passage_display"]["chapter_key"] == chapter.key.stable_key
    assert passage["passage_display"]["title"] == "（四）"
    assert passage["passage_display"]["text"].endswith("进行审计；")
    assert passage["text"] == "审计；"  # 保留底层真实命中，列表展示完整条款。
    assert related["text"] is None and related["passage_display"] is None
    assert related["content_ref"] == {"retrieval_ref": "fixture", "key": linked.key.stable_key}
    assert [value["anchor_key"] for value in result["linked"]] == [item.key.stable_key]
    assert picture["mime_type"] == "image/png" and picture["passage_key"] is None
    assert picture["content_ref"]["key"] == image.key.stable_key


@pytest.mark.asyncio
@pytest.mark.parametrize(("mime", "locators", "expected"), [
    ("application/pdf", (Locator(6),), 6),
    ("image/png", (Locator(1),), None),
    ("application/pdf", (Locator(logical_ref="figure"),), None),
])
async def test_image_context_resolves_pinned_pdf_page_without_loading_passages(mime, locators, expected):
    version_id = uuid4()
    access = object()
    item = SimpleNamespace(key=EvidenceKey("owner", str(version_id), "parse", "figure"),
                           mime_type="image/png", locators=locators)
    calls = []

    async def versions(seen_access, refs):
        assert seen_access is access and refs == {version_id}
        calls.append("version")
        return {str(version_id): {"mime_type": mime}}

    @asynccontextmanager
    async def uow():
        yield SimpleNamespace(knowledge_retrieval=SimpleNamespace(passage_versions=versions))

    async def reauthorize():
        calls.append("authorize")

    service = SimpleNamespace(uow_factory=uow)
    context = await KnowledgeLibraryService._passage_context(
        service, access, item, (), "", before=2, after=2, reauthorize=reauthorize)
    assert context == {"blocks": [], "has_before": False, "has_after": False, "pdf_page": expected}
    assert calls == (["version", "authorize"] if locators[0].page_no else ["authorize"])


@pytest.mark.asyncio
async def test_split_clause_joins_in_reading_order_even_if_continuation_is_ranked_first():
    first = node(1, "第九条 内部审计人员应熟悉公司的经营业务和内", y=0.26)
    second = node(2, "部控制规范，具备相关专业知识。", y=0.285)
    stranded = node(3, "的舞弊行为；", y=0.32)
    document = FixtureDocument([first, second, stranded])
    after, before = hit(second), hit(first)
    key, display = await document.display(after, after.source_keys)
    assert (key, display) == await document.display(before, before.source_keys)
    assert key == first.key.stable_key
    assert display == {"order": 0, "title": "第九条", "kind": "PARAGRAPH", "pages": [2],
                       "text": "内部审计人员应熟悉公司的经营业务和内部控制规范，具备相关专业知识。"}
    _, other = await document.display(hit(stranded), (stranded.key,))
    assert other["title"] is None
    context = await document.context(after, [after, before], before=0, after=1)
    assert context["pdf_page"] == 2
    assert [block["anchor"] for block in context["blocks"]] == [True, True, False]
    assert [block["segments"][0]["matched"] for block in context["blocks"]] == [True, True, False]


@pytest.mark.parametrize("changes", [
    {"text": "第十条 独立的条款"}, {"text": "（一）单独列项"}, {"heading": ("第三章",)},
    {"page": 3}, {"x": 0.55}, {"y": 0.4}, {"kind": ContentKind.TABLE},
])
def test_clause_does_not_cross_missing_or_conflicting_layout(changes):
    right = {"y": 0.126, "text": "部控制规范"}
    right.update(changes)
    assert not continues_clause(node(1, "第九条 业务和内"), node(2, **right))


@pytest.mark.asyncio
async def test_restored_paragraph_uses_one_display_key_and_marks_the_cross_line_word():
    text = "第九条 审计人员应熟悉公司的经营业务和内部控制规范。"
    item = node(1, text)
    item = replace(item, locators=(Locator(2, (0.19, 0.245, 0.846, 0.259)),
                                  Locator(2, (0.152, 0.273, 0.743, 0.286))))
    document = FixtureDocument([item])
    first = hit(item)
    second = hit(item)
    second.key = replace(first.key, object_key="another-search-unit")
    begin = text.index("内部控制规范")
    second.source_spans = (SourceSpan(item.key.object_key, begin, len(text)),)
    key, display = await document.display(first, first.source_keys)
    assert key == item.key.stable_key
    assert (key, display) == await document.display(second, second.source_keys)
    assert display["text"] == text.removeprefix("第九条 ")
    context = await document.context(second, [second, first], before=0, after=0, query_text="内部控制规范")
    assert context["pdf_page"] == 2
    block, = context["blocks"]
    assert block["anchor"] and len(block["locators"]) == 2
    assert [segment["text"] for segment in block["segments"] if segment["matched"]] == ["内部控制规范"]


@pytest.mark.asyncio
async def test_view_and_run_must_match():
    item = node(1, "第九条 已发布的条款")
    document = FixtureDocument([item])
    value = hit(item)
    value.allowed_view_ref = "another"
    assert await document.display(value, (item.key,)) == (None, None)
    assert await document.context(value, [], before=0, after=0) is None
