"""直接检索历史使用隔离的内存仓储和临时对象；不连接开发数据库。"""

from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from knowledge_algorithms.contracts.content import ContentKind, Provenance, ProvenanceKind
from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require
from knowledge_algorithms.contracts.identity import EvidenceKey, Locator, RetrievalSnapshot
from knowledge_algorithms.contracts.query import QueryMode
from knowledge_algorithms.contracts.retrieval import EvidenceItem, EvidenceSet, PrimaryHit, QueryPlan
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.references.relations import LinkedManifest
from knowledge_algorithms.retrieval.selection import PayloadCost, SelectionResult
from knowledge_retrieval_app.adapters.knowledge_store import LocalKnowledgeStore
from knowledge_retrieval_app.application.knowledge_library import KnowledgeLibraryService
from knowledge_retrieval_app.application.knowledge_retrieval import KnowledgeRetrievalService, RetrievalReceipt
from knowledge_retrieval_app.domain.knowledge_ingestion import BINDING
from knowledge_retrieval_app.domain.knowledge_retrieval import RetrievalAccess
from knowledge_retrieval_app.entities.knowledge_search_history import KnowledgeSearchHistoryEntity
from knowledge_retrieval_app.repositories.knowledge_search_history import KnowledgeSearchHistoryRepository
from platform_core.contracts.knowledge_app import KnowledgeSearchHistoryDetail, KnowledgeSearchRequest


@compiles(JSONB, "sqlite")
def compile_test_jsonb(element, compiler, **kwargs):
    return "JSON"


class FixtureSession:
    def __init__(self, session):
        self.session = session

    def add(self, row):
        self.session.add(row)

    async def delete(self, row):
        self.session.delete(row)

    async def execute(self, *args, **kwargs):
        return self.session.execute(*args, **kwargs)

    async def scalars(self, statement):
        return self.session.scalars(statement)

    async def flush(self):
        self.session.flush()


class HistoryFixture:
    def __init__(self, engine, root):
        table = KnowledgeSearchHistoryEntity.__table__
        self.engine, self.store = engine, LocalKnowledgeStore(root)
        self.owner = (RetrievalAccess(uuid4(), uuid4(), "user-a") if "tenant_id" in table.c
                      else RetrievalAccess(8, None, "user-a"))
        self.library = uuid4()
        self.time = datetime(2026, 11, 23, tzinfo=timezone.utc)
        self.versions, self.active = {}, True
        self.retrieval = KnowledgeRetrievalService(uow_factory=self.uow, object_store=self.store,
            authorization=None, profiles={"fixture": object()}, tokenizer=None, image_cost=None, cursor_codec=None,
            clock=lambda: self.time)
        self.retrieval.retrieve = AsyncMock(side_effect=AssertionError("回放不得重新召回"))
        self.retrieval.validate = AsyncMock(side_effect=AssertionError("回放不应使用已过期的短期 Scope"))
        self.service = KnowledgeLibraryService(uow_factory=self.uow, ingestion=None, retrieval=self.retrieval,
                                               parse_profile=None, configurations={})
        self.history = self.service.search_history

    @asynccontextmanager
    async def uow(self):
        with Session(self.engine) as session:
            wrapped = FixtureSession(session)

            async def allowed(library_id, refs):
                require(self.active and library_id == self.library, "没有 READ")
                return "可读资料库", refs & set(self.versions)

            async def versions(access, refs, **_):
                if access == self.owner:
                    return {str(ref): self.versions[ref] for ref in refs if ref in self.versions}
                return {}

            async def listing():
                return [{"library_id": self.library, "display_name": "可读资料库"}] if self.active else []

            async def commit():
                session.commit()

            yield SimpleNamespace(search_history=lambda access: KnowledgeSearchHistoryRepository(wrapped, access),
                library=lambda access: SimpleNamespace(history_versions=allowed, listing=listing),
                knowledge_retrieval=SimpleNamespace(passage_versions=versions,
                    passage_headings=AsyncMock(return_value={})), commit=commit)

    async def item(self, title, *, image=False):
        version, run = uuid4(), str(uuid4())
        key = EvidenceKey(BINDING.binding_ref, str(version), run, "part")
        raw = title.encode()
        stored = await self.store.put_bytes(raw, max_bytes=1024)
        self.versions[version] = dict(version_id=str(version), document_id=str(uuid4()),
            title=title, view_hash="view-v1", parse_run_id=run)
        return EvidenceItem(key, (key, replace(key, object_key="part-2")),
            ContentKind.FIGURE if image else ContentKind.PARAGRAPH, (Locator(2),),
            Provenance(ProvenanceKind.NATIVE, "fixture", "v1", "hash"), "allowed-view-v1", stored.handle_ref,
            sha256(raw).hexdigest(), "image/png" if image else "text/plain", len(raw),
            PayloadCost(0, 0, len(raw)), int(self.time.timestamp()) + 900)

    async def save(self, request, items=(), hits=(), groups=()):
        owner = BINDING.binding_ref
        snapshot = RetrievalSnapshot(owner, "scope-hash", "snapshot", "generation", "withdrawal")
        mode = QueryMode(request.mode)
        evidence = EvidenceSet("request", owner, "scope", "scope-hash", snapshot, "query", mode,
            "COMPLETE" if hits else "EMPTY", None,
            tuple(PrimaryHit(item.key, rank, str(rank), None) for rank, item in enumerate(hits, 1)),
            tuple(items), tuple(groups), SelectionResult((), tuple(item.key for item in hits),
                PayloadCost(0, 0, 0), (), (), False), (), (), (),
            QueryPlan("query", "scope-hash", "profile", mode, (), (), "profile-hash", ()),
            "old-cursor" if hits else None, True)
        frozen = await self.store.put_bytes(json.dumps(encode(evidence)).encode(), max_bytes=65536)
        return await self.history.save(self.owner, self.library, request,
            RetrievalReceipt("expired-short-receipt", evidence, None, frozen.handle_ref))


@pytest.fixture
def history(tmp_path):
    engine = create_engine("sqlite://")
    table = KnowledgeSearchHistoryEntity.__table__
    table.metadata.create_all(engine, tables=[table])
    try:
        yield HistoryFixture(engine, tmp_path)
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_history_replays_original_first_batch_after_short_receipt_expiry(history):
    first, second, related, hidden = [await history.item(title, image=image) for title, image in
        (("第一份", False), ("第二份", False), ("关联图片", True), ("不能返回的候选", False))]
    entry_id = await history.save(KnowledgeSearchRequest(mode="EXHAUSTIVE", order_by=[{"field": "title"}]),
        (hidden, second, related, first), (first, second),
        (LinkedManifest(first.key, (related.key,), (), "PARTIAL", continuation="old-links"),))
    history.time += timedelta(days=2)
    detail = KnowledgeSearchHistoryDetail.model_validate(await history.service.history_detail(history.owner, entry_id))
    assert detail.availability == "AVAILABLE"
    assert detail.result is not None
    assert [hit.key for hit in detail.result.primary_hits] == [first.key.stable_key, second.key.stable_key]
    assert {item.source_title for item in detail.result.items} == {"第一份", "第二份", "关联图片"}
    assert {item.text for item in detail.result.items} == {"第一份", "第二份", None}
    assert detail.result.continuation is None and detail.result.linked[0].continuation is None
    assert all(item.content_ref.retrieval_ref == str(entry_id) for item in detail.result.items)
    history.retrieval.retrieve.assert_not_awaited()
    history.retrieval.validate.assert_not_awaited()
    with pytest.raises(AlgorithmError):
        await history.service.history_content(history.owner, entry_id, hidden.key.stable_key)
    _, image = await history.service.history_content(history.owner, entry_id, related.key.stable_key)
    assert b"".join([block async for block in image]).decode() == "关联图片"

    history.versions[UUID(related.key.document_version_ref)]["view_hash"] = "view-v2"
    del history.versions[UUID(first.key.document_version_ref)]
    partial = await history.service.history_detail(history.owner, entry_id)
    assert partial["availability"] == "PARTIAL"
    assert [hit["rank"] for hit in partial["result"]["primary_hits"]] == [2]
    assert [item["source_title"] for item in partial["result"]["items"]] == ["第二份"]
    history.active = False
    detail = await history.service.history_detail(history.owner, entry_id)
    assert detail["availability"] == "UNAVAILABLE" and detail["result"] is None
    assert detail["entry"]["library_name"] is None
    with pytest.raises(AlgorithmError):
        await history.service.history_pdf(history.owner, entry_id, second.key.stable_key)


@pytest.mark.asyncio
async def test_private_empty_image_history_expires_and_is_physically_purged(history):
    request = KnowledgeSearchRequest(images=[dict(mime_type="image/png", image_base64="PRIVATE_BASE64")])
    identity = await history.save(request)
    page = await history.history.page(history.owner, limit=1, offset=0)
    conditions = page["items"][0]["conditions"]
    assert conditions["image_count"] == 1 and conditions["text"] is None
    assert conditions["image_intent"] == "SOURCE"
    similar_id = await history.save(request.model_copy(update={"image_intent": "SIMILAR"}))
    similar = (await history.history.record(history.owner, similar_id))["conditions"]
    assert similar["image_intent"] == "SIMILAR" and "PRIVATE_BASE64" not in repr(similar)
    assert "PRIVATE_BASE64" not in repr(conditions) and "images" not in conditions
    detail = await history.service.history_detail(history.owner, identity)
    assert detail["result"]["status"] == "EMPTY" and detail["entry"]["result_count"] == 0
    other_scopes = [replace(history.owner, subject_ref="user-b"),
                    replace(history.owner, domain_id=uuid4() if history.owner.tenant_id else 9)]
    if history.owner.tenant_id:
        other_scopes.append(replace(history.owner, tenant_id=uuid4()))
    for other in other_scopes:
        assert not (await history.history.page(other, limit=1, offset=0))["items"]
        with pytest.raises(AlgorithmError):
            await history.service.history_detail(other, identity)
        await history.history.delete(other, identity)
    assert await history.history.record(history.owner, identity)

    history.time += timedelta(days=29, hours=23)
    other_id = await history.save(KnowledgeSearchRequest(text="第二次检索"))
    assert (await history.history.page(history.owner, limit=1, offset=0))["has_more"]
    await history.history.delete(history.owner, other_id)
    with pytest.raises(AlgorithmError):
        await history.history.record(history.owner, other_id)
    history.time += timedelta(hours=1)
    assert not (await history.history.page(history.owner, limit=1, offset=0))["items"]
    with Session(history.engine) as session:
        assert await KnowledgeSearchHistoryRepository.purge_expired(FixtureSession(session), now=history.time) == 2
        assert session.scalar(select(func.count()).select_from(KnowledgeSearchHistoryEntity)) == 0


@pytest.mark.asyncio
async def test_history_pdf_rechecks_before_releasing_each_chunk():
    record = {"library_id": uuid4()}
    item = SimpleNamespace(locators=[Locator(2)], key=SimpleNamespace(document_version_ref=str(uuid4())))
    seen = []

    async def raw():
        try:
            yield b"%PDF-"
            yield b"tail"
        finally:
            seen.append("closed")

    app = SimpleNamespace(_history_hit=AsyncMock(return_value=(record, None, item)),
        source=AsyncMock(return_value=({"mime_type": "application/pdf", "byte_count": 9}, raw())),
        search_history=SimpleNamespace(reauthorize=AsyncMock(
            side_effect=[None, AlgorithmError(ErrorCode.SNAPSHOT_STALE, "撤回", stage="snapshot")])))
    _, blocks = await KnowledgeLibraryService.history_pdf(app, object(), uuid4(), "anchor")
    assert await anext(blocks) == b"%PDF-"
    with pytest.raises(AlgorithmError):
        await anext(blocks)
    assert seen == ["closed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "hits", "saved"), [
    ("COMPLETE", ("hit",), True), ("PARTIAL", ("hit",), True), ("EMPTY", (), True),
    ("PARTIAL", (), False), ("FAILED", (), False),
])
async def test_direct_search_saves_only_completed_batches(status, hits, saved):
    identity, library, access = uuid4(), uuid4(), object()
    request = KnowledgeSearchRequest(text="提交的关键词")
    receipt = SimpleNamespace(evidence_set=SimpleNamespace(status=status, primary_hits=hits))
    app = SimpleNamespace(get=AsyncMock(return_value={"configuration": {"profile_ref": "base"}}),
        retrieval=SimpleNamespace(create_scope=AsyncMock(), retrieve=AsyncMock(return_value=receipt)),
        present=AsyncMock(return_value={"status": status}),
        search_history=SimpleNamespace(save=AsyncMock(return_value=identity)))
    result = await KnowledgeLibraryService.search(app, access, library, request)
    if saved:
        assert result["history_id"] == identity
        app.search_history.save.assert_awaited_once_with(access, library, request, receipt)
    else:
        assert "history_id" not in result
        app.search_history.save.assert_not_awaited()
