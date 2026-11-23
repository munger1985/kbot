"""已发布索引的批次预算、全范围排序和按需证据回归。"""

import asyncio
from dataclasses import replace
from pathlib import Path
from random import Random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "packages/knowledge_algorithms/src"), str(ROOT / "services/knowledge_retrieval_app/src"),
               str(ROOT / "tests/unit/knowledge_algorithms")]

from knowledge_algorithm_fixtures import (AllowedReader, CharacterTokenizer, allowed, node,
                                          projection_profile, spec)
from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode
from knowledge_algorithms.contracts.identity import canonical_json, fingerprint
from knowledge_algorithms.contracts.query import OrderBy, QueryInput, QueryMode
from knowledge_algorithms.contracts.retrieval import ChannelRequest, ChannelSpec
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.projections.builder import ProjectionBuilder
from knowledge_algorithms.projections.contracts import Representation
from knowledge_algorithms.retrieval.exact import ExactIndex, vector_score
from knowledge_algorithms.retrieval.maxsim import TokenMatrix
from knowledge_algorithms.retrieval.published_metadata import PublishedMetadataCache
from knowledge_retrieval_app.application.knowledge_hydration import hydrate_published
from knowledge_retrieval_app.adapters.knowledge_maxsim import prepare_maxsim
from knowledge_retrieval_app.adapters.knowledge_store import LocalKnowledgeStore
from knowledge_retrieval_app.domain.knowledge_ingestion import OWNER
from knowledge_retrieval_app.domain.knowledge_retrieval import RetrievalLimits


class RecordingStore(LocalKnowledgeStore):
    def __init__(self, path):
        super().__init__(path)
        self.reads, self.writes = [], []

    async def stream(self, handle):
        self.reads.append(handle)
        async for part in super().stream(handle):
            yield part

    async def put_bytes(self, value, **kwargs):
        result = await super().put_bytes(value, **kwargs)
        self.writes.append(result.handle_ref)
        return result


class VectorModel:
    def output(self, value, specification, kind):
        row = (value,) + (0.123456789,) * (specification.dimension - 1)
        if kind == "DENSE":
            return Representation(specification.space_id, vector=row)
        return Representation(specification.space_id,
            matrix=TokenMatrix(specification.space_id, (row,) * 4, (True,) * 4))

    async def represent(self, value, specification):
        return self.output(2.0 if "末尾" in value.text else 1.0, specification, specification.kind)


@pytest.mark.parametrize("kind", ["DENSE", "MULTIVECTOR"])
def test_published_vectors_over_cumulative_budget_rank_across_batches(tmp_path, kind):
    async def verify():
        store = RecordingStore(tmp_path)
        model = VectorModel()
        specs = (replace(spec(kind=kind), dimension=512),)
        contents = []
        for i, text in enumerate(("目标句 甲", "目标句 两个词", "目标句 末尾", "其他内容")):
            original = node(str(i), text=text)
            contents.append(allowed(replace(original, key=replace(original.key, owner_binding_ref=OWNER))))
        reader = AllowedReader(tuple(contents), owner=OWNER)
        profile = projection_profile(representations=specs, batch_size=1, max_representation_elements=8192)
        builder = ProjectionBuilder(reader.binding, CharacterTokenizer(), model)
        batches = [batch async for batch in builder.build(reader, profile)]
        assert len(batches) == 4
        allowed_entries, batch_entries, lengths, metadata = [], {}, [], 0

        async def put(value):
            raw = canonical_json(value).encode("utf-8")
            return await store.put_bytes(raw, max_bytes=1024 * 1024)

        for content in contents:
            result = await put(encode(content))
            metadata += result.byte_count
            allowed_entries.append({"handle": result.handle_ref, "key": content.key.stable_key,
                                    "max_bytes": result.byte_count})
        manifest = await put({"items": allowed_entries, "view_ref": reader.allowed_view_ref,
                              "view_hash": reader.allowed_view_hash})
        metadata += manifest.byte_count
        for batch in batches:
            wire = encode(batch)
            metadata += sum(len(canonical_json({**unit, "representations": []}).encode("utf-8")) for unit in wire["units"])
            result = await put(wire)
            lengths.append(result.byte_count)
            batch_entries[str(batch.sequence)] = {"ref": result.handle_ref, "hash": batch.payload_hash,
                                                  "count": len(batch.units)}
        generation = {"profile": encode(profile), "batch_count": len(batches), "batches": batch_entries,
                      "batch_digest": fingerprint([batch.payload_hash for batch in batches]), "unit_count": len(batches)}
        bundle = {"scope": {"expires_at": 2000000000}, "assets": [], "relations": [], "versions": [{
            "version_id": reader.document_version_ref, "parse_run_id": reader.parse_run_ref,
            "projection_ref": batches[0].projection_ref, "allowed_view_hash": reader.allowed_view_hash,
            "allowed_view_manifest_ref": manifest.handle_ref, "generation_manifest": generation,
            "title": "资料", "business_asset_ref": "asset", "created_at": "2026-01-01",
        }]}
        vector_elements = 512 if kind == "DENSE" else 2048
        # 旧实现以总向量体积拒绝整库，新实现仅限制单批向量与常驻元数据。
        assert sum(lengths) > metadata
        limits = RetrievalLimits(max_object_bytes=max(lengths) + 1, max_total_object_bytes=metadata,
                                 max_vector_elements=vector_elements)
        store.reads.clear()
        store.writes.clear()
        published = await hydrate_published(bundle, store=store,
            tokenizer=CharacterTokenizer(), image_cost=lambda _: 1, limits=limits)
        index, items, relations, vectors, materialize = published.bind(
            store=store, max_bytes=limits.max_object_bytes, expires_at=2000000000)
        other_index, other_items, _, other_vectors, _ = published.bind(
            store=store, max_bytes=limits.max_object_bytes, expires_at=2000000010)
        assert other_index is index and other_vectors is not vectors
        assert items[index.units[0].key].expires_at == 2000000000
        assert other_items[index.units[0].key].expires_at == 2000000010
        assert len(index.units) == 4 and not store.writes
        assert all(not indexed.projected.representations for indexed in index.units)
        baseline = ExactIndex([replace(indexed, projected=batch.units[0]) for indexed, batch in zip(index.units, batches)],
            max_units=100, max_text_chars=10000, max_vector_elements=10000)
        assert index.statistics_ref == baseline.statistics_ref
        store.reads.clear()
        query = QueryInput(QueryMode.RELEVANCE, 3, ("TEXT",), text="目标句")
        lexical = ChannelRequest(query, ChannelSpec("lexical", "lexical", "LEXICAL", "TEXT_TO_TEXT", 1.0, 3, 5, True), None)
        assert index.search(lexical) == baseline.search(lexical)
        ordered = QueryInput(QueryMode.EXHAUSTIVE, 3, ("TEXT",), order_by=(OrderBy("stable_key", False),))
        assert index.ordered(ordered) == baseline.ordered(ordered)
        assert not store.reads
        specification = specs[0]
        request = ChannelRequest(query, ChannelSpec(kind, kind, kind, "TEXT_TO_TEXT", 1.0, 3, 5, False, specification),
                                 model.output(1., specification, kind))
        hits = await vectors.search(index, request)
        expected = baseline.search(request)
        assert [hit.key for hit in hits] == [hit.key for hit in expected]
        assert [hit.score for hit in hits] == pytest.approx([hit.score for hit in expected], rel=1e-12, abs=1e-12)
        assert hits[0].key == index.units[2].key and len(store.reads) == 4
        unsupported = replace(query, output_types=("FIGURE",))
        store.reads.clear()
        assert await vectors.search(index, replace(request, query=unsupported)) == () and not store.reads
        item = items[hits[0].key]
        await materialize(item)
        await materialize(item)
        assert store.writes == [item.content_handle]
        assert (await store.read(item.content_handle, max_bytes=100)).decode() == contents[2].text

        with pytest.raises(AlgorithmError) as error:
            await hydrate_published(bundle, store=store, tokenizer=CharacterTokenizer(), image_cost=lambda _: 1,
                                    limits=replace(limits, max_object_bytes=max(lengths) - 1))
        assert (error.value.code, error.value.stage) == (ErrorCode.RESOURCE_LIMIT, "index_batch")

    asyncio.run(verify())


@pytest.mark.asyncio
@pytest.mark.parametrize("similarity", ["DOT", "COSINE"])
async def test_image_maxsim_matches_reference_across_tiles(similarity):
    rng = Random(7)
    specification = replace(spec(kind="MULTIVECTOR"), dimension=8, input_kind="IMAGE",
                            similarity=similarity, directions=frozenset({"IMAGE_TO_IMAGE"}))
    space = specification.space_id

    def representation(size, offset):
        # 有效点积始终为负；被掩盖的零向量和高分向量不得参与最大值。
        rows = tuple(tuple(offset * rng.uniform(.25, 1) for _ in range(specification.dimension)) for _ in range(size))
        return Representation(space, matrix=TokenMatrix(space, rows + ((0.,) * 8, (-offset,) * 8),
                                                         (True,) * size + (False, False)))

    query = QueryInput(QueryMode.RELEVANCE, 3, ("FIGURE",), image_handles=("query-image",))
    channel = ChannelSpec("visual", "visual", "MULTIVECTOR", "IMAGE_TO_IMAGE", 1., 3, 5, False, specification)
    request = ChannelRequest(query, channel, representation(262, 1.), input_index=0)
    document = representation(273, -1.)
    score = prepare_maxsim(request)
    expected = vector_score(request, document)
    assert expected < 0
    assert await score(document) == pytest.approx(expected, rel=1e-12, abs=1e-12)
    wrong_space = replace(document, matrix=replace(document.matrix, space_id="wrong-space"))
    with pytest.raises(AlgorithmError) as error:
        await score(wrong_space)
    assert error.value.code == ErrorCode.MODEL_SPACE_MISMATCH

    pending = asyncio.create_task(score(document))
    await asyncio.sleep(0)
    assert not pending.done(), "单张多 token 图片的计算也必须有取消检查点"
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending


@pytest.mark.asyncio
async def test_metadata_cache_deduplicates_loading_and_evicts_by_weight_and_ttl():
    now = 0
    cache = PublishedMetadataCache(max_entries=2, max_bytes=10, ttl=30, clock=lambda: now)
    calls = []

    def build(generation, weight):
        async def load():
            calls.append(generation)
            await asyncio.sleep(0)
            return object(), weight
        return load

    first, second = await asyncio.gather(cache.get("g1", build("g1", 6)), cache.get("g1", build("g1", 6)))
    assert first is second and calls == ["g1"]
    new = await cache.get("g2", build("g2", 6))
    assert new is not first
    assert await cache.get("g2", build("g2", 6)) is new
    assert await cache.get("g1", build("g1", 6)) is not first  # 超出权重，旧条目已淘汰。
    now = 31
    await cache.get("g1", build("g1", 6))
    oversized = await cache.get("large", build("large", 11))
    assert await cache.get("large", build("large", 11)) is not oversized
    assert calls == ["g1", "g2", "g1", "g1", "large", "large"]
