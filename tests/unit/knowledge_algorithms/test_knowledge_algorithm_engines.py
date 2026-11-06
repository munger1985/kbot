"""无真实依赖的解析、投影、召回链路及阶段故障验收。"""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages/knowledge_algorithms/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from knowledge_algorithm_fixtures import (AllowedReader, CharacterTokenizer, FakeBackend, FakeModel, allowed,
    candidate, node, original, parse_profile, projection_profile, provenance, retrieval_profile, run_for, source, spec)
from knowledge_algorithms.contracts.content import ContentKind, ProvenanceKind
from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode
from knowledge_algorithms.contracts.identity import AppBackendBinding, SourceSpan
from knowledge_algorithms.contracts.query import OrderBy, QueryInput, QueryMode
from knowledge_algorithms.contracts.retrieval import ChannelSpec, EvidenceBatch, RelationPage, ValidationResult
from knowledge_algorithms.parsing.contracts import ExtractedRegion, ExtractionBatch
from knowledge_algorithms.parsing.engine import ParseEngine
from knowledge_algorithms.parsing.events import ParseEventType, ParseStreamValidator
from knowledge_algorithms.parsing.plain_text import PlainTextAdapter
from knowledge_algorithms.projections.builder import ProjectionBuilder
from knowledge_algorithms.projections.contracts import Representation
from knowledge_algorithms.references.relations import RelationType, ResolvedRelation
from knowledge_algorithms.retrieval.engine import RetrievalEngine
from knowledge_algorithms.retrieval.selection import PayloadCost


class ParsingEngineTests(unittest.IsolatedAsyncioTestCase):
    async def parse(self, text="第一段。\n\n第二段。", profile=None, adapter=None):
        reader, profile = source(text), profile or parse_profile()
        return await ParseEngine(AppBackendBinding("app-a", "app-a"),
            (adapter or PlainTextAdapter(),)).parse(reader, profile, run_for(reader, profile))

    async def test_real_utf8_parse_to_canonical_nodes(self):
        parsed = await self.parse()
        self.assertEqual([n.text for n in parsed.nodes], ["第一段。", "第二段。"])
        self.assertEqual(parsed.quality_status, "COMPLETE")
        self.assertEqual(parsed.pages, ())
        self.assertTrue(all(n.key.projection_ref is None for n in parsed.nodes))

    async def test_same_event_stream_validates_independently(self):
        reader, profile = source(), parse_profile()
        run = run_for(reader, profile)
        validator = ParseStreamValidator(run, max_events=100, max_objects=1000, max_payload_bytes=100000)
        async for event in ParseEngine(AppBackendBinding("app-a", "a"), (PlainTextAdapter(),)).parse_events(reader, profile, run):
            self.assertTrue(validator.accept(event))
        self.assertTrue(validator.completed)

    async def test_source_hash_tamper_is_explicit_failure(self):
        reader, profile = source(), parse_profile()
        reader.descriptor = replace(reader.descriptor, sha256="f" * 64)
        with self.assertRaises(AlgorithmError):
            await ParseEngine(AppBackendBinding("app-a", "a"), (PlainTextAdapter(),)).parse(reader, profile, run_for(reader, profile))

    async def test_input_budget_rejected_before_adapter(self):
        with self.assertRaises(AlgorithmError) as result:
            await self.parse(profile=parse_profile(max_input_bytes=1))
        self.assertEqual(result.exception.code, ErrorCode.RESOURCE_LIMIT)

    async def test_wrong_run_profile_rejected(self):
        reader, profile = source(), parse_profile()
        with self.assertRaises(AlgorithmError):
            await ParseEngine(AppBackendBinding("app-a", "a"), (PlainTextAdapter(),)).parse(reader, profile,
                replace(run_for(reader, profile), profile_hash="f" * 64))

    async def test_cross_app_rejected(self):
        reader, profile = source(owner="other"), parse_profile()
        with self.assertRaises(AlgorithmError) as result:
            await ParseEngine(AppBackendBinding("app-a", "a"), (PlainTextAdapter(),)).parse(reader, profile, run_for(reader, profile))
        self.assertEqual(result.exception.code, ErrorCode.SCOPE_CLOSURE_BROKEN)

    async def test_photo_without_text_is_valid(self):
        class VisualAdapter(PlainTextAdapter):
            async def extract(self, source, profile, run):
                async for _ in source.chunks():
                    pass
                yield ExtractionBatch((), (ExtractedRegion("photo", (node("photo", "", ContentKind.PHOTO),), 0.0),), ())
        parsed = await self.parse(adapter=VisualAdapter())
        self.assertEqual(parsed.quality_status, "COMPLETE")

    async def test_local_repair_retains_original_and_checks_improvement(self):
        class Adapter(PlainTextAdapter):
            async def extract(self, source, profile, run):
                async for _ in source.chunks():
                    pass
                yield ExtractionBatch((), (ExtractedRegion("bad", (node(text="\ufffd\ufffd"),), 0.1),), ())

        class Repair:
            async def repair(self, region, **kwargs):
                repaired = replace(region.nodes[0], text="正确转写", provenance=provenance(ProvenanceKind.OCR))
                return replace(region, nodes=(repaired,), text_coverage=1.0)
        reader, profile = source(), parse_profile()
        engine = ParseEngine(AppBackendBinding("app-a", "a"), (Adapter(),), Repair())
        events = [event async for event in engine.parse_events(reader, profile, run_for(reader, profile))]
        finalized = json.loads(next(event.payload_json for event in events if event.event_type == ParseEventType.STRUCTURE_FINALIZED))
        self.assertEqual(finalized["alternatives"][0]["reason"], "ORIGINAL")
        self.assertTrue(finalized["alternatives"][1]["adopted"])
        self.assertEqual(json.loads(events[-1].payload_json)["quality_report"]["status"], "COMPLETE")

    async def test_no_improvement_stops_with_partial_not_infinite_retry(self):
        class Adapter(PlainTextAdapter):
            async def extract(self, source, profile, run):
                async for _ in source.chunks():
                    pass
                yield ExtractionBatch((), (ExtractedRegion("bad", (node(text="\ufffd\ufffd"),), 0.1),), ())
        class Repair:
            count = 0
            async def repair(self, region, **kwargs):
                self.count += 1
                return replace(region, nodes=(replace(region.nodes[0], provenance=provenance(ProvenanceKind.OCR)),))
        reader, profile, repair = source(), parse_profile(), Repair()
        parsed = await ParseEngine(AppBackendBinding("app-a", "a"), (Adapter(),), repair).parse(reader, profile, run_for(reader, profile))
        self.assertEqual(repair.count, 1)
        self.assertEqual(parsed.quality_status, "PARTIAL")

    async def test_timeout_emits_failure_and_closes_adapter(self):
        closed = []
        class Adapter(PlainTextAdapter):
            async def extract(self, source, profile, run):
                try:
                    await asyncio.Event().wait()
                    yield None
                finally:
                    closed.append(True)
        with self.assertRaises(AlgorithmError) as result:
            await self.parse(profile=parse_profile(timeout_seconds=0.01), adapter=Adapter())
        self.assertEqual(result.exception.code, ErrorCode.TIMEOUT)
        self.assertTrue(closed)


class ProjectionEngineTests(unittest.IsolatedAsyncioTestCase):
    async def build(self, contents, profile=None, model=None, reader=None):
        document = reader or AllowedReader(contents)
        return [batch async for batch in ProjectionBuilder(document.binding, CharacterTokenizer(), model).build(
            document, profile or projection_profile())]

    async def test_sentence_split_has_exact_source_mapping(self):
        content = allowed(node(text="第一句。第二句。第三句。"))
        batches = await self.build([content], projection_profile(max_tokens=8))
        units = [projected.unit for batch in batches for projected in batch.units]
        self.assertEqual(len(units), 2)
        self.assertEqual("".join(unit.text for unit in units), content.text)
        self.assertEqual(len({unit.key for unit in units}), 2)
        for unit in units:
            for mapping in unit.sources:
                self.assertEqual(unit.text[mapping.unit_start:mapping.unit_end],
                                  content.text[mapping.source_start:mapping.source_end])

    async def test_heading_context_not_forged_as_original(self):
        content = replace(allowed(node(text="正文。")), heading_path=("第一章",))
        batch = (await self.build([content]))[0]
        unit = batch.units[0].unit
        self.assertEqual(unit.text, "正文。")
        self.assertEqual(unit.context, "第一章")
        self.assertTrue(all(s.unit_end <= len(unit.text) for s in unit.sources))

    async def test_oversized_code_not_arbitrarily_cut(self):
        with self.assertRaises(AlgorithmError):
            await self.build([allowed(node(kind=ContentKind.CODE, text="x" * 50))], projection_profile(max_tokens=10))

    async def test_table_row_groups_keep_repeated_header(self):
        content = replace(allowed(node(kind=ContentKind.TABLE, text="单位mm\nrow1\nrow2\nrow3")),
                          table_header=(0, 4), table_rows=((5, 9), (10, 14), (15, 19)))
        batches = await self.build([content], projection_profile(max_tokens=10))
        units = [p.unit for b in batches for p in b.units]
        self.assertEqual(len(units), 3)
        self.assertTrue(all(u.text.startswith("单位mm\n") for u in units))

    async def test_allowed_view_hash_changes_projection_not_source(self):
        content = allowed(node())
        a = (await self.build([content], reader=AllowedReader([content], view_hash="a")))[0]
        b = (await self.build([content], reader=AllowedReader([content], view_hash="b")))[0]
        self.assertNotEqual(a.projection_ref, b.projection_ref)
        self.assertEqual(a.units[0].unit.source_key, b.units[0].unit.source_key)

    async def test_multivector_preserves_tokens_and_mask(self):
        model = FakeModel()
        batch = (await self.build([allowed(node())], projection_profile(representations=(spec("MULTIVECTOR"),)), model))[0]
        representation = batch.units[0].representations[0]
        self.assertIsNone(representation.vector)
        self.assertEqual(representation.matrix.mask, (True,))

    async def test_no_model_when_only_lexical_projection(self):
        batches = await self.build([allowed(node())])
        self.assertEqual(batches[0].units[0].representations, ())

    async def test_wrong_model_space_rejected(self):
        class BadModel:
            async def represent(self, value, specification):
                return Representation("wrong-space", vector=(1.0, 0.0))
        with self.assertRaises(AlgorithmError):
            await self.build([allowed(node())], projection_profile(representations=(spec(),)), BadModel())

    async def test_reader_closed_on_resource_failure(self):
        document = AllowedReader([allowed(node())])
        with self.assertRaises(AlgorithmError):
            await self.build([], projection_profile(max_input_chars=1), reader=document)
        self.assertTrue(document.closed)


class RetrievalEngineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.query = QueryInput(QueryMode.RELEVANCE, 2, ("MIXED",), text="查询")

    async def retrieve(self, profile=None, query=None, model=None):
        return await RetrievalEngine(self.backend, model).retrieve(query or self.query, self.backend.scope,
                                                                   profile or retrieval_profile())

    async def test_healthy_empty_not_channel_failure(self):
        result = await self.retrieve()
        self.assertEqual((result.status, result.reason), ("EMPTY", "NO_MATCH"))

    async def test_candidates_survive_failed_optional_dense(self):
        key = self.backend.add("p")
        self.backend.rows = {"lexical": ((key, 1),), "dense": RuntimeError("不可用")}
        dense = ChannelSpec("dense", "text-dense", "DENSE", "TEXT_TO_TEXT", 1, 10, 1, False, spec())
        profile = retrieval_profile(channels=retrieval_profile().channels + (dense,))
        result = await self.retrieve(profile, model=FakeModel())
        self.assertEqual(result.status, "PARTIAL")
        self.assertEqual(result.primary_hits[0].evidence_key, key)
        self.assertEqual(result.context_selection.selected_evidence_keys, (key,))

    async def test_failed_required_channel_not_empty(self):
        self.backend.rows = {"lexical": RuntimeError("不可用")}
        result = await self.retrieve()
        self.assertEqual((result.status, result.reason), ("FAILED", "CHANNEL_FAILED"))

    async def test_text_brings_figure_and_original_image(self):
        p = self.backend.add("p")
        f = self.backend.add("f", ContentKind.FIGURE, projected=False)
        image = self.backend.add("image", ContentKind.IMAGE, projected=False)
        self.backend.rows = {"lexical": ((p, 1),)}
        self.backend.relations = (
            ResolvedRelation("r", RelationType.REFERS_TO, original("p"), f, "LABEL_SCOPE", "occ", SourceSpan("p", 0, 4)),
            ResolvedRelation("a", RelationType.HAS_ASSET, f, image, "NATIVE"))
        result = await self.retrieve()
        self.assertEqual(set(result.groups[0].linked_keys), {f, image})
        self.assertEqual({item.key for item in result.evidence}, {p, f, image})
        self.assertTrue(all(item.payload_status == "DEFERRED" for item in result.evidence))

    async def test_figure_brings_referencing_paragraph(self):
        f = self.backend.add("f", ContentKind.FIGURE)
        p = self.backend.add("p", projected=False)
        self.backend.rows = {"lexical": ((f, 1),)}
        self.backend.relations = (ResolvedRelation("r", RelationType.REFERS_TO, p, original("f"),
                                   "LABEL_SCOPE", "occ", SourceSpan("p", 0, 4)),)
        result = await self.retrieve()
        self.assertIn(p, result.groups[0].linked_keys)

    async def test_chunk_does_not_follow_reference_outside_source_span(self):
        p = self.backend.add("p")
        f = self.backend.add("f", ContentKind.FIGURE, projected=False)
        self.backend.items[p] = replace(self.backend.items[p], source_spans=(SourceSpan("p", 20, 40),))
        self.backend.rows = {"lexical": ((p, 1),)}
        self.backend.relations = (ResolvedRelation("r", RelationType.REFERS_TO, original("p"), f,
                                   "LABEL_SCOPE", "occ", SourceSpan("p", 0, 4)),)
        self.assertEqual((await self.retrieve()).groups[0].linked_keys, ())

    async def test_context_budget_does_not_reduce_display_count(self):
        p, q = self.backend.add("p", cost=50), self.backend.add("q", cost=1)
        self.backend.rows = {"lexical": ((p, 2), (q, 1))}
        result = await self.retrieve(retrieval_profile(payload_budget=PayloadCost(2, 1, 100)))
        self.assertEqual(len(result.primary_hits), 2)
        self.assertEqual(result.context_selection.selected_evidence_keys, (q,))

    async def test_all_context_oversized_distinct_from_no_data(self):
        p = self.backend.add("p", cost=50)
        self.backend.rows = {"lexical": ((p, 1),)}
        result = await self.retrieve(retrieval_profile(payload_budget=PayloadCost(1, 0, 1)))
        self.assertEqual(result.reason, "CONTEXT_BUDGET_INSUFFICIENT")
        self.assertTrue(result.primary_hits)

    async def test_missing_payload_distinct_from_no_match(self):
        self.backend.rows = {"lexical": ((candidate("missing"), 1),)}
        result = await self.retrieve()
        self.assertEqual(result.reason, "NO_ACCESSIBLE_PAYLOAD")
        self.assertEqual(result.trace[2].excluded[0][1], "NO_ACCESSIBLE_PAYLOAD")

    async def test_final_revocation_never_leaks_attachment(self):
        p, f = self.backend.add("p"), self.backend.add("f", ContentKind.FIGURE, projected=False)
        self.backend.rows = {"lexical": ((p, 1),)}
        self.backend.relations = (ResolvedRelation("r", RelationType.REFERS_TO, original("p"), f,
                                   "LABEL_SCOPE", "occ", SourceSpan("p", 0, 4)),)
        self.backend.final_revoked = {f}
        result = await self.retrieve()
        self.assertNotIn(f, {item.key for item in result.evidence})
        self.assertEqual(result.groups[0].manifest_status, "PARTIAL")

    async def test_relations_failure_keeps_valid_primary(self):
        p = self.backend.add("p")
        self.backend.rows = {"lexical": ((p, 1),)}
        self.backend.relation_error = RuntimeError("关系读取失败")
        result = await self.retrieve()
        self.assertEqual(result.status, "PARTIAL")
        self.assertEqual(result.primary_hits[0].evidence_key, p)

    async def test_cross_app_candidate_rejected(self):
        self.backend.rows = {"lexical": ((candidate("p", "other"), 1),)}
        with self.assertRaises(AlgorithmError) as result:
            await self.retrieve()
        self.assertEqual(result.exception.code, ErrorCode.SCOPE_CLOSURE_BROKEN)

    async def test_ordered_mode_never_calls_model_or_dense(self):
        p = self.backend.add("new")
        self.backend.rows = {"ordered": ((p, 1),)}
        query = QueryInput(QueryMode.FILTERED_ORDER, 1, ("TEXT",), order_by=(OrderBy("date", True), OrderBy("id", False)))
        model = FakeModel()
        result = await self.retrieve(query=query, model=model)
        self.assertEqual(result.primary_hits[0].evidence_key, p)
        self.assertEqual(model.calls, [])
        self.assertNotIn("LEXICAL", self.backend.calls)

    async def test_image_query_uses_image_input_and_real_direction(self):
        f = self.backend.add("f", ContentKind.FIGURE)
        self.backend.rows = {"image": ((f, 1),)}
        channel = ChannelSpec("image", "image-dense", "DENSE", "IMAGE_TO_IMAGE", 1, 10, 1, True,
                              spec(input_kind="IMAGE", direction="IMAGE_TO_IMAGE"))
        query = QueryInput(QueryMode.RELEVANCE, 1, ("FIGURE",), image_handles=("image-query",))
        model = FakeModel()
        result = await self.retrieve(retrieval_profile(channels=(channel,)), query, model)
        self.assertEqual(result.primary_hits[0].evidence_key, f)
        self.assertEqual(model.calls[0].image_handle, "image-query")

    async def test_multiple_images_one_signal_vote_and_all_target_coverage(self):
        a, b = self.backend.add("a", ContentKind.FIGURE), self.backend.add("b", ContentKind.FIGURE)
        self.backend.rows = {("image", 0): ((a, 1),), ("image", 1): ((b, 1),)}
        channel = ChannelSpec("image", "image-dense", "DENSE", "IMAGE_TO_IMAGE", 1, 10, 1, True,
                              spec(input_kind="IMAGE", direction="IMAGE_TO_IMAGE"))
        query = QueryInput(QueryMode.RELEVANCE, 2, ("FIGURE",), image_handles=("i0", "i1"), combination="ALL")
        result = await self.retrieve(retrieval_profile(channels=(channel,)), query, FakeModel())
        self.assertEqual(len(result.channel_status), 1)
        self.assertTrue(all(c.status == "MATCHED" for c in result.context_selection.coverage))

    async def test_unimplemented_composed_query_explicitly_rejected(self):
        query = QueryInput(QueryMode.RELEVANCE, 2, ("FIGURE",), image_handles=("i0", "i1"), combination="COMPOSED")
        with self.assertRaises(AlgorithmError) as result:
            await self.retrieve(query=query)
        self.assertEqual(result.exception.code, ErrorCode.MODALITY_UNAVAILABLE)

    async def test_engine_cancellation_propagates_to_parallel_channels(self):
        started, finished = asyncio.Event(), []
        async def blocked(*args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.append(True)
        self.backend.search_lexical = blocked
        task = asyncio.create_task(self.retrieve())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(finished)


if __name__ == "__main__":
    unittest.main()
