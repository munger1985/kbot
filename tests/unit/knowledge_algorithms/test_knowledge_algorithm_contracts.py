"""K0 冻结线合同与 K1 结构、映射及故障反例。"""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages/knowledge_algorithms/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from knowledge_algorithm_fixtures import (AllowedReader, CharacterTokenizer, FakeBackend, FakeModel,
    allowed, candidate, node, original, parse_profile, projection_profile, provenance, retrieval_profile,
    run_for, source, spec)
from knowledge_algorithms.contracts.content import ContentKind, Page, VisualAsset
from knowledge_algorithms.contracts.cursor import CursorBinding, CursorCodec
from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode
from knowledge_algorithms.contracts.identity import AppBackendBinding, Locator, SourceSpan
from knowledge_algorithms.contracts.query import QueryInput, QueryMode
from knowledge_algorithms.contracts.registry import CONTRACTS, from_envelope, schema_manifest, to_envelope
from knowledge_algorithms.contracts.retrieval import ChannelSpec, EvidenceBatch, RelationPage, ValidationResult
from knowledge_algorithms.contracts.runtime import (CitationLabel, CitationLabelMap, DelegationClaims,
    DeliveredEvidence, ErrorEnvelope, GroundedAnswerArtifact, ModelEvidenceDelivery,
    RetrievalEvidenceArtifact, RetrievalRequest)
from knowledge_algorithms.contracts.wire import decode, encode
from knowledge_algorithms.parsing.contracts import ExtractedRegion, ExtractionBatch
from knowledge_algorithms.parsing.engine import ParseEngine
from knowledge_algorithms.parsing.events import ParseArtifact, ParseEvent, ParseEventType, ParseStreamValidator
from knowledge_algorithms.parsing.plain_text import PlainTextAdapter
from knowledge_algorithms.parsing.structure import order_and_structure
from knowledge_algorithms.parsing.wire import event_from_wire, event_to_wire
from knowledge_algorithms.projections.builder import ProjectionBuilder
from knowledge_algorithms.projections.contracts import UnitSource
from knowledge_algorithms.retrieval.engine import RetrievalEngine
from knowledge_algorithms.retrieval.selection import CoverageTarget, PayloadCost


async def contract_samples():
    backend = FakeBackend()
    p = backend.add("p")
    backend.rows = {"lexical": ((p, 1),)}
    query, profile = QueryInput(QueryMode.RELEVANCE, 1, ("TEXT",), text="合成样例"), retrieval_profile()
    evidence = await RetrievalEngine(backend, None).retrieve(query, backend.scope, profile)
    reader, parsing = source(), parse_profile()
    parsed = await ParseEngine(backend.binding, (PlainTextAdapter(),)).parse(reader, parsing, run_for(reader, parsing))
    projection = projection_profile()
    document = AllowedReader([allowed(node())])
    batches = [b async for b in ProjectionBuilder(backend.binding, CharacterTokenizer(), None).build(document, projection)]
    label = CitationLabel("run-1", "1", p, "PRIMARY", (Locator(logical_ref="p"),))
    labels = CitationLabelMap("run-1", (label,))
    delivery = ModelEvidenceDelivery("run-1", "model-fixture/v1", "PREPARED",
        (DeliveredEvidence(p, "1", "a" * 64, "TEXT", PayloadCost(1, 0, 1)),), None)
    values = (backend.binding, backend.scope, backend.fixed_snapshot, p, query, evidence.plan, profile, evidence,
        reader.descriptor, parsing, parsed, projection, batches[0], spec(),
        DelegationClaims("app-a", "app-a-service", "user-1", None, "domain-1", None, "run-1", "spec-hash",
            "scope-hash", ("retrieve",), 2000000000),
        RetrievalRequest("request-1", "scope-ref", "scope-hash", query, "profile-1", PayloadCost(10, 1, 100),
            "run-1", "spec-hash"), labels, RetrievalEvidenceArtifact("run-1", "turn-1", evidence, labels),
        delivery, GroundedAnswerArtifact("run-1", "示例", ("1",), (label,), "COMPLETE"),
        ErrorEnvelope(ErrorCode.CHANNEL_FAILED, "channel", True), cursor())
    from knowledge_algorithms.contracts.runtime import AppRetrievalArtifact, KnowledgeCompositionArtifact
    values += (AppRetrievalArtifact("receipt-1", RetrievalEvidenceArtifact("run-1", "turn-1", evidence, labels)),
        KnowledgeCompositionArtifact("receipt-1", evidence, delivery, None, labels, (), evidence.context_selection))
    return dict(zip(CONTRACTS, values))


def cursor():
    return CursorBinding("app-a", "user-1", "scope-hash", "snapshot-1", "query-hash", "profile-hash",
                         "PRIMARY", "EXHAUSTIVE", "[]", '{"offset":1}', 2000000000)


class FrozenContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_registered_contract_roundtrips(self):
        samples = await contract_samples()
        self.assertEqual(set(samples), set(CONTRACTS))
        for name, value in samples.items():
            with self.subTest(schema=name):
                wire = json.loads(json.dumps(to_envelope(name, value), allow_nan=False))
                self.assertEqual(from_envelope(wire), value)

    async def test_frozen_schema_and_examples(self):
        base = ROOT / "tests/contract/knowledge_algorithms"
        self.assertEqual(schema_manifest(), json.loads((base / "schemas-v1.json").read_text()))
        frozen = json.loads((base / "examples-v1.json").read_text())
        self.assertEqual(frozen, {name: to_envelope(name, value) for name, value in (await contract_samples()).items()})

    async def test_evidence_assembly_roundtrip_keeps_expiry_and_not_sent(self):
        evidence = (await contract_samples())["knowledge.evidence_set.v1"]
        wire = to_envelope("knowledge.evidence_set.v1", evidence)
        self.assertEqual(wire["evidence"][0]["payload_status"], "DEFERRED")
        self.assertGreater(wire["evidence"][0]["expires_at"], 0)
        self.assertNotIn("provided_to_answer_model", wire["evidence"][0])

    def test_strict_unknown_versions_and_fields(self):
        for value in (None, {}, {"schema_version": []}, {"schema_version": "legacy.v1"},
                      {"schema_version": "knowledge.backend_binding.v1", "binding_ref": "a", "namespace": "a", "target_app": "b"}):
            with self.subTest(value=value), self.assertRaises(AlgorithmError):
                from_envelope(value)

    def test_strict_scalars_union_and_matrix(self):
        self.assertEqual(decode(str | int | float | bool | None, "a"), "a")
        self.assertEqual(decode(tuple[int, int] | None, [1, 2]), (1, 2))
        self.assertIsNone(decode(tuple[int, int] | None, None))
        for contract, value in ((int, True), (float, True), (float, float("nan")),
                                (tuple[int, int], [1]), (QueryMode, "LEGACY")):
            with self.subTest(contract=contract), self.assertRaises(AlgorithmError):
                decode(contract, value)
        with self.assertRaises(AlgorithmError):
            encode(float("inf"))

    def test_parse_event_flat_wire_and_hash(self):
        reader, profile = source(), parse_profile()
        run = run_for(reader, profile)
        event = ParseEvent.create(run=run, sequence=1, batch_key="b1",
            event_type=ParseEventType.RUN_STARTED, payload={"source": "fixture"})
        self.assertEqual(event_from_wire(event_to_wire(event)), event)
        wire = event_to_wire(event)
        self.assertNotIn("run", wire)
        wire["payload"]["source"] = "tampered"
        with self.assertRaises(AlgorithmError):
            event_from_wire(wire)

    def test_controlled_artifact_roundtrip_and_ownership(self):
        reader, profile = source(), parse_profile()
        run = run_for(reader, profile)
        artifact = ParseArtifact("controlled-image-1", "image-1", "a" * 64, "image/png", 128, 2000000000)
        event = ParseEvent.create(run=run, sequence=2, batch_key="b2", event_type=ParseEventType.ASSET_BATCH,
            payload={"objects": [{"local_key": "image-1"}]}, artifact_handles=(artifact,))
        self.assertEqual(event_from_wire(event_to_wire(event)), event)
        validator = ParseStreamValidator(run, max_events=10, max_objects=10, max_payload_bytes=10000)
        validator.accept(ParseEvent.create(run=run, sequence=1, batch_key="b1",
            event_type=ParseEventType.RUN_STARTED, payload={}))
        validator.accept(event)
        with self.assertRaises(AlgorithmError):
            validator.accept(replace(event, sequence=3, batch_key="b3",
                artifact_handles=(replace(artifact, object_key="foreign"),)))
        with self.assertRaises(AlgorithmError):
            replace(artifact, handle_ref="https://untrusted/image")

    def test_cursor_tamper_scope_subject_expiry_and_role(self):
        codec, expected = CursorCodec(b"x" * 32, max_bytes=4096), cursor()
        token = codec.encode(expected)
        self.assertEqual(codec.decode(token, expected=expected, now=1), expected)
        for changed in (replace(expected, subject_ref="other"), replace(expected, scope_hash="other"),
                        replace(expected, role="REFERENCED"), replace(expected, profile_hash="other")):
            with self.assertRaises(AlgorithmError) as error:
                codec.decode(token, expected=changed, now=1)
            self.assertEqual(error.exception.code, ErrorCode.CURSOR_INVALID)
        with self.assertRaises(AlgorithmError) as error:
            codec.decode(token, expected=expected, now=expected.expires_at)
        self.assertEqual(error.exception.code, ErrorCode.CURSOR_EXPIRED)
        with self.assertRaises(AlgorithmError):
            codec.decode("!" + token[1:], expected=expected, now=1)

    async def test_delegation_context_and_delivery_not_fabricated(self):
        values = await contract_samples()
        claims = values["knowledge.delegation_claims.v1"]
        claims.validate(values["knowledge.backend_binding.v1"], values["knowledge.authorized_scope.v1"],
            audience="app-a-service", operation="retrieve", now=1, run_ref="run-1", spec_hash="spec-hash")
        for name, change in (("audience", "other"), ("scope_hash", "other"), ("run_ref", "other")):
            with self.assertRaises(AlgorithmError):
                replace(claims, **{name: change}).validate(values["knowledge.backend_binding.v1"],
                    values["knowledge.authorized_scope.v1"], audience="app-a-service",
                    operation="retrieve", now=1, run_ref="run-1", spec_hash="spec-hash")
        with self.assertRaises(AlgorithmError):
            replace(values["knowledge.model_evidence_delivery.v1"], status="SENT")
        with self.assertRaises(AlgorithmError):
            replace(values["knowledge.citation_label_map.v1"],
                    entries=values["knowledge.citation_label_map.v1"].entries * 2)


class StructureAndMappingTests(unittest.TestCase):
    def test_two_columns_not_interleaved_by_y(self):
        def positioned(name, bbox):
            return node(name, name, locators=(Locator(page_no=1, bbox=bbox),))
        heading = positioned("heading", (0.05, 0.01, 0.95, 0.1))
        left1, left2 = positioned("left1", (0.05, 0.2, 0.45, 0.3)), positioned("left2", (0.05, 0.6, 0.45, 0.7))
        right1 = positioned("right1", (0.55, 0.21, 0.95, 0.31))
        ordered, _ = order_and_structure((right1, left2, heading, left1), parse_profile().layout)
        self.assertEqual([n.text for n in ordered], ["heading", "left1", "left2", "right1"])

    def test_heading_parent_and_cycle_rejection(self):
        heading = node("h", "第一章", ContentKind.SECTION, heading_level=1)
        paragraph = node("p", "正文", reading_order=1)
        ordered, _ = order_and_structure((heading, paragraph), parse_profile().layout)
        self.assertEqual(ordered[1].parent_key, heading.key.object_key)
        with self.assertRaises(AlgorithmError):
            order_and_structure((replace(heading, parent_key=paragraph.key.object_key),
                replace(paragraph, parent_key=heading.key.object_key)), parse_profile().layout)

    def test_mapping_rejects_cross_version_and_overlap(self):
        content = allowed(node(text="abcdef"))
        for mapping in ((UnitSource(0, 6, replace(content.key, document_version_ref="other"), 0, 6),),
                        (UnitSource(0, 4, content.key, 0, 4), UnitSource(3, 6, content.key, 3, 6))):
            with self.assertRaises(AlgorithmError):
                replace(content, sources=mapping)

    def test_table_row_overlap_rejected(self):
        with self.assertRaises(AlgorithmError):
            replace(allowed(node(text="header-row", kind=ContentKind.TABLE)),
                    table_header=(0, 6), table_rows=((5, 10),))


class ExtraEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_redacted_and_cross_page_mapping_preserved(self):
        first, second = node("p1", "publicSECRET"), node("p2", "body")
        content = replace(allowed(first), text="public\nbody",
            locators=(Locator(page_no=1), Locator(page_no=2)),
            sources=(UnitSource(0, 6, first.key, 0, 6), UnitSource(7, 11, second.key, 0, 4)))
        reader = AllowedReader([content])
        batches = [b async for b in ProjectionBuilder(reader.binding, CharacterTokenizer(), None).build(reader, projection_profile())]
        unit = batches[0].units[0].unit
        self.assertNotIn("SECRET", unit.text)
        self.assertEqual({s.source_key for s in unit.sources}, {first.key, second.key})
        self.assertEqual(len(unit.locators), 2)

    async def test_real_parser_figure_caption_asset_relations(self):
        p, f = node("p"), node("f", "", ContentKind.FIGURE, display_label="图 1-1", numbering_scope="document")
        caption = node("caption", "图 1-1 测试", ContentKind.CAPTION, parent_key=f.key.object_key)
        asset = VisualAsset(original("asset"), f.key, "controlled-asset", "a" * 64, "image/png", 128,
                            Locator(logical_ref="asset"), provenance())
        class Adapter(PlainTextAdapter):
            async def extract(self, source, profile, run):
                async for _ in source.chunks():
                    pass
                yield ExtractionBatch((), (ExtractedRegion("region", (p, f, caption), 1.0),), (asset,))
        reader, profile = source(), parse_profile()
        parsed = await ParseEngine(AppBackendBinding("app-a", "a"), (Adapter(),)).parse(reader, profile, run_for(reader, profile))
        self.assertEqual({r.relation_type.value for r in parsed.relations}, {"REFERS_TO", "CAPTION_OF", "HAS_ASSET"})
        self.assertTrue(parsed.references)

    async def test_parser_cancellation_closes_iterator(self):
        started, closed = asyncio.Event(), []
        class Adapter(PlainTextAdapter):
            async def extract(self, source, profile, run):
                try:
                    started.set()
                    await asyncio.Event().wait()
                    yield None
                finally:
                    closed.append(True)
        reader, profile = source(), parse_profile()
        task = asyncio.create_task(ParseEngine(AppBackendBinding("app-a", "a"), (Adapter(),)).parse(reader, profile, run_for(reader, profile)))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(closed)

    async def test_text_to_image_uses_text_encoder_input_in_image_space(self):
        backend, model = FakeBackend(), FakeModel()
        f = backend.add("f", ContentKind.FIGURE)
        backend.rows = {"image": ((f, 1),)}
        channel = ChannelSpec("image", "cross-modal", "DENSE", "TEXT_TO_IMAGE", 1, 10, 1, True,
                              spec(input_kind="IMAGE", direction="TEXT_TO_IMAGE"))
        result = await RetrievalEngine(backend, model).retrieve(QueryInput(QueryMode.RELEVANCE, 1, ("FIGURE",), text="系统图"),
            backend.scope, retrieval_profile(channels=(channel,)))
        self.assertEqual(result.primary_hits[0].evidence_key, f)
        self.assertEqual(model.calls[0].kind, "TEXT")

    async def test_full_profile_and_targets_participate_in_request_identity(self):
        backend = FakeBackend()
        engine, query = RetrievalEngine(backend, None), QueryInput(QueryMode.RELEVANCE, 1, ("TEXT",), text="a")
        a = await engine.retrieve(query, backend.scope, retrieval_profile())
        b = await engine.retrieve(query, backend.scope, retrieval_profile(rrf_constant=70))
        target = CoverageTarget("t", "SUBQUESTION", True, "user:0:1", "plan/v1")
        c = await engine.retrieve(query, backend.scope, retrieval_profile(), targets=(target,))
        self.assertNotEqual(a.request_id, b.request_id)
        self.assertNotEqual(a.plan.profile_hash, b.plan.profile_hash)
        self.assertNotEqual(a.request_id, c.request_id)

    async def test_relation_page_budget_keeps_controlled_continuation(self):
        backend = FakeBackend()
        p = backend.add("p")
        backend.rows = {"lexical": ((p, 1),)}
        async def pages(*args):
            return RelationPage((), False, "controlled-next-page")
        backend.follow_relations = pages
        result = await RetrievalEngine(backend, None).retrieve(QueryInput(QueryMode.RELEVANCE, 1, ("TEXT",), text="a"),
            backend.scope, retrieval_profile(max_relation_pages=1))
        self.assertEqual(result.groups[0].continuation, "controlled-next-page")
        self.assertEqual(result.groups[0].incomplete_reason, "RELATION_PAGE_BUDGET")
        self.assertEqual(result.status, "PARTIAL")

    async def test_malformed_load_and_revalidation_cannot_become_no_match(self):
        for stage in ("load", "validate"):
            backend = FakeBackend()
            p = backend.add("p")
            backend.rows = {"lexical": ((p, 1),)}
            async def bad_load(*args):
                return EvidenceBatch(())
            async def bad_validate(*args):
                return ValidationResult((), ())
            if stage == "load":
                backend.load_evidence = bad_load
            else:
                backend.revalidate = bad_validate
            with self.subTest(stage=stage), self.assertRaises(AlgorithmError):
                await RetrievalEngine(backend, None).retrieve(QueryInput(QueryMode.RELEVANCE, 1, ("TEXT",), text="a"),
                    backend.scope, retrieval_profile())

    async def test_exhaustive_uses_only_native_ordered_backend(self):
        backend, model = FakeBackend(), FakeModel()
        p = backend.add("p")
        backend.rows = {"ordered": ((p, 1),)}
        from knowledge_algorithms.contracts.query import OrderBy
        query = QueryInput(QueryMode.EXHAUSTIVE, 1, ("TEXT",), order_by=(OrderBy("id", False),))
        result = await RetrievalEngine(backend, model).retrieve(query, backend.scope, retrieval_profile())
        self.assertEqual(result.primary_hits[0].evidence_key, p)
        self.assertNotIn("LEXICAL", backend.calls)
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
