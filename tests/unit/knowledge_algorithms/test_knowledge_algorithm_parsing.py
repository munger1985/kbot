"""知识算法解析事件、图号歧义和双向关联的纯合同反例。"""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages/knowledge_algorithms/src"))

from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode
from knowledge_algorithms.contracts.identity import EvidenceKey, SourceSpan, fingerprint
from knowledge_algorithms.parsing.events import (
    ParseEvent, ParseEventType, ParseRunIdentity, ParseStreamValidator, local_key,
)
from knowledge_algorithms.references.figures import (
    FigureIdentity, ReferenceOccurrence, ResolutionStatus, extract_figure_references,
    parse_figure_label, resolve_figure_reference,
)
from knowledge_algorithms.references.relations import RelationType, ResolvedRelation, direct_figure_manifest


def key(name, **overrides):
    return EvidenceKey(**({"owner_binding_ref": "owner-a", "document_version_ref": "doc-v1",
                          "parse_run_ref": "run-1", "object_key": name,
                          "projection_ref": None} | overrides))


def refers(name, paragraph, figure, start=0):
    return ResolvedRelation(name, RelationType.REFERS_TO, key(paragraph), key(figure), "LABEL_SCOPE",
                            "occurrence-" + name, SourceSpan(paragraph, start, start + 4))


class EventTests(unittest.TestCase):
    def setUp(self):
        self.run = ParseRunIdentity("owner-a", "job-1", "run-1", 1, 5, "a" * 64, "b" * 64)
        self.validator = ParseStreamValidator(self.run, max_events=20, max_objects=20, max_payload_bytes=4096)

    def event(self, kind, payload=None, **kwargs):
        return ParseEvent.create(run=kwargs.get("run", self.run),
                                 sequence=kwargs.get("sequence", self.validator.event_count + 1),
                                 batch_key=kwargs.get("batch_key", f"batch-{self.validator.event_count + 1}"),
                                 event_type=kind, payload=payload or {},
                                 dependency_keys=kwargs.get("dependencies", ()))

    def start(self):
        event = self.event(ParseEventType.RUN_STARTED)
        self.validator.accept(event)
        return event

    def node(self, name="p", **kwargs):
        event = self.event(ParseEventType.NODE_BATCH, {"objects": [{"local_key": name}]}, **kwargs)
        self.validator.accept(event)
        return event

    def final(self):
        self.validator.accept(self.event(ParseEventType.STRUCTURE_FINALIZED))

    def completion(self, **overrides):
        manifest = {"counts": self.validator.object_counts, "event_count": self.validator.event_count,
                    "ordered_digest": self.validator.ordered_digest, "quality_report": {"status": "COMPLETE"}}
        return self.event(ParseEventType.RUN_COMPLETED, manifest | overrides)

    def test_complete_stream_and_same_event_replay(self):
        start = self.start()
        node = self.node()
        self.final()
        completion = self.completion()
        self.assertTrue(self.validator.accept(completion))
        self.assertTrue(self.validator.completed)
        digest = self.validator.ordered_digest
        for event in (node, start, completion):
            self.assertFalse(self.validator.accept(event))
        self.assertEqual(digest, self.validator.ordered_digest)

    def test_same_sequence_different_content_conflict(self):
        self.start()
        changed = self.event(ParseEventType.RUN_STARTED, {"changed": True}, sequence=1, batch_key="batch-1")
        with self.assertRaises(AlgorithmError) as result:
            self.validator.accept(changed)
        self.assertEqual(result.exception.code, ErrorCode.EVENT_CONFLICT)
        self.assertEqual(self.validator.event_count, 1)

    def test_batch_key_cannot_rebind_sequence(self):
        self.start()
        with self.assertRaises(AlgorithmError) as result:
            self.node(batch_key="batch-1")
        self.assertEqual(result.exception.code, ErrorCode.EVENT_CONFLICT)

    def test_forward_dependency_allowed_until_finalization(self):
        self.start()
        self.node("p", dependencies=("figure",))
        digest = self.validator.ordered_digest
        with self.assertRaises(AlgorithmError):
            self.final()
        self.assertEqual(self.validator.ordered_digest, digest)
        self.validator.accept(self.event(ParseEventType.ASSET_BATCH, {"objects": [{"local_key": "figure"}]}))
        self.final()
        self.assertTrue(self.validator.accept(self.completion()))

    def test_missing_sequence_rejected_without_state_change(self):
        self.start()
        with self.assertRaises(AlgorithmError) as result:
            self.node(sequence=3)
        self.assertEqual(result.exception.code, ErrorCode.EVENT_SEQUENCE_INVALID)
        self.assertEqual(self.validator.event_count, 1)

    def test_late_fence_or_other_app_rejected(self):
        for run in (replace(self.run, fence=4), replace(self.run, owner_binding_ref="owner-b"),
                    replace(self.run, source_sha256="c" * 64), replace(self.run, attempt=2)):
            with self.subTest(run=run), self.assertRaises(AlgorithmError) as result:
                self.validator.accept(self.event(ParseEventType.RUN_STARTED, run=run))
            self.assertEqual(result.exception.code, ErrorCode.SCOPE_CLOSURE_BROKEN)

    def test_bad_payload_hash_rejected(self):
        event = replace(self.event(ParseEventType.RUN_STARTED), payload_hash="0" * 64)
        with self.assertRaises(AlgorithmError):
            self.validator.accept(event)

    def test_nan_and_duplicate_json_keys_rejected(self):
        for payload in ('{"x":NaN}', '{"x":1,"x":2}'):
            from hashlib import sha256
            event = replace(self.event(ParseEventType.RUN_STARTED), payload_json=payload,
                            payload_hash=sha256(payload.encode()).hexdigest())
            with self.subTest(payload=payload), self.assertRaises(AlgorithmError):
                self.validator.accept(event)

    def test_wrong_manifest_has_explicit_failure(self):
        self.start()
        self.node()
        self.final()
        for changes in ({"ordered_digest": "0" * 64}, {"counts": {}}, {"event_count": 999},
                        {"quality_report": {"status": "UNKNOWN"}}):
            with self.subTest(changes=changes), self.assertRaises(AlgorithmError) as result:
                self.validator.accept(self.completion(**changes))
            self.assertEqual(result.exception.code, ErrorCode.MANIFEST_INVALID)
        self.assertFalse(self.validator.completed)

    def test_no_completion_before_structure_finalized(self):
        self.start()
        with self.assertRaises(AlgorithmError):
            self.validator.accept(self.completion())

    def test_object_identity_cannot_be_overwritten(self):
        self.start()
        self.node()
        with self.assertRaises(AlgorithmError):
            self.node()

    def test_no_append_after_structure_finalized(self):
        self.start()
        self.final()
        with self.assertRaises(AlgorithmError):
            self.node()

    def test_failure_is_not_successful_completion(self):
        self.start()
        self.final()
        self.validator.accept(self.event(ParseEventType.RUN_FAILED, {"reason": "RESOURCE_LIMIT"}))
        self.assertFalse(self.validator.completed)
        with self.assertRaises(AlgorithmError):
            self.validator.accept(self.completion())

    def test_partial_quality_completion_not_publication_permission(self):
        self.start()
        self.final()
        self.validator.accept(self.completion(quality_report={"status": "PARTIAL", "failed_pages": [2]}))
        self.assertTrue(self.validator.completed)

    def test_event_object_and_payload_budgets(self):
        for kwargs, scenario in (({"max_events": 1}, "events"),
                                 ({"max_objects": 1}, "objects"),
                                 ({"max_payload_bytes": 2}, "bytes")):
            values = {"max_events": 10, "max_objects": 10, "max_payload_bytes": 1000} | kwargs
            self.validator = ParseStreamValidator(self.run, **values)
            self.start()
            with self.subTest(scenario=scenario), self.assertRaises(AlgorithmError) as result:
                if scenario == "objects":
                    self.validator.accept(self.event(ParseEventType.NODE_BATCH,
                                                     {"objects": [{"local_key": "a"}, {"local_key": "b"}]}))
                else:
                    self.node()
            self.assertEqual(result.exception.code, ErrorCode.RESOURCE_LIMIT)

    def test_local_key_depends_on_algorithm_and_source_not_database(self):
        args = {"algorithm_version": "normalizer/v1", "native_ref": "#/texts/1", "kind": "PARAGRAPH"}
        self.assertEqual(local_key(**args), local_key(**args))
        self.assertNotEqual(local_key(**args), local_key(**(args | {"algorithm_version": "normalizer/v2"})))

    def test_canonical_digest_independent_of_dict_order(self):
        self.assertEqual(fingerprint({"b": 2, "a": 1}), fingerprint({"a": 1, "b": 2}))


class FigureTests(unittest.TestCase):
    def test_parse_figure_identity_does_not_require_search_projection(self):
        figure = FigureIdentity(key("f"), "图 1-1", "chapter-1")
        self.assertIsNone(figure.key.projection_ref)

    def occurrence(self, text="如图 1-1 所示", scope="chapter-1"):
        return extract_figure_references(text, key("p"), numbering_scope=scope, policy_version="figures/v1")[0]

    def figure(self, name="f", label="图 1-1", scope="chapter-1", **kwargs):
        return FigureIdentity(key(name, **kwargs), label, scope, "anchor-" + name)

    def test_extract_multiple_occurrences_and_original_spans(self):
        text = "如图 1-1 所示，再参见 Figure 2.3(a) 和图 1-1。"
        refs = extract_figure_references(text, key("p"), numbering_scope="chapter-1", policy_version="figures/v1")
        self.assertEqual(len(refs), 3)
        self.assertEqual(len({r.occurrence_key for r in refs}), 3)
        for ref in refs:
            self.assertEqual(text[ref.span.char_start:ref.span.char_end], ref.raw_label)

    def test_dot_and_dash_not_normalized_to_same_number(self):
        self.assertNotEqual(parse_figure_label("图1.1"), parse_figure_label("图1-1"))

    def test_sentence_period_and_english_word_boundary(self):
        text = "config 1 is a variable. See Fig. 1."
        refs = extract_figure_references(text, key("p"), numbering_scope=None, policy_version="figures/v1")
        self.assertEqual([ref.raw_label for ref in refs], ["Fig. 1"])

    def test_ocr_confusable_not_corrected(self):
        self.assertIsNone(parse_figure_label("图1-l"))
        self.assertEqual(extract_figure_references("如图1-l所示", key("p"), numbering_scope=None,
                                                 policy_version="figures/v1"), ())

    def test_scope_disambiguates_reset_numbering(self):
        f1, f2 = self.figure("one"), self.figure("two", scope="chapter-2")
        result = resolve_figure_reference(self.occurrence(), (f1, f2))
        self.assertEqual(result.status, ResolutionStatus.RESOLVED)
        self.assertEqual(result.target_key, f1.key)

    def test_no_scope_and_repeated_number_is_ambiguous(self):
        result = resolve_figure_reference(self.occurrence(scope=None),
                                          (self.figure("a"), self.figure("b", scope="chapter-2")))
        self.assertEqual(result.status, ResolutionStatus.AMBIGUOUS)
        self.assertIsNone(result.target_key)

    def test_old_version_or_parse_not_substituted(self):
        for overrides in ({"document_version_ref": "doc-v2"}, {"parse_run_ref": "run-2"}):
            result = resolve_figure_reference(self.occurrence(), (self.figure(**overrides),))
            self.assertEqual(result.status, ResolutionStatus.MISSING_TARGET)

    def test_other_app_figure_rejected_even_same_number(self):
        with self.assertRaises(AlgorithmError):
            resolve_figure_reference(self.occurrence(), (self.figure(owner_binding_ref="owner-b"),))

    def test_native_anchor_conflicting_label_reported(self):
        ref = replace(self.occurrence(), native_target_anchor="anchor-f")
        result = resolve_figure_reference(ref, (self.figure(label="图 1-2"),))
        self.assertEqual(result.status, ResolutionStatus.CONFLICT)

    def test_native_anchor_missing_not_silently_replaced_by_label(self):
        ref = replace(self.occurrence(), native_target_anchor="missing")
        result = resolve_figure_reference(ref, (self.figure(),))
        self.assertEqual(result.status, ResolutionStatus.MISSING_TARGET)

    def test_subfigure_requires_precise_target(self):
        ref = self.occurrence("如图 1-1(a) 所示")
        self.assertEqual(resolve_figure_reference(ref, (self.figure(),)).status, ResolutionStatus.MISSING_TARGET)
        self.assertEqual(resolve_figure_reference(ref, (self.figure(label="图 1-1(a)"),)).status,
                         ResolutionStatus.RESOLVED)

    def test_valid_native_anchor_has_traceable_basis(self):
        ref = replace(self.occurrence(), native_target_anchor="anchor-f")
        result = resolve_figure_reference(ref, (self.figure(),))
        self.assertEqual(result.basis, "NATIVE_ANCHOR")
        self.assertEqual(result.target_key, key("f"))


class RelationTests(unittest.TestCase):
    def setUp(self):
        self.relations = (
            refers("r1", "p", "f"), refers("r2", "p", "f", 10),
            refers("r3", "p-other", "f"), refers("r4", "p-other", "other-figure"),
            ResolvedRelation("caption", RelationType.CAPTION_OF, key("c"), key("f"), "NATIVE"),
            ResolvedRelation("asset", RelationType.HAS_ASSET, key("f"), key("image"), "NATIVE"))

    def test_text_hit_brings_figure_caption_and_asset(self):
        manifest = direct_figure_manifest(key("p"), anchor_kind="TEXT", relations=self.relations,
                                          relation_pages_exhausted=True)
        self.assertEqual(set(manifest.linked_keys), {key("f"), key("c"), key("image")})
        self.assertEqual({r.relation_key for r in manifest.relations}, {"r1", "r2", "caption", "asset"})
        self.assertEqual(manifest.payload_status, "DEFERRED")

    def test_figure_hit_brings_both_referencing_paragraphs(self):
        manifest = direct_figure_manifest(key("f"), anchor_kind="FIGURE", relations=self.relations,
                                          relation_pages_exhausted=True)
        self.assertEqual(set(manifest.linked_keys), {key("p"), key("p-other"), key("c"), key("image")})
        self.assertNotIn(key("other-figure"), manifest.linked_keys)
        self.assertEqual(len([r for r in manifest.relations if r.relation_type == RelationType.REFERS_TO]), 3)

    def test_caption_hit_also_brings_actual_figure(self):
        manifest = direct_figure_manifest(key("c"), anchor_kind="CAPTION", relations=self.relations,
                                          relation_pages_exhausted=True)
        self.assertIn(key("f"), manifest.linked_keys)
        self.assertIn(key("p"), manifest.linked_keys)

    def test_unfinished_relation_pages_not_reported_complete(self):
        manifest = direct_figure_manifest(key("p"), anchor_kind="TEXT", relations=self.relations,
                                          relation_pages_exhausted=False)
        self.assertEqual(manifest.manifest_status, "PARTIAL")

    def test_subfigure_keeps_parent_and_crop_not_other_panels(self):
        relations = (ResolvedRelation("parent", RelationType.PART_OF, key("panel"), key("f"), "NATIVE"),
                     ResolvedRelation("crop", RelationType.HAS_ASSET, key("panel"), key("crop"), "NATIVE"),
                     ResolvedRelation("other-panel", RelationType.PART_OF, key("panel-b"), key("f"), "NATIVE"))
        manifest = direct_figure_manifest(key("panel"), anchor_kind="REGION", relations=self.relations + relations,
                                          relation_pages_exhausted=True)
        self.assertIn(key("f"), manifest.linked_keys)
        self.assertIn(key("crop"), manifest.linked_keys)
        self.assertNotIn(key("panel-b"), manifest.linked_keys)

    def test_cross_version_relation_cannot_be_constructed(self):
        with self.assertRaises(AlgorithmError):
            ResolvedRelation("bad", RelationType.CAPTION_OF, key("c"),
                             key("f", document_version_ref="new"), "NATIVE")


if __name__ == "__main__":
    unittest.main()
