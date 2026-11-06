"""知识算法身份守恒、全局通道归并、原生评分、覆盖选择和排序反例。"""

from dataclasses import replace
import itertools
import json
from pathlib import Path
import random
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
BASELINE = json.loads((ROOT / "tests/evaluation/knowledge_algorithms/baseline-v1.json").read_text(encoding="utf-8"))
PARAMETERS = BASELINE["parameters"]
sys.path.insert(0, str(ROOT / "packages/knowledge_algorithms/src"))

from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode
from knowledge_algorithms.contracts.identity import (
    AppBackendBinding, AuthorizedScope, EvidenceKey, Locator, RetrievalSnapshot, SourceSpan,
)
from knowledge_algorithms.contracts.query import BackendCapabilities, OrderBy, Predicate, QueryInput, QueryMode
from knowledge_algorithms.ports.ordered import OrderedPage
from knowledge_algorithms.retrieval.fusion import ChannelShard, NativeHit, merge_channel_shards, reciprocal_rank_fusion
from knowledge_algorithms.retrieval.maxsim import MaxSimSpec, TokenMatrix, maxsim
from knowledge_algorithms.retrieval.ordered import retrieve_ordered
from knowledge_algorithms.retrieval.selection import (
    CandidateTargetMatch, CoverageTarget, EvidenceGroup, MatchStatus, PayloadCost, PayloadUnit,
    select_evidence_groups,
)


def key(name, **overrides):
    return EvidenceKey(**({"owner_binding_ref": "owner-a", "document_version_ref": "doc-v1",
                          "parse_run_ref": "parse-1", "object_key": name,
                          "projection_ref": "projection-1"} | overrides))


SNAPSHOT = RetrievalSnapshot("owner-a", "scope-hash", "snapshot-1", "index-1", "withdrawal-1")


def shard(name, hits, **overrides):
    return ChannelShard(**({"channel_key": name, "signal_family": name, "space_id": name,
                           "scoring_contract": "native/v1", "statistics_ref": "global-stats",
                           "snapshot": SNAPSHOT, "hits": tuple(NativeHit(key(k), v) for k, v in hits)} | overrides))


def group(name, cost, *attachments):
    return EvidenceGroup(name, key(name), (PayloadUnit(key(name), PayloadCost(cost, 0, cost)),) + attachments)


def target(name, required=True):
    return CoverageTarget(name, "SUBQUESTION", required, "user:0:10", "plan-1")


def match(name, obj, status=MatchStatus.MATCHED):
    return CandidateTargetMatch(name, key(obj), status, ("native-anchor",), "fixture/v1")


class IdentityTests(unittest.TestCase):
    def test_evidence_span_cannot_point_to_different_source(self):
        with self.assertRaises(AlgorithmError):
            key("p", source_span=SourceSpan("other", 0, 2))

    def test_precise_identity_keeps_same_page_objects(self):
        values = {key("page"), key("figure-a"), key("figure-b"), key("region")}
        self.assertEqual(len(values), 4)

    def test_version_parse_projection_and_span_are_identity(self):
        base = key("p")
        variants = [base, replace(base, parse_run_ref="new"), replace(base, projection_ref="new"),
                    replace(base, document_version_ref="new"), replace(base, owner_binding_ref="other"),
                    replace(base, source_span=SourceSpan("p", 0, 2))]
        self.assertEqual(len({v.stable_key for v in variants}), len(variants))

    def test_empty_identity_rejected(self):
        with self.assertRaises(AlgorithmError):
            key("")

    def test_other_app_scope_rejected(self):
        scope = AuthorizedScope("owner-b", "ref", "scope-hash", "auth-1")
        with self.assertRaises(AlgorithmError) as result:
            SNAPSHOT.validate_scope(AppBackendBinding("owner-a", "a"), scope)
        self.assertEqual(result.exception.code, ErrorCode.SCOPE_CLOSURE_BROKEN)

    def test_scope_snapshot_mismatch_rejected(self):
        with self.assertRaises(AlgorithmError) as result:
            SNAPSHOT.validate_scope(AppBackendBinding("owner-a", "a"),
                                    AuthorizedScope("owner-a", "ref", "other", "auth-1"))
        self.assertEqual(result.exception.code, ErrorCode.SNAPSHOT_STALE)

    def test_real_logical_locator_needs_no_fake_page(self):
        self.assertIsNone(Locator(logical_ref="sheet:1!A1").page_no)

    def test_invalid_locators_rejected(self):
        for kwargs in ({}, {"page_no": 0}, {"page_no": True},
                       {"page_no": 1, "logical_ref": "p"},
                       {"page_no": 1, "bbox": (0, 0, float("nan"), 1)},
                       {"page_no": 1, "bbox": (0, 0, 0, 1)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(AlgorithmError):
                Locator(**kwargs)


class FusionTests(unittest.TestCase):
    def test_search_candidate_requires_exact_projection(self):
        with self.assertRaises(AlgorithmError):
            NativeHit(replace(key("p"), projection_ref=None), 1)

    def test_global_merge_before_fusion(self):
        merged = merge_channel_shards((shard("lexical", [("a", 1)]), shard("lexical", [("b", 4)])))
        self.assertEqual([h.key.object_key for h in merged.hits], ["b", "a"])

    def test_shard_partition_and_return_order_invariance(self):
        rng = random.Random(PARAMETERS["partition_seed"])
        rows = [(str(i), float(i % 7)) for i in range(35)]
        baseline = merge_channel_shards((shard("dense", rows),))
        for size in range(1, 8):
            for _ in range(6):
                rng.shuffle(rows)
                pieces = [shard("dense", rows[i::size]) for i in range(size)]
                rng.shuffle(pieces)
                self.assertEqual(merge_channel_shards(tuple(pieces)), baseline)

    def test_incomparable_local_bm25_statistics_rejected(self):
        with self.assertRaises(AlgorithmError) as result:
            merge_channel_shards((shard("lexical", [("a", 4)]),
                                  shard("lexical", [("b", 8)], statistics_ref="local")))
        self.assertEqual(result.exception.code, ErrorCode.INCOMPARABLE_CHANNEL)

    def test_different_spaces_or_snapshots_rejected(self):
        for override in ({"space_id": "different"}, {"snapshot": replace(SNAPSHOT, index_generation="new")}):
            with self.subTest(override=override), self.assertRaises(AlgorithmError):
                merge_channel_shards((shard("dense", [("a", 1)]), shard("dense", [("b", 2)], **override)))

    def test_duplicate_hit_only_once(self):
        merged = merge_channel_shards((shard("dense", [("a", 1)]), shard("dense", [("a", 1)])))
        self.assertEqual(len(merged.hits), 1)

    def test_conflicting_duplicate_score_not_silently_merged(self):
        with self.assertRaises(AlgorithmError):
            merge_channel_shards((shard("dense", [("a", 1)]), shard("dense", [("a", 2)])))

    def test_rrf_preserves_all_identities_without_threshold(self):
        channels = tuple(merge_channel_shards((shard(name, rows),)) for name, rows in (
            ("lexical", [("a", 1e9), ("b", -1e9)]), ("dense", [("b", 0.02), ("c", -0.9)])))
        fused = reciprocal_rank_fusion(channels, weights={"lexical": 1, "dense": 1}, constant=60)
        self.assertEqual(fused[0].key.object_key, "b")
        self.assertEqual({h.key for h in fused}, {key("a"), key("b"), key("c")})
        self.assertEqual(len(fused), 3)

    def test_rrf_does_not_depend_on_raw_score_scale(self):
        first = merge_channel_shards((shard("a", [("p", 0.2), ("q", -0.1)]),))
        second = merge_channel_shards((shard("a", [("p", 2000), ("q", -1000)]),))
        a = reciprocal_rank_fusion((first,), weights={"a": 1}, constant=30)
        b = reciprocal_rank_fusion((second,), weights={"a": 1}, constant=30)
        self.assertEqual([(v.key, v.score) for v in a], [(v.key, v.score) for v in b])

    def test_duplicate_signal_family_not_extra_vote(self):
        a = merge_channel_shards((shard("fulltext", [("a", 1)], signal_family="lexical"),))
        b = merge_channel_shards((shard("scan", [("a", 1)], signal_family="lexical"),))
        with self.assertRaises(AlgorithmError):
            reciprocal_rank_fusion((a, b), weights={"fulltext": 1, "scan": 1}, constant=60)

    def test_approximation_and_truncation_remain_visible(self):
        channel = merge_channel_shards((shard("a", [("p", 1)], approximate=True, exhausted=False),))
        self.assertTrue(channel.approximate)
        self.assertFalse(channel.exhausted)

    def test_fusion_order_invariance(self):
        channels = tuple(merge_channel_shards((shard(name, [(name, 1), ("common", 0)]),))
                         for name in ("a", "b", "c"))
        expected = reciprocal_rank_fusion(channels, weights={"a": 1, "b": 2, "c": 3}, constant=60)
        for order in itertools.permutations(channels):
            self.assertEqual(reciprocal_rank_fusion(order, weights={"a": 1, "b": 2, "c": 3}, constant=60), expected)

    def test_nonfinite_scores_rejected(self):
        for score in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(score=score), self.assertRaises(AlgorithmError):
                NativeHit(key("a"), score)

    def test_invalid_weights_rejected(self):
        channel = merge_channel_shards((shard("a", [("p", 1)]),))
        for weights in ({}, {"a": 0}, {"a": float("nan")}, {"a": -1}):
            with self.subTest(weights=weights), self.assertRaises(AlgorithmError):
                reciprocal_rank_fusion((channel,), weights=weights, constant=60)

    def test_empty_is_legitimate_not_exception(self):
        channel = merge_channel_shards((shard("a", []),))
        self.assertEqual(reciprocal_rank_fusion((channel,), weights={"a": 1}, constant=60), ())


class MaxSimTests(unittest.TestCase):
    spec = MaxSimSpec("space-1", 2, "DOT", frozenset({"TEXT_TO_PAGE"}))

    def matrix(self, rows, mask=None, space="space-1"):
        return TokenMatrix(space, tuple(tuple(row) for row in rows), tuple(mask or [True] * len(rows)))

    def test_late_interaction_not_mean_pooling(self):
        matrix = self.matrix([(1, 0), (0, 1)])
        self.assertEqual(maxsim(matrix, matrix, spec=self.spec, direction="TEXT_TO_PAGE"), 2)

    def test_padding_never_participates(self):
        q = self.matrix([(1, 0), (999, 999)], [True, False])
        d = self.matrix([(0.5, 0), (999, 999)], [True, False])
        self.assertEqual(maxsim(q, d, spec=self.spec, direction="TEXT_TO_PAGE"), 0.5)

    def test_negative_similarity_not_clipped_to_zero(self):
        q, d = self.matrix([(1, 0)]), self.matrix([(-1, 0)])
        self.assertEqual(maxsim(q, d, spec=self.spec, direction="TEXT_TO_PAGE"), -1)

    def test_query_weights_applied_without_length_average(self):
        matrix = self.matrix([(1, 0), (0, 1)])
        self.assertEqual(maxsim(matrix, matrix, spec=self.spec, direction="TEXT_TO_PAGE", query_weights=(2, 3)), 5)

    def test_cosine_native_normalization(self):
        spec = replace(self.spec, similarity="COSINE")
        self.assertAlmostEqual(maxsim(self.matrix([(2, 0)]), self.matrix([(8, 0)]),
                                      spec=spec, direction="TEXT_TO_PAGE"), 1)

    def test_wrong_space_rejected(self):
        with self.assertRaises(AlgorithmError) as result:
            maxsim(self.matrix([(1, 0)], space="other"), self.matrix([(1, 0)]),
                   spec=self.spec, direction="TEXT_TO_PAGE")
        self.assertEqual(result.exception.code, ErrorCode.MODEL_SPACE_MISMATCH)

    def test_unverified_image_direction_rejected(self):
        with self.assertRaises(AlgorithmError) as result:
            maxsim(self.matrix([(1, 0)]), self.matrix([(1, 0)]), spec=self.spec, direction="IMAGE_TO_IMAGE")
        self.assertEqual(result.exception.code, ErrorCode.MODALITY_UNAVAILABLE)

    def test_zero_cosine_norm_rejected(self):
        with self.assertRaises(AlgorithmError):
            maxsim(self.matrix([(0, 0)]), self.matrix([(1, 0)]),
                   spec=replace(self.spec, similarity="COSINE"), direction="TEXT_TO_PAGE")

    def test_invalid_tensor_shapes_and_nan_rejected(self):
        for rows, mask in (((), ()), (((1, 0),), (False,)), (((1, 0), (1,)), (True, True)),
                           (((float("nan"), 0),), (True,))):
            with self.subTest(rows=rows), self.assertRaises(AlgorithmError):
                TokenMatrix("space-1", rows, mask)

    def test_overflow_rejected(self):
        with self.assertRaises(AlgorithmError):
            maxsim(self.matrix([(1e308, 1e308)]), self.matrix([(1e308, 1e308)]),
                   spec=self.spec, direction="TEXT_TO_PAGE")


class SelectionTests(unittest.TestCase):
    def choose(self, groups, targets=(), matches=(), budget=10, **kwargs):
        return select_evidence_groups(tuple(groups), targets=tuple(targets), matches=tuple(matches),
                                     budget=PayloadCost(budget, PARAMETERS["payload_budget"]["images"],
                                                        PARAMETERS["payload_budget"]["byte_count"]),
                                     max_cover_states=kwargs.get("states", PARAMETERS["max_cover_states"]))

    def test_large_high_rank_cannot_block_smaller_same_target(self):
        result = self.choose([group("a-big", 8), group("b", 5), group("a-small", 4)],
                             [target("a"), target("b")],
                             [match("a", "a-big"), match("a", "a-small"), match("b", "b")])
        self.assertEqual(result.selected_group_keys, ("b", "a-small"))
        self.assertTrue(all(c.status == "MATCHED" for c in result.coverage))

    def test_unknown_does_not_clear_candidates(self):
        result = self.choose([group("p", 2)], [target("question")],
                             [match("question", "p", MatchStatus.UNKNOWN)])
        self.assertEqual(result.selected_group_keys, ("p",))
        self.assertEqual(result.coverage[0].status, "UNKNOWN")

    def test_conflict_not_counted_as_coverage(self):
        result = self.choose([group("p", 2)], [target("q")], [match("q", "p", MatchStatus.CONFLICT)])
        self.assertEqual(result.coverage[0].status, "NO_CONFIRMED_MATCH")

    def test_oversized_first_item_does_not_force_zero(self):
        result = self.choose([group("large", 100), group("small", 2)])
        self.assertEqual(result.selected_group_keys, ("small",))
        self.assertEqual(result.omitted, (("large", "CONTEXT_BUDGET_INSUFFICIENT"),))

    def test_all_oversized_has_explicit_reason(self):
        result = self.choose([group("large", 100)])
        self.assertEqual(result.selected_group_keys, ())
        self.assertTrue(result.omitted)

    def test_shared_figure_charged_once_but_occurrences_retained(self):
        attachment = PayloadUnit(key("figure"), PayloadCost(2, 1, 10))
        a, b = group("p1", 2, attachment), group("p2", 2, attachment)
        result = self.choose([replace(a, relation_keys=("r1",)), replace(b, relation_keys=("r2",))])
        self.assertEqual(result.selected_group_keys, ("p1", "p2"))
        self.assertEqual(result.cost.tokens, 6)
        self.assertEqual(result.cost.images, 1)
        self.assertEqual(len(result.selected_evidence_keys), 3)

    def test_multiple_same_document_evidence_kept(self):
        result = self.choose([group("a", 2), group("b", 2), group("c", 2)])
        self.assertEqual(len(result.selected_group_keys), 3)

    def test_mandatory_attachment_counts_before_selection(self):
        figure = PayloadUnit(key("figure"), PayloadCost(9, 1, 10))
        result = self.choose([group("p", 2, figure), group("small", 2)])
        self.assertEqual(result.selected_group_keys, ("small",))

    def test_target_mapping_must_have_evidence_and_basis(self):
        with self.assertRaises(AlgorithmError):
            self.choose([group("p", 2)], [target("q")], [match("q", "unknown")])
        with self.assertRaises(AlgorithmError):
            CandidateTargetMatch("q", key("p"), MatchStatus.MATCHED, (), "fixture")

    def test_bounded_search_disclosed(self):
        result = self.choose([group("p", 2)], [target("q")], [match("q", "p")], states=1)
        self.assertTrue(result.coverage_search_truncated)
        self.assertEqual(result.selected_group_keys, ("p",))

    def test_empty_legal_result(self):
        result = self.choose([])
        self.assertEqual(result.selected_evidence_keys, ())

    def test_cross_app_candidates_rejected(self):
        other = key("other", owner_binding_ref="owner-b")
        with self.assertRaises(AlgorithmError):
            EvidenceGroup("bad", key("p"), (PayloadUnit(key("p"), PayloadCost(1, 0, 1)),
                                            PayloadUnit(other, PayloadCost(1, 0, 1))))


class OrderedTests(unittest.IsolatedAsyncioTestCase):
    capabilities = BackendCapabilities(frozenset({"category"}), frozenset({"EQ"}),
                                       frozenset({"date", "id"}), "id",
                                       frozenset({QueryMode.FILTERED_ORDER, QueryMode.EXHAUSTIVE}))

    def query(self, **overrides):
        return QueryInput(**({"mode": QueryMode.FILTERED_ORDER, "requested_count": 2,
                              "output_types": ("TEXT",), "predicates": (Predicate("category", "EQ", "paper"),),
                              "order_by": (OrderBy("date", True), OrderBy("id", False))} | overrides))

    async def test_latest_n_from_complete_eligible_scope_not_dense_topk(self):
        calls = []
        capabilities = self.capabilities

        class FakeBackend:
            binding = AppBackendBinding("owner-a", "a")

            async def snapshot(self, scope):
                calls.append("snapshot")
                return SNAPSHOT

            async def search_ordered(self, query, scope, snapshot):
                calls.append("ordered")
                rows = [("old", 1, 0.99), ("new", 4, 0.01), ("middle", 3, 0.02)]
                eligible = sorted(rows, key=lambda row: (-row[1], row[0]))
                return OrderedPage(snapshot, query.query_hash,
                                   tuple(key(row[0]) for row in eligible[:query.requested_count]), False, "opaque-cursor")

            async def search_dense(self, *args):
                raise AssertionError("结构化排序不得先取 dense Top-K")

        backend = FakeBackend()
        backend.capabilities = capabilities
        page = await retrieve_ordered(backend, self.query(), AuthorizedScope("owner-a", "ref", "scope-hash", "auth-1"))
        self.assertEqual([hit.object_key for hit in page.hits], ["new", "middle"])
        self.assertEqual(calls, ["snapshot", "ordered"])

    def test_semantic_query_cannot_claim_exhaustive(self):
        with self.assertRaises(AlgorithmError):
            self.query(mode=QueryMode.EXHAUSTIVE, text="所有相关论文")

    def test_relevance_cannot_hide_date_sort(self):
        with self.assertRaises(AlgorithmError):
            self.query(mode=QueryMode.RELEVANCE, text="论文")

    def test_stable_tie_breaker_required(self):
        with self.assertRaises(AlgorithmError):
            self.capabilities.validate(self.query(order_by=(OrderBy("date", True),)))

    def test_unregistered_field_rejected(self):
        with self.assertRaises(AlgorithmError):
            self.capabilities.validate(self.query(predicates=(Predicate("raw_sql", "EQ", "select"),)))

    def test_multi_image_requires_explicit_semantics(self):
        with self.assertRaises(AlgorithmError):
            QueryInput(QueryMode.RELEVANCE, 2, ("FIGURE",), image_handles=("i1", "i2"))

    def test_requested_count_part_of_query_identity(self):
        self.assertNotEqual(self.query().query_hash, self.query(requested_count=3).query_hash)


if __name__ == "__main__":
    unittest.main()
