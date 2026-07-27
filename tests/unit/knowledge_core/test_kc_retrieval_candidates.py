import unittest

from knowledge_core.application.retrieval import DiscoveryHit, aggregate_candidates


class DiscoveryCandidateAggregationTest(unittest.TestCase):
    def test_document_hits_roll_up_to_one_bundle_and_preserve_member_focus(self):
        hits = [
            DiscoveryHit(1, "a", 10, 100, "DOCUMENT", "member:one", "Bundle A", 1, "TEXT", matched_member_key="member:one", member_count=2),
            DiscoveryHit(1, "a", 10, 100, "DOCUMENT", "member:two", "Bundle A", 2, "VECTOR", matched_member_key="member:two", member_count=2),
            DiscoveryHit(1, "a", 11, 101, "BUNDLE", "bundle", "Bundle B", 1, "TEXT", member_count=1),
        ]
        result = aggregate_candidates(hits)
        self.assertEqual([(item.bundle_id, item.matched_members) for item in result[:2]], [
            (10, ["member:one", "member:two"]), (11, []),
        ])
        self.assertEqual(result[1].candidate_scope, "SINGLE_MEMBER")

    def test_collections_are_interleaved_without_primary_priority(self):
        hits = [
            DiscoveryHit(1, "a", 1, 1, "BUNDLE", "bundle", "A", 1, "TEXT"),
            DiscoveryHit(1, "a", 2, 2, "BUNDLE", "bundle", "A2", 2, "TEXT"),
            DiscoveryHit(2, "b", 3, 3, "BUNDLE", "bundle", "B", 1, "TEXT"),
        ]
        self.assertEqual([item.collection_key for item in aggregate_candidates(hits)], ["a", "b", "a"])


if __name__ == "__main__":
    unittest.main()
