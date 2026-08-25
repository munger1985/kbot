import unittest
from uuid import UUID

from knowledge_core.application.retrieval import (
    DiscoveryHit,
    KnowledgeCoreDiscoveryService,
    aggregate_candidates,
)


COLLECTION_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a001")
BUNDLE_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a002")
REVISION_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a003")


class _DiscoveryPort:
    def __init__(self):
        self.bundle_revision_ids = ()

    async def search_text(self, **kwargs):
        self.bundle_revision_ids = kwargs.get("bundle_revision_ids") or ()
        return [
            DiscoveryHit(
                COLLECTION_ID,
                BUNDLE_ID,
                REVISION_ID,
                "BUNDLE",
                "bundle",
                "示例",
                1,
                "TEXT",
            )
        ]

    async def search_vector(self, **kwargs):
        return [
            DiscoveryHit(
                COLLECTION_ID,
                BUNDLE_ID,
                REVISION_ID,
                "BUNDLE",
                "bundle",
                "示例",
                1,
                "VECTOR",
            )
        ]


class _FailingTextDiscoveryPort(_DiscoveryPort):
    async def search_text(self, **kwargs):
        raise RuntimeError("全文索引不可用")


class DiscoveryCandidateAggregationTest(unittest.TestCase):
    def test_document_hits_roll_up_to_one_bundle_and_preserve_member_focus(self):
        hits = [
            DiscoveryHit(1, 10, 100, "DOCUMENT", "member:one", "Bundle A", 1, "TEXT", matched_member_key="member:one", member_count=2),
            DiscoveryHit(1, 10, 100, "DOCUMENT", "member:two", "Bundle A", 2, "VECTOR", matched_member_key="member:two", member_count=2),
            DiscoveryHit(1, 11, 101, "BUNDLE", "bundle", "Bundle B", 1, "TEXT", member_count=1),
        ]
        result = aggregate_candidates(hits)
        self.assertEqual([(item.bundle_id, item.matched_members) for item in result[:2]], [
            (10, ["member:one", "member:two"]), (11, []),
        ])
        self.assertEqual(result[1].candidate_scope, "SINGLE_MEMBER")

    def test_collections_are_interleaved_without_primary_priority(self):
        hits = [
            DiscoveryHit(1, 1, 1, "BUNDLE", "bundle", "A", 1, "TEXT"),
            DiscoveryHit(1, 2, 2, "BUNDLE", "bundle", "A2", 2, "TEXT"),
            DiscoveryHit(2, 3, 3, "BUNDLE", "bundle", "B", 1, "TEXT"),
        ]
        self.assertEqual(
            [item.collection_id for item in aggregate_candidates(hits)],
            [1, 2, 1],
        )


class DiscoveryDiagnosticsTest(unittest.IsolatedAsyncioTestCase):
    async def test_diagnostics_separate_text_vector_and_bundle_counts(self):
        port = _DiscoveryPort()
        service = KnowledgeCoreDiscoveryService(search_port=port)
        candidates, diagnostics = await service.discover_with_diagnostics(
            collection_ids=[COLLECTION_ID],
            query="数据库性能",
            query_vectors={COLLECTION_ID: [0.1, 0.2]},
            bundle_revision_ids=[REVISION_ID],
        )
        self.assertEqual(1, len(candidates))
        self.assertEqual(1, diagnostics["text_hits"])
        self.assertEqual(1, diagnostics["vector_hits"])
        self.assertEqual(2, diagnostics["raw_hits"])
        self.assertEqual(1, diagnostics["bundle_candidates"])
        self.assertEqual((REVISION_ID,), tuple(port.bundle_revision_ids))
        self.assertEqual(1, diagnostics["revision_scope_count"])

    async def test_vector_channel_survives_text_channel_failure(self):
        service = KnowledgeCoreDiscoveryService(
            search_port=_FailingTextDiscoveryPort()
        )
        candidates, diagnostics = await service.discover_with_diagnostics(
            collection_ids=[COLLECTION_ID],
            query="数据库性能",
            query_vectors={COLLECTION_ID: [0.1, 0.2]},
        )
        self.assertEqual(1, len(candidates))
        self.assertEqual(0, diagnostics["text_hits"])
        self.assertEqual(1, diagnostics["vector_hits"])
        self.assertEqual(
            "RuntimeError",
            diagnostics["collections"][0]["text_error"],
        )
        self.assertTrue(diagnostics["warnings"])


if __name__ == "__main__":
    unittest.main()
