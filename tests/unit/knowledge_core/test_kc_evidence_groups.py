import unittest
from uuid import UUID

from knowledge_core.application.evidence_retrieval import (
    EvidenceHit,
    EvidenceScope,
    KnowledgeCoreEvidenceRetrievalService,
    assemble_groups,
    build_citation_pack,
)


def hit(evidence_id, version=1, view=1, rank=1, content="fact"):
    return EvidenceHit(
        evidence_id=evidence_id, collection_id=1, bundle_id=10,
        bundle_revision_id=100, bundle_revision_document_id=20,
        document_id=30, document_version_id=version, parse_view_id=view,
        evidence_key=f"e-{evidence_id}", evidence_type="PARAGRAPH",
        content_text=content, retrieval_text=content, heading_path=("Overview",),
        locator={"page": evidence_id}, source_spans=(), provenance={},
        section_key="s1", parent_evidence_key=None, ordinal=evidence_id,
        quality_score=0.9, local_rank=rank, channel="TEXT",
    )


class EvidenceGroupTest(unittest.TestCase):
    def test_groups_do_not_cross_versions_and_context_is_not_primary(self):
        groups = assemble_groups([hit(1), hit(2, version=2)], [hit(3)])
        self.assertEqual(len(groups), 2)
        self.assertIn("STRUCTURAL_CONTEXT", {item.final_role for item in groups[0].items})
        citations = build_citation_pack(groups)
        self.assertEqual([citation.citation_label for citation in citations], ["C1", "C2"])


class _EvidencePort:
    async def search_text(self, **kwargs):
        return [hit(1)]

    async def search_vector(self, **kwargs):
        return [hit(2, rank=2)]

    async def expand_context(self, **kwargs):
        return [hit(3, rank=3)]


class _FailingTextEvidencePort(_EvidencePort):
    async def search_text(self, **kwargs):
        raise RuntimeError("全文索引不可用")


class EvidenceDiagnosticsTest(unittest.IsolatedAsyncioTestCase):
    async def test_diagnostics_record_anchor_context_and_citation_counts(self):
        collection_id = UUID(
            "019f8eae-2c25-7d48-b044-350ec3f5a001"
        )
        service = KnowledgeCoreEvidenceRetrievalService(
            search_port=_EvidencePort()
        )
        citations, diagnostics = await service.retrieve_with_diagnostics(
            scopes=[
                EvidenceScope(
                    collection_id=collection_id,
                    bundle_id=10,
                    bundle_revision_id=100,
                )
            ],
            query="数据库性能",
            query_vectors={collection_id: [0.1, 0.2]},
        )
        self.assertEqual(1, diagnostics["text_hits"])
        self.assertEqual(1, diagnostics["vector_hits"])
        self.assertEqual(2, diagnostics["selected_anchors"])
        self.assertEqual(1, diagnostics["expanded_contexts"])
        self.assertEqual(1, diagnostics["citation_groups"])
        self.assertEqual(1, len(citations))

    async def test_vector_channel_survives_text_channel_failure(self):
        collection_id = UUID(
            "019f8eae-2c25-7d48-b044-350ec3f5a001"
        )
        service = KnowledgeCoreEvidenceRetrievalService(
            search_port=_FailingTextEvidencePort()
        )
        citations, diagnostics = await service.retrieve_with_diagnostics(
            scopes=[
                EvidenceScope(
                    collection_id=collection_id,
                    bundle_id=10,
                    bundle_revision_id=100,
                )
            ],
            query="数据库性能",
            query_vectors={collection_id: [0.1, 0.2]},
        )
        self.assertEqual(0, diagnostics["text_hits"])
        self.assertEqual(1, diagnostics["vector_hits"])
        self.assertEqual(
            "RuntimeError",
            diagnostics["scopes"][0]["text_error"],
        )
        self.assertEqual(1, len(citations))


if __name__ == "__main__":
    unittest.main()
