import unittest

from knowledge_core.application.evidence_retrieval import EvidenceHit, assemble_groups, build_citation_pack


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


if __name__ == "__main__":
    unittest.main()
