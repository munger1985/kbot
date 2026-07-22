import unittest

from knowledge_core.application.evidence_retrieval import CitationGroup, EvidenceGroupItem, EvidenceHit
from knowledge_core.application.grounding import AnswerClaim, AnswerDraft, AnswerGroundingVerifier


def citation(label="C1", bundle=10, evidence_id=1):
    evidence = EvidenceHit(
        evidence_id=evidence_id, collection_id=1, bundle_id=bundle,
        bundle_revision_id=100, bundle_revision_document_id=20,
        document_id=30, document_version_id=40, parse_view_id=50,
        evidence_key=f"e-{evidence_id}", evidence_type="PARAGRAPH",
        content_text="fact", retrieval_text="fact", heading_path=(), locator={"page": 1},
        source_spans=(), provenance={}, section_key="s", parent_evidence_key=None,
        ordinal=1, quality_score=1.0, local_rank=1, channel="TEXT",
    )
    return CitationGroup(
        citation_label=label, collection_id=1, bundle_id=bundle,
        bundle_revision_id=100, document_version_id=40, parse_view_id=50,
        primary_evidence_ids=[evidence_id], structural_context_ids=[],
        neighbor_evidence_ids=[], items=[EvidenceGroupItem(
            item_label="G1-A1", evidence=evidence, input_role="ANCHOR", final_role="PRIMARY",
        )],
    )


class GroundingTest(unittest.TestCase):
    def test_only_used_verified_citations_become_doc_results(self):
        result = AnswerGroundingVerifier().verify(
            draft=AnswerDraft(
                answer_markdown="答案 [C1]", claims=(AnswerClaim("c1", "事实", ("C1",)),),
                used_citation_labels=("C1", "C9"), selected_bundle_ids=(10, 99),
            ), citation_pack=[citation()],
        )
        self.assertEqual(result.grounding_status, "VERIFIED")
        self.assertEqual(result.dropped_citation_labels, ["C9"])
        self.assertNotIn("[C9]", result.answer_markdown)
        self.assertEqual([item.bundle_id for item in result.doc_results_v2], [10])

    def test_missing_claim_citation_is_partial(self):
        result = AnswerGroundingVerifier().verify(
            draft=AnswerDraft(answer_markdown="事实", claims=(AnswerClaim("c1", "事实"),)),
            citation_pack=[citation()],
        )
        self.assertEqual(result.grounding_status, "INSUFFICIENT")
        self.assertEqual(result.doc_results_v2, [])


if __name__ == "__main__":
    unittest.main()
