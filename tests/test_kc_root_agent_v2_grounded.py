import unittest
from dataclasses import asdict

from agent.agent.root_agent_v2 import RootAgentV2
from knowledge_core.application.answer_generation import AnswerDraft, AnswerClaim, _parse_draft
from knowledge_core.application.evidence_retrieval import CitationGroup, EvidenceGroupItem, EvidenceHit
from knowledge_core.application.task_dto import KnowledgeTask, KnowledgeTaskResult


def _citation_payload():
    hit = EvidenceHit(
        evidence_id=7, collection_id=1, bundle_id=11, bundle_revision_id=12,
        bundle_revision_document_id=13, document_id=14, document_version_id=15,
        parse_view_id=16, evidence_key="e-7", evidence_type="PARAGRAPH",
        content_text="可靠事实", retrieval_text="可靠事实", heading_path=("概述",),
        locator={"page": 1}, source_spans=(), provenance={}, section_key="s",
        parent_evidence_key=None, ordinal=1, quality_score=1.0, local_rank=1,
        channel="TEXT",
    )
    citation = CitationGroup(
        citation_label="C1", collection_id=1, bundle_id=11,
        bundle_revision_id=12, document_version_id=15, parse_view_id=16,
        primary_evidence_ids=[7], structural_context_ids=[], neighbor_evidence_ids=[],
        items=[EvidenceGroupItem("G1-A1", hit, "ANCHOR", "PRIMARY")],
    )
    return {"citations": [asdict(citation)]}


class FakeDocumentAgent:
    async def retrieve(self, task):
        return KnowledgeTaskResult(task_id=task.task_id, status="READY", citation_pack=_citation_payload())


class FakeAnswerGenerator:
    async def generate(self, *, task, citation_pack):
        return AnswerDraft(
            answer_markdown="可靠事实 [C1] [C99]",
            claims=(AnswerClaim("claim-1", "可靠事实", ("C1",)),),
            used_citation_labels=("C1", "C99"), selected_bundle_ids=(11,),
        )


class RootGroundedTest(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_doc_results_only_include_model_used_verified_docs(self):
        root = RootAgentV2(document_agent=FakeDocumentAgent(), answer_generator=FakeAnswerGenerator())
        task = KnowledgeTask("t", "r", 1, 2, "q", "q", collection_ids=(1,))
        events = [chunk.decode() async for chunk in root.stream(task)]
        joined = "".join(events)
        self.assertIn('event: doc_results_v2', joined)
        self.assertIn('"bundle_id": 11', joined)
        self.assertNotIn("[C99]", joined)
        self.assertIn('"grounding_status": "VERIFIED"', joined)

    def test_llm_json_draft_contract(self):
        draft = _parse_draft('{"answer_markdown":"ok [C1]","claims":[{"claim_id":"x","text":"ok","citation_labels":["C1"]}],"used_citation_labels":["C1"],"selected_bundle_ids":[11]}')
        self.assertEqual(draft.claims[0].claim_id, "x")
        self.assertEqual(draft.used_citation_labels, ("C1",))


if __name__ == "__main__":
    unittest.main()
