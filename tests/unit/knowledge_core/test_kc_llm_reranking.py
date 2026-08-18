"""Knowledge Core LLM 对象级与 Evidence Group 重排测试。"""

import unittest
from types import SimpleNamespace

from knowledge_core.application.evidence_retrieval import (
    CitationGroup,
    EvidenceGroupItem,
    EvidenceHit,
)
from knowledge_core.application.llm_reranking import (
    KnowledgeCoreLlmReranker,
)
from knowledge_core.application.retrieval import BundleCandidate
from platform_core.identity import uuid7


class _ModelResolver:
    async def resolve(self, collection_id):
        return uuid7(), "retrieval-llm"


class _PromptResolver:
    async def resolve(self, prompt_key):
        return SimpleNamespace(
            content=f"prompt:{prompt_key}",
            ref=lambda: {"prompt_key": prompt_key, "version": "1.0.0"},
        )


class _ModelClient:
    def __init__(self, responses):
        self._responses = iter(responses)

    async def get_llm_json(self, **kwargs):
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


def _candidate(collection_id, rank, title):
    return BundleCandidate(
        collection_id=collection_id,
        bundle_id=uuid7(),
        bundle_revision_id=uuid7(),
        display_title=title,
        member_count=1,
        matched_members=[],
        match_signals=["TEXT", "VECTOR"],
        local_rank=rank,
        rrf_score=1 / (60 + rank),
        candidate_scope="SINGLE_MEMBER",
        profile_text=f"{title} 的检索画像",
    )


def _citation(
    collection_id,
    *,
    label="C1",
    content="通过建立联合索引将查询耗时从 8 秒降低到 1 秒。",
):
    evidence = EvidenceHit(
        evidence_id=uuid7(),
        collection_id=collection_id,
        bundle_id=uuid7(),
        bundle_revision_id=uuid7(),
        bundle_revision_document_id=uuid7(),
        document_id=uuid7(),
        document_version_id=uuid7(),
        parse_view_id=uuid7(),
        evidence_key="evidence-1",
        evidence_type="TEXT",
        content_text=content,
        retrieval_text="联合索引 查询耗时",
        heading_path=("优化结果",),
        locator={"page": 3},
        locator_schema_version="document/v1",
        source_spans=(),
        provenance={},
        section_key="section-1",
        parent_evidence_key=None,
        ordinal=1,
        quality_score=0.9,
        local_rank=1,
        channel="TEXT",
    )
    item = EvidenceGroupItem(
        item_label="E1",
        evidence=evidence,
        input_role="ANCHOR",
        final_role="PRIMARY",
    )
    return CitationGroup(
        citation_label=label,
        collection_id=collection_id,
        bundle_id=evidence.bundle_id,
        bundle_revision_id=evidence.bundle_revision_id,
        document_version_id=evidence.document_version_id,
        parse_view_id=evidence.parse_view_id,
        primary_evidence_ids=[evidence.evidence_id],
        structural_context_ids=[],
        neighbor_evidence_ids=[],
        items=[item],
    )


class KnowledgeCoreLlmRerankingTest(
    unittest.IsolatedAsyncioTestCase
):
    async def test_candidate_rerank_removes_irrelevant_object(self):
        collection_id = uuid7()
        candidates = [
            _candidate(collection_id, 1, "语义相似但无关"),
            _candidate(collection_id, 2, "数据库优化案例"),
        ]
        reranker = KnowledgeCoreLlmReranker(
            model_resolver=_ModelResolver(),
            model_client=_ModelClient(
                [
                    {
                        "decisions": [
                            {
                                "candidate_label": "B1",
                                "relevance": "IRRELEVANT",
                            },
                            {
                                "candidate_label": "B2",
                                "relevance": "DIRECT",
                            },
                        ]
                    }
                ]
            ),
            prompt_resolver=_PromptResolver(),
        )

        output, report, warnings = await reranker.rerank_candidates(
            query="如何降低数据库查询耗时？",
            candidates=candidates,
        )

        self.assertEqual([item.display_title for item in output], [
            "数据库优化案例"
        ])
        self.assertEqual(report["status"], "SUCCEEDED")
        self.assertEqual(warnings, [])

    async def test_invalid_candidate_labels_fall_back_to_rrf(self):
        collection_id = uuid7()
        candidates = [_candidate(collection_id, 1, "案例")]
        reranker = KnowledgeCoreLlmReranker(
            model_resolver=_ModelResolver(),
            model_client=_ModelClient(
                [{"decisions": [{"candidate_label": "X1",
                                 "relevance": "DIRECT"}]}]
            ),
            prompt_resolver=_PromptResolver(),
        )

        output, report, warnings = await reranker.rerank_candidates(
            query="查询案例", candidates=candidates
        )

        self.assertEqual(output, candidates)
        self.assertEqual(report["status"], "DEGRADED")
        self.assertEqual(len(warnings), 1)

    async def test_breadth_candidate_rerank_preserves_topic_matches(self):
        collection_id = uuid7()
        candidates = [
            _candidate(
                collection_id,
                1,
                "Deep Data Security in an Agentic Application",
            ),
            _candidate(
                collection_id,
                2,
                "Conversational Banking with Select AI Agents",
            ),
            _candidate(collection_id, 3, "OKE Workbench Utility"),
        ]
        reranker = KnowledgeCoreLlmReranker(
            model_resolver=_ModelResolver(),
            model_client=_ModelClient([{
                "decisions": [
                    {
                        "candidate_label": f"B{index}",
                        "relevance": "IRRELEVANT",
                    }
                    for index in range(1, 4)
                ]
            }]),
            prompt_resolver=_PromptResolver(),
        )

        output, report, warnings = await reranker.rerank_candidates(
            query="list all assets related to agent",
            candidates=candidates,
            coverage_mode="BREADTH",
        )

        self.assertEqual(
            [item.display_title for item in output],
            [
                "Deep Data Security in an Agentic Application",
                "Conversational Banking with Select AI Agents",
            ],
        )
        self.assertEqual(
            2, report["collections"][0]["recall_guard_count"]
        )
        self.assertEqual(warnings, [])

    async def test_evidence_rerank_keeps_selected_primary(self):
        collection_id = uuid7()
        citation = _citation(collection_id)
        reranker = KnowledgeCoreLlmReranker(
            model_resolver=_ModelResolver(),
            model_client=_ModelClient(
                [
                    {
                        "decisions": [
                            {
                                "group_label": "C1",
                                "support": "DIRECT_SUPPORT",
                                "primary_item_labels": ["E1"],
                            }
                        ]
                    }
                ]
            ),
            prompt_resolver=_PromptResolver(),
        )

        output, report, warnings = await reranker.rerank_evidence(
            query="优化后耗时是多少？", citations=[citation]
        )

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].primary_evidence_ids,
                         citation.primary_evidence_ids)
        self.assertEqual(report["status"], "SUCCEEDED")
        self.assertEqual(warnings, [])

    async def test_breadth_evidence_rerank_preserves_topic_groups(self):
        collection_id = uuid7()
        citations = [
            _citation(
                collection_id,
                label="C1",
                content="Agentic Application security asset metadata.",
            ),
            _citation(
                collection_id,
                label="C2",
                content="Select AI Agents for conversational banking.",
            ),
            _citation(
                collection_id,
                label="C3",
                content="OKE autoscaling workbench utility.",
            ),
        ]
        reranker = KnowledgeCoreLlmReranker(
            model_resolver=_ModelResolver(),
            model_client=_ModelClient([{
                "decisions": [
                    {
                        "group_label": f"C{index}",
                        "support": "NO_SUPPORT",
                    }
                    for index in range(1, 4)
                ]
            }]),
            prompt_resolver=_PromptResolver(),
        )

        output, report, warnings = await reranker.rerank_evidence(
            query="list all assets related to agent",
            citations=citations,
            coverage_mode="BREADTH",
        )

        self.assertEqual(2, len(output))
        self.assertEqual(["C1", "C2"], [item.citation_label for item in output])
        self.assertEqual(2, report["recall_guard_count"])
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
