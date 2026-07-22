import unittest

from agent.agent.document_agent_v2 import DocumentAgentV2
from knowledge_core.application.task_dto import KnowledgeTask
from skills.knowledge_retrieval_v2 import KnowledgeRetrievalSkillV2


class FakeKcClient:
    async def discover(self, **kwargs):
        return {"candidates": [{"collection_id": 1, "bundle_id": 10, "bundle_revision_id": 100}]}

    async def retrieve_evidence(self, **kwargs):
        return {"citations": [{"citation_label": "C1", "primary_evidence_ids": [1]}]}


class DocumentAgentV2Test(unittest.IsolatedAsyncioTestCase):
    async def test_agent_delegates_to_stateless_skill_and_returns_citations(self):
        skill = KnowledgeRetrievalSkillV2(kc_client=FakeKcClient())
        agent = DocumentAgentV2(retrieval_skill=skill)
        result = await agent.retrieve(KnowledgeTask(
            task_id="t1", parent_run_id="r1", domain_id=1, agent_id=2,
            original_query="找案例", standalone_query="找案例", collection_ids=(1,),
        ))
        self.assertEqual(result.status, "READY")
        self.assertEqual(result.citation_pack["citations"][0]["citation_label"], "C1")


if __name__ == "__main__":
    unittest.main()
