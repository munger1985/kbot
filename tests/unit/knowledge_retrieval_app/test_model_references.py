"""应用私有 Agent 模型引用边界测试。"""

from types import SimpleNamespace
import unittest

from aiops_agent.application.agents import AIOpsAgentService
from knowledge_retrieval_app.application import KnowledgeRetrievalAgentService
from platform_core.identity import uuid7


class _Repository:
    def __init__(self, rows):
        self._rows = rows

    async def model_references(self, *, model_id):
        del model_id
        return self._rows


class _UnitOfWork:
    def __init__(self, rows):
        self.agents = _Repository(rows)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class PrivateAgentModelReferenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_knowledge_retrieval_reference_includes_domain_and_role(self):
        agent_id = uuid7()
        agent = SimpleNamespace(
            agent_id=agent_id,
            domain_id=101,
            display_name="知识助手",
            status="ACTIVE",
        )
        service = KnowledgeRetrievalAgentService(
            uow_factory=lambda: _UnitOfWork([(agent, "composer_llm")])
        )

        references = await service.model_references(model_id=uuid7())

        self.assertEqual("knowledge-retrieval-app", references[0]["service"])
        self.assertEqual("101", references[0]["domain_id"])
        self.assertEqual("composer_llm", references[0]["binding_role"])
        self.assertEqual(str(agent_id), references[0]["resource_id"])

    async def test_aiops_reference_includes_domain_and_image_role(self):
        agent_id = uuid7()
        agent = SimpleNamespace(
            agent_id=agent_id,
            domain_id=202,
            display_name="运维助手",
            status="ACTIVE",
        )
        service = AIOpsAgentService(
            uow_factory=lambda: _UnitOfWork([(agent, "image_ocr")])
        )

        references = await service.model_references(model_id=uuid7())

        self.assertEqual("aiops-agent", references[0]["service"])
        self.assertEqual("202", references[0]["domain_id"])
        self.assertEqual("image_ocr", references[0]["binding_role"])
        self.assertEqual(str(agent_id), references[0]["resource_id"])


if __name__ == "__main__":
    unittest.main()
