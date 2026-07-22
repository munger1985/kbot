import unittest

from agent.agent.document_agent_v2 import DocumentAgentV2
from agent.agent.root_agent_v2 import RootAgentV2
from knowledge_core.application.task_dto import KnowledgeTask, KnowledgeTaskResult


class FakeDocumentAgent:
    async def retrieve(self, task):
        return KnowledgeTaskResult(task_id=task.task_id, status="READY", citation_pack={"citations": []})


class RootAgentV2Test(unittest.IsolatedAsyncioTestCase):
    async def test_stream_is_v2_only_and_has_terminal_event(self):
        root = RootAgentV2(document_agent=FakeDocumentAgent())
        task = KnowledgeTask("t", "r", 1, 2, "q", "q", collection_ids=(1,))
        events = [chunk.decode() async for chunk in root.stream(task)]
        self.assertTrue(any("citations_v2" in event for event in events))
        self.assertTrue(events[-1].startswith("event: done"))


if __name__ == "__main__":
    unittest.main()
