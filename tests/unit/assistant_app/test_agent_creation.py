"""智能工作台 Agent 的绑定与版本测试。"""

import unittest

from assistant_app.application import AgentApplicationError, AssistantAgentService, CreateAgentCommand, UpdateAgentCommand
from platform_core.identity import uuid7


class _Repository:
    def __init__(self) -> None:
        self.agent = None
        self.version = None
        self.events: list[str] = []

    async def add_agent(self, row) -> None:
        self.events.append("agent")
        if row.current_version_id is not None:
            raise AssertionError("首次插入 Agent 时不得引用尚未创建的版本")
        row.row_version = 1
        self.agent = row

    async def add_version(self, row) -> None:
        self.events.append("version")
        if self.agent is None or row.agent_id != self.agent.agent_id:
            raise AssertionError("创建 Version 前必须先创建所属 Agent")
        self.version = row

    async def get(self, *, domain_id, agent_id, lock=False):
        del lock
        if self.agent is not None and self.agent.domain_id == domain_id and self.agent.agent_id == agent_id:
            return self.agent
        return None

    async def current_version(self, *, agent_id, agent_version_id):
        if self.version is not None and self.version.agent_id == agent_id and self.version.agent_version_id == agent_version_id:
            return self.version
        return None

    async def next_version_no(self, *, agent_id):
        if self.version is None or self.version.agent_id != agent_id:
            return 1
        return int(self.version.version_no) + 1

    async def list(self, *, domain_id):
        return [self.agent] if self.agent is not None and self.agent.domain_id == domain_id else []


class _UnitOfWork:
    def __init__(self, repository: _Repository) -> None:
        self.agents = repository

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def commit(self) -> None:
        if self.agents.agent.current_version_id != self.agents.version.agent_version_id:
            raise AssertionError("提交前必须回填 Agent 当前版本")
        self.agents.events.append("commit")


class AssistantAgentCreationTest(unittest.IsolatedAsyncioTestCase):
    async def test_draft_can_be_created_before_knowledge_binding(self) -> None:
        repository = _Repository()
        service = AssistantAgentService(uow_factory=lambda: _UnitOfWork(repository))

        result = await service.create(CreateAgentCommand(domain_id=41, display_name="销售知识助手", actor_id="user-41"))

        self.assertEqual(["agent", "version", "commit"], repository.events)
        self.assertEqual("DRAFT", result["status"])
        self.assertIsNone(result["knowledge_core_id"])
        self.assertEqual(["conversation", "document"], result["enabled_capabilities"])

    async def test_active_agent_requires_a_knowledge_core(self) -> None:
        service = AssistantAgentService(uow_factory=lambda: _UnitOfWork(_Repository()))

        with self.assertRaises(AgentApplicationError) as raised:
            await service.create(CreateAgentCommand(domain_id=41, display_name="无知识库", status="ACTIVE", actor_id="user-41"))

        self.assertEqual("AGENT_KNOWLEDGE_CORE_REQUIRED", raised.exception.code)

    async def test_data_model_binding_creates_a_new_version(self) -> None:
        repository = _Repository()
        service = AssistantAgentService(uow_factory=lambda: _UnitOfWork(repository))
        knowledge_core_id, data_model_id = uuid7(), uuid7()
        created = await service.create(CreateAgentCommand(domain_id=41, display_name="分析助手", knowledge_core_id=knowledge_core_id, actor_id="user-41"))

        updated = await service.update(UpdateAgentCommand(domain_id=41, agent_id=repository.agent.agent_id, expected_row_version=created["row_version"], data_model_ids=(data_model_id,), actor_id="user-41"))

        self.assertEqual(2, repository.version.version_no)
        self.assertEqual([str(data_model_id)], updated["data_model_ids"])
        self.assertEqual(["conversation", "document", "data_query"], updated["enabled_capabilities"])


if __name__ == "__main__":
    unittest.main()
