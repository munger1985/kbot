"""知识检索 Agent 创建事务顺序测试。"""

import unittest

from knowledge_retrieval_app.application import (
    AgentApplicationError,
    CreateAgentCommand,
    KnowledgeRetrievalAgentService,
    UpdateAgentCommand,
)
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
        if (
            self.agent is not None
            and self.agent.domain_id == domain_id
            and self.agent.agent_id == agent_id
        ):
            return self.agent
        return None

    async def current_version(self, *, agent_id, agent_version_id):
        if (
            self.version is not None
            and self.version.agent_id == agent_id
            and self.version.agent_version_id == agent_version_id
        ):
            return self.version
        return None

    async def next_version_no(self, *, agent_id):
        if self.version is None or self.version.agent_id != agent_id:
            return 1
        return int(self.version.version_no) + 1


class _UnitOfWork:
    def __init__(self, repository: _Repository) -> None:
        self.agents = repository

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def commit(self) -> None:
        if self.agents.agent.current_version_id != (
            self.agents.version.agent_version_id
        ):
            raise AssertionError("提交前必须回填 Agent 当前版本")
        self.agents.events.append("commit")


class AgentCreationTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_resolves_agent_version_foreign_key_cycle(self) -> None:
        repository = _Repository()
        service = KnowledgeRetrievalAgentService(
            uow_factory=lambda: _UnitOfWork(repository)
        )

        result = await service.create(
            CreateAgentCommand(
                domain_id=41,
                display_name="知识助手",
                enabled_capabilities=("document",),
                actor_id="user-41",
            )
        )

        self.assertEqual(["agent", "version", "commit"], repository.events)
        self.assertEqual("知识助手", result["display_name"])
        self.assertEqual(
            result["agent_version_id"],
            str(repository.agent.current_version_id),
        )
        self.assertEqual(result["enabled_capabilities"], ["document"])
        self.assertEqual(
            result["config"]["resource_mode"], "managed_resources"
        )

    async def test_active_multi_capability_requires_router_model(self) -> None:
        service = KnowledgeRetrievalAgentService(
            uow_factory=lambda: _UnitOfWork(_Repository())
        )

        with self.assertRaises(AgentApplicationError) as raised:
            await service.create(
                CreateAgentCommand(
                    domain_id=41,
                    display_name="综合助手",
                    enabled_capabilities=("conversation", "document"),
                    models={"composer_llm": uuid7()},
                    status="ACTIVE",
                    actor_id="user-41",
                )
            )

        self.assertEqual(
            raised.exception.code, "AGENT_ROUTER_MODEL_REQUIRED"
        )

    async def test_update_creates_version_with_selected_capabilities(self) -> None:
        repository = _Repository()
        service = KnowledgeRetrievalAgentService(
            uow_factory=lambda: _UnitOfWork(repository)
        )
        created = await service.create(
            CreateAgentCommand(
                domain_id=41,
                display_name="知识助手",
                enabled_capabilities=("document",),
                actor_id="user-41",
            )
        )

        updated = await service.update(
            UpdateAgentCommand(
                domain_id=41,
                agent_id=repository.agent.agent_id,
                expected_row_version=created["row_version"],
                enabled_capabilities=("data_query", "conversation"),
                actor_id="user-41",
            )
        )

        self.assertEqual(
            updated["enabled_capabilities"],
            ["conversation", "data_query"],
        )
        self.assertEqual(repository.version.version_no, 2)

    async def test_execution_spec_rejects_inactive_agent(self) -> None:
        repository = _Repository()
        service = KnowledgeRetrievalAgentService(
            uow_factory=lambda: _UnitOfWork(repository)
        )
        created = await service.create(
            CreateAgentCommand(
                domain_id=41,
                display_name="未启用助手",
                enabled_capabilities=("document",),
                actor_id="user-41",
            )
        )

        with self.assertRaises(AgentApplicationError) as raised:
            await service.execution_spec(
                domain_id=41,
                agent_id=repository.agent.agent_id,
            )

        self.assertEqual("AGENT_NOT_ACTIVE", raised.exception.code)
        self.assertEqual("DRAFT", created["status"])


if __name__ == "__main__":
    unittest.main()
