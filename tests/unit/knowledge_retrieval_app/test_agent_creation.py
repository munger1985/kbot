"""知识检索 Agent 版本、执行规格与 X Search 模型约束测试。"""

import unittest
from uuid import UUID

from knowledge_retrieval_app.application import (
    AgentApplicationError,
    CreateAgentCommand,
    KnowledgeRetrievalAgentService,
)
from platform_core.contracts import AgentExecutionSpec
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
        self.version = row

    async def get(self, *, domain_id, agent_id, lock=False):
        del lock
        if self.agent is not None and int(self.agent.domain_id) == domain_id and self.agent.agent_id == agent_id:
            return self.agent
        return None

    async def current_version(self, *, agent_id, agent_version_id):
        if self.version is not None and self.version.agent_id == agent_id and self.version.agent_version_id == agent_version_id:
            return self.version
        return None

    async def next_version_no(self, *, agent_id):
        del agent_id
        return int(self.version.version_no) + 1 if self.version else 1

    async def list(self, *, domain_id):
        if self.agent is not None and int(self.agent.domain_id) == domain_id:
            return [self.agent]
        return []


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


class _Catalog:
    def __init__(self, row):
        self.row = row

    async def get_model(self, model_id):
        if str(model_id) != str(self.row["model_id"]):
            raise LookupError(model_id)
        return self.row


def _active_models(model_id):
    return {
        "context_llm": model_id,
        "composer_llm": model_id,
        "memory_llm": model_id,
        "memory_embedding": model_id,
        "router_llm": model_id,
    }


class AgentCreationTest(unittest.IsolatedAsyncioTestCase):
    async def test_draft_can_be_created_before_knowledge_binding(self) -> None:
        repository = _Repository()
        service = KnowledgeRetrievalAgentService(
            uow_factory=lambda: _UnitOfWork(repository)
        )

        result = await service.create(CreateAgentCommand(
            domain_id=41, display_name="销售知识助手", actor_id="user-41"
        ))

        self.assertEqual(["agent", "version", "commit"], repository.events)
        self.assertEqual("DRAFT", result["status"])
        self.assertIsNone(result["knowledge_core_id"])
        self.assertEqual(["conversation", "document"], result["enabled_capabilities"])

    async def test_active_agent_requires_a_knowledge_core(self) -> None:
        service = KnowledgeRetrievalAgentService(
            uow_factory=lambda: _UnitOfWork(_Repository())
        )

        with self.assertRaises(AgentApplicationError) as raised:
            await service.create(CreateAgentCommand(
                domain_id=41,
                display_name="无知识库",
                status="ACTIVE",
                actor_id="user-41",
            ))

        self.assertEqual("AGENT_KNOWLEDGE_CORE_REQUIRED", raised.exception.code)

    async def test_x_search_model_must_be_active_grok_llm(self) -> None:
        model_id = uuid7()
        row = {
            "model_id": str(model_id),
            "provider_model_name": "xai.command-r-plus",
            "status": "ACTIVE",
            "category": 1,
        }
        service = KnowledgeRetrievalAgentService(
            uow_factory=lambda: _UnitOfWork(_Repository()),
            catalog_client=_Catalog(row),
        )
        with self.assertRaises(AgentApplicationError) as raised:
            await service.create(CreateAgentCommand(
                domain_id=41,
                display_name="错误搜索模型",
                models={"x_search_llm": model_id},
                actor_id="user-41",
            ))
        self.assertEqual("X_SEARCH_GROK_REQUIRED", raised.exception.code)

        row["provider_model_name"] = "xai.grok-4.6"
        row["category"] = 3
        with self.assertRaises(AgentApplicationError) as raised:
            await service.create(CreateAgentCommand(
                domain_id=41,
                display_name="错误模型类别",
                models={"x_search_llm": model_id},
                actor_id="user-41",
            ))
        self.assertEqual("X_SEARCH_MODEL_UNAVAILABLE", raised.exception.code)

    async def test_grok_binding_enables_x_search_without_leaking_into_chat_runtime(self) -> None:
        repository = _Repository()
        model_id = uuid7()
        service = KnowledgeRetrievalAgentService(
            uow_factory=lambda: _UnitOfWork(repository),
            catalog_client=_Catalog({
                "model_id": str(model_id),
                "provider_model_name": "xai.grok-4.6",
                "status": "ACTIVE",
                "category": 1,
            }),
        )
        models = {**_active_models(model_id), "x_search_llm": model_id}
        created = await service.create(CreateAgentCommand(
            domain_id=41,
            display_name="外部知识助手",
            knowledge_core_id=uuid7(),
            models=models,
            status="ACTIVE",
            actor_id="user-41",
        ))

        self.assertIn("x_search", created["enabled_capabilities"])
        spec = AgentExecutionSpec.model_validate(await service.execution_spec(
            domain_id=41, agent_id=repository.agent.agent_id
        ))
        self.assertEqual("knowledge_retrieval", spec.owner_app_id)
        self.assertNotIn("x_search", spec.enabled_capabilities)
        self.assertEqual(model_id, UUID(str(spec.models["x_search_llm"])))


if __name__ == "__main__":
    unittest.main()
