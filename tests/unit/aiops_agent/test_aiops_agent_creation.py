"""AIOps 私有 Agent 创建顺序测试。"""

import unittest

from aiops_agent.application.agents import (
    AIOpsAgentService,
    CreateAIOpsAgentCommand,
)
from platform_core.identity import uuid7


class _AgentRepository:
    def __init__(self):
        self.agents = {}
        self.versions = {}
        self.current_version_at_agent_insert = "NOT_CAPTURED"

    async def resource_states(self, **kwargs):
        del kwargs
        return {
            "monitor": "ACTIVE",
            "policy": "ACTIVE",
            "target": None,
            "plan": None,
        }

    async def add_agent(self, row):
        self.current_version_at_agent_insert = row.current_version_id
        row.row_version = 1
        self.agents[row.agent_id] = row

    async def add_version(self, row):
        agent = self.agents[row.agent_id]
        if agent.current_version_id is not None:
            raise AssertionError("版本插入前不得写入 CURRENT_VERSION_ID")
        self.versions[row.agent_version_id] = row

    async def get(self, *, domain_id, agent_id, lock=False):
        del lock
        row = self.agents.get(agent_id)
        return row if row is not None and int(row.domain_id) == domain_id else None

    async def version(self, *, agent_id, agent_version_id):
        row = self.versions.get(agent_version_id)
        return row if row is not None and row.agent_id == agent_id else None


class _UnitOfWork:
    def __init__(self, repository):
        self.agents = repository
        self.commit_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def commit(self):
        self.commit_count += 1


class AIOpsAgentCreationTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_inserts_version_before_current_version_pointer(self):
        repository = _AgentRepository()
        unit_of_work = _UnitOfWork(repository)
        service = AIOpsAgentService(uow_factory=lambda: unit_of_work)

        result = await service.create(
            CreateAIOpsAgentCommand(
                domain_id=100,
                display_name="数据库诊断助手",
                monitor_source_id=uuid7(),
                policy_id=uuid7(),
                actor_id="kbotui_dev",
            )
        )

        self.assertIsNone(repository.current_version_at_agent_insert)
        self.assertEqual(1, unit_of_work.commit_count)
        self.assertEqual(
            result["agent_version_id"],
            str(repository.agents[next(iter(repository.agents))].current_version_id),
        )


if __name__ == "__main__":
    unittest.main()
