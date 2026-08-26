"""AIOps 私有 Agent 创建顺序测试。"""

import unittest
from types import SimpleNamespace
from aiops_agent.application.agents import (
    AIOpsAgentService,
    CreateAIOpsAgentCommand,
)
from pydantic import ValidationError
from platform_core.identity import uuid7


class _AgentRepository:
    def __init__(self):
        self.agents = {}
        self.versions = {}
        self.current_version_at_agent_insert = "NOT_CAPTURED"
        self.version_sources = {}

    async def resource_states(self, **kwargs):
        del kwargs
        return {"policy": "ACTIVE"}

    async def add_agent(self, row):
        self.current_version_at_agent_insert = row.current_version_id
        row.row_version = 1
        self.agents[row.agent_id] = row

    async def add_version(self, row):
        agent = self.agents[row.agent_id]
        if agent.current_version_id is not None:
            raise AssertionError("版本插入前不得写入 CURRENT_VERSION_ID")
        self.versions[row.agent_version_id] = row

    async def add_version_sources(self, *, version_id, source_ids):
        self.version_sources[version_id] = list(source_ids)

    async def version_source_ids(self, *, agent_version_id):
        return self.version_sources.get(agent_version_id, [])

    async def get(self, *, domain_id, agent_id, lock=False):
        del lock
        row = self.agents.get(agent_id)
        return row if row is not None and int(row.domain_id) == domain_id else None

    async def version(self, *, agent_id, agent_version_id):
        row = self.versions.get(agent_version_id)
        return row if row is not None and row.agent_id == agent_id else None


class _UnitOfWork:
    def __init__(self, repository, source_id):
        self.agents = repository
        self.diagnostic_sources = SimpleNamespace(
            get_scoped=self._get_source
        )
        self.targets = SimpleNamespace(get_scoped=self._get_target)
        self.policies = _PolicyRepository()
        self.source_id = source_id
        self.commit_count = 0

    async def _get_source(self, *, diagnostic_source_id, domain_id):
        del domain_id
        if diagnostic_source_id != self.source_id:
            return None
        return SimpleNamespace(
            status="ENABLED", connectivity_status="CONNECTED"
        )

    async def _get_target(self, **kwargs):
        del kwargs
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def commit(self):
        self.commit_count += 1


class _PolicyRepository:
    def __init__(self):
        self.rows = {}

    async def add(self, row):
        self.rows[row.policy_id] = row

    async def get_scoped(self, *, policy_id, domain_id):
        row = self.rows.get(policy_id)
        return row if row is not None and int(row.domain_id) == domain_id else None


class AIOpsAgentCreationTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_inserts_version_before_current_version_pointer(self):
        repository = _AgentRepository()
        source_id = uuid7()
        unit_of_work = _UnitOfWork(repository, source_id)
        service = AIOpsAgentService(uow_factory=lambda: unit_of_work)

        result = await service.create(
            CreateAIOpsAgentCommand(
                domain_id=100,
                display_name="数据库诊断助手",
                diagnostic_source_ids=(source_id,),
                actor_id="kbotui_dev",
            )
        )

        self.assertIsNone(repository.current_version_at_agent_insert)
        self.assertEqual(1, unit_of_work.commit_count)
        self.assertEqual(
            result["agent_version_id"],
            str(repository.agents[next(iter(repository.agents))].current_version_id),
        )

    async def test_active_agent_exposes_selected_sources_without_target(self):
        source_id = uuid7()
        unit_of_work = _UnitOfWork(_AgentRepository(), source_id)
        service = AIOpsAgentService(
            uow_factory=lambda: unit_of_work
        )

        result = await service.create(
            CreateAIOpsAgentCommand(
                domain_id=100,
                display_name="数据库诊断助手",
                diagnostic_source_ids=(source_id,),
                status="ACTIVE",
                actor_id="kbotui_dev",
            )
        )

        self.assertEqual("ACTIVE", result["status"])
        self.assertIsNone(result["target_id"])
        self.assertEqual([str(source_id)], result["diagnostic_source_ids"])
        self.assertFalse(result["allow_change_execution"])
        policy = next(iter(unit_of_work.policies.rows.values()))
        self.assertEqual(f"agent.{result['agent_id']}", policy.policy_key)
        self.assertEqual("ACTIVE", policy.status)
        self.assertFalse(policy.rules_json["readonly_database_enabled"])
        self.assertTrue(policy.rules_json["auto_alert_enabled"])
        self.assertEqual(900, policy.rules_json["alert_cooldown_seconds"])

    async def test_change_execution_requires_selected_target(self):
        with self.assertRaises(ValidationError):
            CreateAIOpsAgentCommand(
                domain_id=100,
                display_name="数据库诊断助手",
                diagnostic_source_ids=(uuid7(),),
                allow_change_execution=True,
                actor_id="kbotui_dev",
            )


if __name__ == "__main__":
    unittest.main()
