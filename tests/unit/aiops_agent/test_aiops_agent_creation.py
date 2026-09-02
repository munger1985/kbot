"""AIOps 私有 Agent 创建顺序测试。"""

import unittest
from types import SimpleNamespace
from aiops_agent.application.agents import (
    AIOpsAgentError,
    AIOpsAgentService,
    CreateAIOpsAgentCommand,
    TargetControlledActionExecution,
    UpdateAIOpsAgentCommand,
)
from pydantic import ValidationError
from platform_core.identity import uuid7


class _AgentRepository:
    def __init__(self):
        self.agents = {}
        self.versions = {}
        self.current_version_at_agent_insert = "NOT_CAPTURED"
        self.version_sources = {}
        self.version_targets = {}
        self.controlled_action_policies = {}

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

    async def add_version_targets(
        self, *, version_id, target_ids, controlled_action_policies
    ):
        self.version_targets[version_id] = list(target_ids)
        self.controlled_action_policies[version_id] = dict(
            controlled_action_policies
        )

    async def version_target_ids(self, *, agent_version_id):
        return self.version_targets.get(agent_version_id, [])

    async def version_target_policies(self, *, agent_version_id):
        return self.controlled_action_policies.get(agent_version_id, {})

    async def get(self, *, domain_id, agent_id, lock=False):
        del lock
        row = self.agents.get(agent_id)
        return row if row is not None and int(row.domain_id) == domain_id else None

    async def version(self, *, agent_id, agent_version_id):
        row = self.versions.get(agent_version_id)
        return row if row is not None and row.agent_id == agent_id else None


class _UnitOfWork:
    def __init__(self, repository, source_id, target=None):
        self.agents = repository
        self.diagnostic_sources = SimpleNamespace(
            get_scoped=self._get_source
        )
        self.targets = SimpleNamespace(
            get_scoped=self._get_target,
            target_ids_shared_by_sources=self._target_ids_shared_by_sources,
        )
        self.policies = _PolicyRepository()
        self.source_id = source_id
        self.target = target or SimpleNamespace(
            target_id=uuid7(), display_name="逻辑测试库", db_type="ORACLE",
            status="ENABLED", connectivity_status="UNKNOWN",
            readonly_connection_enabled=False,
            controlled_change_enabled=False, execution_credential_id=None,
            version_code=None, capabilities_json={},
        )
        self.targets.list_source_bindings = self._list_source_bindings
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
        return self.target

    async def _target_ids_shared_by_sources(self, **kwargs):
        del kwargs
        return [self.target.target_id] if self.target is not None else []

    async def _list_source_bindings(self, **kwargs):
        del kwargs
        return [SimpleNamespace(diagnostic_source_id=self.source_id)]

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
    async def test_controlled_action_scope_registers_sensitive_targets(self):
        policy = TargetControlledActionExecution(
            target_id=uuid7(),
            enabled=True,
            allowed_action_ids=("db.parameter.set", "db.user.privilege.grant"),
            object_scopes={
                "schemas": ("APP",),
                "dynamic_parameters": (
                    {"name": "cursor_sharing", "allowed_values": ("EXACT", "FORCE")},
                ),
                "resource_manager_plans": ("APP_PLAN",),
                "privilege_grantees": ("APPUSER",),
                "system_privileges": ("CREATE SESSION",),
                "object_privileges": ("SELECT",),
            },
        )

        self.assertEqual(
            ("FORCE",),
            (policy.object_scopes.dynamic_parameters[0].allowed_values[1],),
        )
        self.assertEqual(("APPUSER",), policy.object_scopes.privilege_grantees)
        with self.assertRaises(ValidationError):
            TargetControlledActionExecution(
                target_id=uuid7(),
                enabled=True,
                allowed_action_ids=("db.user.privilege.grant",),
                object_scopes={"system_privileges": ("DROP ANY TABLE",)},
            )

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
                target_ids=(unit_of_work.target.target_id,),
                actor_id="kbotui_dev",
            )
        )

        self.assertIsNone(repository.current_version_at_agent_insert)
        self.assertEqual(1, unit_of_work.commit_count)
        self.assertEqual(
            result["agent_version_id"],
            str(repository.agents[next(iter(repository.agents))].current_version_id),
        )
        self.assertEqual({}, result["models"])

    async def test_active_agent_exposes_selected_sources_and_logical_target(self):
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
                target_ids=(unit_of_work.target.target_id,),
                models={
                    "planner_llm": uuid7(),
                    "diagnosis_llm": uuid7(),
                },
                status="ACTIVE",
                actor_id="kbotui_dev",
            )
        )

        self.assertEqual("ACTIVE", result["status"])
        self.assertEqual([str(unit_of_work.target.target_id)], result["target_ids"])
        self.assertEqual([str(source_id)], result["diagnostic_source_ids"])
        self.assertEqual([], result["controlled_action_execution"])
        self.assertFalse(
            result["target_candidates"][0]["readonly_connection_enabled"]
        )
        policy = next(iter(unit_of_work.policies.rows.values()))
        self.assertEqual(f"agent.{result['agent_id']}", policy.policy_key)
        self.assertEqual("ACTIVE", policy.status)
        self.assertTrue(policy.rules_json["readonly_database_enabled"])
        self.assertTrue(policy.rules_json["auto_alert_enabled"])
        self.assertEqual(900, policy.rules_json["alert_cooldown_seconds"])

    async def test_active_agent_requires_diagnosis_model(self):
        source_id = uuid7()
        unit_of_work = _UnitOfWork(_AgentRepository(), source_id)
        service = AIOpsAgentService(uow_factory=lambda: unit_of_work)

        with self.assertRaises(AIOpsAgentError) as raised:
            await service.create(
                CreateAIOpsAgentCommand(
                    domain_id=100,
                    display_name="缺少模型的诊断助手",
                    diagnostic_source_ids=(source_id,),
                    target_ids=(unit_of_work.target.target_id,),
                    status="ACTIVE",
                    actor_id="kbotui_dev",
                )
            )

        self.assertEqual(
            "AIOPS_AGENT_DIAGNOSIS_MODEL_REQUIRED", raised.exception.code
        )
        self.assertEqual(0, unit_of_work.commit_count)

    async def test_draft_without_model_cannot_be_enabled(self):
        source_id = uuid7()
        unit_of_work = _UnitOfWork(_AgentRepository(), source_id)
        service = AIOpsAgentService(uow_factory=lambda: unit_of_work)
        draft = await service.create(
            CreateAIOpsAgentCommand(
                domain_id=100,
                display_name="待配置模型的诊断助手",
                diagnostic_source_ids=(source_id,),
                target_ids=(unit_of_work.target.target_id,),
                actor_id="kbotui_dev",
            )
        )

        with self.assertRaises(AIOpsAgentError) as raised:
            await service.update(
                UpdateAIOpsAgentCommand(
                    domain_id=100,
                    agent_id=draft["agent_id"],
                    expected_row_version=draft["row_version"],
                    status="ACTIVE",
                    actor_id="kbotui_dev",
                )
            )

        self.assertEqual(
            "AIOPS_AGENT_DIAGNOSIS_MODEL_REQUIRED", raised.exception.code
        )

    async def test_active_agent_requires_planner_model(self):
        source_id = uuid7()
        unit_of_work = _UnitOfWork(_AgentRepository(), source_id)
        service = AIOpsAgentService(uow_factory=lambda: unit_of_work)

        with self.assertRaises(AIOpsAgentError) as raised:
            await service.create(
                CreateAIOpsAgentCommand(
                    domain_id=100,
                    display_name="缺少规划模型的诊断助手",
                    diagnostic_source_ids=(source_id,),
                    target_ids=(unit_of_work.target.target_id,),
                    models={"diagnosis_llm": uuid7()},
                    status="ACTIVE",
                    actor_id="kbotui_dev",
                )
            )

        self.assertEqual(
            "AIOPS_AGENT_PLANNER_MODEL_REQUIRED", raised.exception.code
        )

    async def test_agent_requires_selected_logical_target(self):
        with self.assertRaises(ValidationError):
            CreateAIOpsAgentCommand(
                domain_id=100,
                display_name="数据库诊断助手",
                diagnostic_source_ids=(uuid7(),),
                actor_id="kbotui_dev",
            )

    async def test_change_permission_requires_change_capable_target(self):
        source_id = uuid7()
        target_id = uuid7()
        target = SimpleNamespace(
            target_id=target_id,
            display_name="测试数据库",
            db_type="ORACLE",
            status="ENABLED",
            connectivity_status="CONNECTED",
            controlled_change_enabled=False,
            execution_credential_id=None,
            version_code=None,
            capabilities_json={},
        )
        unit_of_work = _UnitOfWork(
            _AgentRepository(), source_id, target=target
        )
        service = AIOpsAgentService(uow_factory=lambda: unit_of_work)

        with self.assertRaises(AIOpsAgentError) as raised:
            await service.create(
                CreateAIOpsAgentCommand(
                    domain_id=100,
                    display_name="数据库变更助手",
                    diagnostic_source_ids=(source_id,),
                    target_ids=(target_id,),
                    controlled_action_execution=(
                        {
                            "target_id": target_id,
                            "enabled": True,
                            "allowed_action_ids": (
                                "db.session.terminate",
                            ),
                        },
                    ),
                    actor_id="kbotui_dev",
                )
            )
        self.assertEqual("AIOPS_AGENT_CHANGE_TARGET_REQUIRED", raised.exception.code)


if __name__ == "__main__":
    unittest.main()
