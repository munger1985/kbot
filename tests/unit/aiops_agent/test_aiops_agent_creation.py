"""AIOps Agent 的业务系统授权与资源动作策略测试。"""

import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from aiops_agent.actions import ActionRegistry
from aiops_agent.application.agents import (
    AIOpsAgentError,
    AIOpsAgentService,
    CreateAIOpsAgentCommand,
    ResourceControlledActionExecution,
    UpdateAIOpsAgentCommand,
)
from platform_core.contracts.aiops import AIOpsCapability, AgentKnowledgePolicy
from platform_core.identity import uuid7


class _AgentRepository:
    def __init__(self):
        self.agents = {}
        self.versions = {}
        self.version_systems = {}
        self.version_knowledge_bases = {}
        self.resource_policies = {}

    async def add_agent(self, row):
        row.row_version = 1
        self.agents[row.agent_id] = row

    async def add_version(self, row):
        if self.agents[row.agent_id].current_version_id == row.agent_version_id:
            raise AssertionError("版本插入前不得指向待插入版本")
        self.versions[row.agent_version_id] = row

    async def next_version_no(self, *, agent_id):
        return 1 + max(
            (int(row.version_no) for row in self.versions.values() if row.agent_id == agent_id),
            default=0,
        )

    async def add_version_business_systems(
        self, *, version_id, business_system_ids
    ):
        self.version_systems[version_id] = list(business_system_ids)

    async def add_version_knowledge_bases(
        self, *, version_id, knowledge_base_ids
    ):
        self.version_knowledge_bases[version_id] = list(knowledge_base_ids)

    async def add_version_resource_policies(
        self, *, version_id, controlled_action_policies
    ):
        self.resource_policies[version_id] = dict(controlled_action_policies)

    async def get(self, *, domain_id, agent_id, lock=False):
        del lock
        row = self.agents.get(agent_id)
        return row if row is not None and int(row.domain_id) == domain_id else None

    async def version(self, *, agent_id, agent_version_id):
        row = self.versions.get(agent_version_id)
        return row if row is not None and row.agent_id == agent_id else None

    async def version_business_system_ids(self, *, agent_version_id):
        return self.version_systems.get(agent_version_id, [])

    async def version_resource_policies(self, *, agent_version_id):
        return self.resource_policies.get(agent_version_id, {})

    async def version_target_ids(self, *, agent_version_id):
        del agent_version_id
        return []


class _PolicyRepository:
    def __init__(self):
        self.rows = {}

    async def add(self, row):
        self.rows[row.policy_id] = row

    async def get_scoped(self, *, policy_id, domain_id):
        row = self.rows.get(policy_id)
        return row if row is not None and int(row.domain_id) == domain_id else None


class _BusinessSystemRepository:
    def __init__(self, system, rows=()):
        self.system = system
        self.rows = tuple(rows)

    async def get_scoped(self, *, business_system_id, domain_id):
        if (
            business_system_id == self.system.business_system_id
            and domain_id == self.system.domain_id
        ):
            return self.system
        return None

    async def list_system_resource_rows(self, business_system_id):
        return (
            list(self.rows)
            if business_system_id == self.system.business_system_id
            else []
        )

    async def get_resource_for_target(self, *, target_id, domain_id):
        for _link, resource, database, _target in self.rows:
            if database is not None and database.target_id == target_id:
                return resource if resource.domain_id == domain_id else None
        return None


class _UnitOfWork:
    def __init__(self, *, rows=()):
        self.agents = _AgentRepository()
        self.policies = _PolicyRepository()
        self.system = SimpleNamespace(
            business_system_id=uuid7(),
            domain_id=100,
            system_code="MES",
            display_name="MES 生产系统",
            environment="PROD",
            status="ENABLED",
        )
        self.business_systems = _BusinessSystemRepository(self.system, rows)
        self.knowledge_base_id = uuid7()
        self.operations_knowledge = SimpleNamespace(
            get_knowledge_base=self._get_knowledge_base,
            get_tag=self._get_tag,
        )
        self.targets = SimpleNamespace(get_scoped=self._get_target)
        self.rows = tuple(rows)
        self.commit_count = 0

    async def _get_target(self, *, target_id, domain_id):
        for _link, _resource, database, target in self.rows:
            if database is not None and database.target_id == target_id:
                return target if target.domain_id == domain_id else None
        return None

    async def _get_knowledge_base(self, *, domain_id, knowledge_base_id):
        if domain_id == 100 and knowledge_base_id == self.knowledge_base_id:
            return SimpleNamespace(status="ACTIVE")
        return None

    async def _get_tag(self, **kwargs):
        del kwargs
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def commit(self):
        self.commit_count += 1


def _database_row():
    target_id = uuid7()
    resource = SimpleNamespace(
        resource_id=uuid7(),
        domain_id=100,
        display_name="MES 主库",
        resource_type_key="DATABASE",
        lifecycle_status="ACTIVE",
    )
    link = SimpleNamespace(system_resource_id=uuid7(), status="ACTIVE")
    database = SimpleNamespace(resource_id=resource.resource_id, target_id=target_id)
    target = SimpleNamespace(
        target_id=target_id,
        domain_id=100,
        display_name="MES 主库 Target",
        db_type="ORACLE",
        status="ENABLED",
        connectivity_status="CONNECTED",
        readonly_connection_enabled=True,
        controlled_change_enabled=False,
        execution_credential_id=None,
        importance_level=5,
        version_code="19c",
        environment="PROD",
        capabilities_json={},
    )
    return link, resource, database, target


class AIOpsAgentCreationTest(unittest.IsolatedAsyncioTestCase):
    def _change_context(self):
        row = _database_row()
        target = row[3]
        target.controlled_change_enabled = True
        target.execution_credential_id = uuid7()
        target.environment = "DEV"
        target.capabilities_json = {"session_management": True}
        uow = _UnitOfWork(rows=(row,))
        service = AIOpsAgentService(
            uow_factory=lambda: uow, action_registry=ActionRegistry.load()
        )
        values = dict(
            domain_id=100,
            display_name="数据库诊断助手",
            allowed_business_system_ids=(uow.system.business_system_id,),
            capabilities=(AIOpsCapability.ACTIVE_DIAGNOSIS, AIOpsCapability.CONTROLLED_CHANGE),
            knowledge_policy=AgentKnowledgePolicy(enabled=False),
            models={"planner_llm": uuid7(), "diagnosis_llm": uuid7(), "composer_llm": uuid7()},
            status="ACTIVE",
            actor_id="tester",
        )
        return row, uow, service, values

    async def test_active_change_capability_without_authorization_allows_diagnostic_updates(self):
        for disabled_policy in (False, True):
            with self.subTest(disabled_policy=disabled_policy):
                row, _uow, service, values = self._change_context()
                row[3].controlled_change_enabled = False
                row[3].execution_credential_id = None
                if disabled_policy:
                    values["controlled_action_execution"] = (
                        ResourceControlledActionExecution(resource_id=row[1].resource_id, enabled=False),
                    )
                created = await service.create(CreateAIOpsAgentCommand(**values))
                updated = await service.update(
                    UpdateAIOpsAgentCommand(
                        domain_id=100,
                        agent_id=created["agent_id"],
                        expected_row_version=created["row_version"],
                        instruction="优先读取执行计划后分析",
                        capabilities=values["capabilities"],
                        models=values["models"],
                        actor_id="tester",
                    )
                )
                self.assertEqual("ACTIVE", updated["status"])
                self.assertEqual(2, updated["version_no"])
                self.assertIn("CONTROLLED_CHANGE", updated["capabilities"])
                self.assertEqual(created["controlled_action_execution"], updated["controlled_action_execution"])

    async def test_action_policy_patch_preserves_or_explicitly_revokes_authorization(self):
        row, uow, service, values = self._change_context()
        values["controlled_action_execution"] = (
            ResourceControlledActionExecution(
                resource_id=row[1].resource_id, enabled=True,
                allowed_action_ids=("db.session.terminate",),
            ),
        )
        created = await service.create(CreateAIOpsAgentCommand(**values))
        updated = await service.update(
            UpdateAIOpsAgentCommand(
                domain_id=100, agent_id=created["agent_id"],
                expected_row_version=created["row_version"],
                instruction="仅执行审批动作", actor_id="tester",
            )
        )
        self.assertEqual(created["controlled_action_execution"], updated["controlled_action_execution"])
        null_update = await service.update(
            UpdateAIOpsAgentCommand(
                domain_id=100, agent_id=created["agent_id"],
                expected_row_version=updated["row_version"],
                instruction="空值不撤销动作授权",
                controlled_action_execution=None, actor_id="tester",
            )
        )
        self.assertEqual(created["controlled_action_execution"], null_update["controlled_action_execution"])
        revoked = await service.update(
            UpdateAIOpsAgentCommand(
                domain_id=100, agent_id=created["agent_id"],
                expected_row_version=null_update["row_version"],
                controlled_action_execution=(), actor_id="tester",
            )
        )
        self.assertEqual([], revoked["controlled_action_execution"])
        self.assertEqual("ACTIVE", revoked["status"])
        self.assertIn("CONTROLLED_CHANGE", revoked["capabilities"])
        original_policies = uow.agents.resource_policies[next(iter(uow.agents.versions))]
        self.assertTrue(original_policies[row[1].resource_id]["enabled"])

    async def test_enabled_action_policy_still_requires_execution_authorization(self):
        for invalid, code in (
            ("capability", "AIOPS_AGENT_CHANGE_CAPABILITY_REQUIRED"),
            ("target", "AIOPS_AGENT_CHANGE_TARGET_REQUIRED"),
            ("credential", "AIOPS_AGENT_CHANGE_TARGET_REQUIRED"),
            ("action", "AIOPS_AGENT_ACTION_NOT_COMPATIBLE"),
            ("resource", "AIOPS_AGENT_ACTION_RESOURCE_NOT_ALLOWED"),
            ("scope", "AIOPS_AGENT_ACTION_SCOPE_REQUIRED"),
        ):
            with self.subTest(invalid=invalid):
                row, uow, service, values = self._change_context()
                if invalid == "capability":
                    values["capabilities"] = (AIOpsCapability.ACTIVE_DIAGNOSIS,)
                if invalid == "target":
                    row[3].controlled_change_enabled = False
                if invalid == "credential":
                    row[3].execution_credential_id = None
                if invalid == "scope":
                    row[3].capabilities_json = {"dba_catalog_views": True}
                action_id = {
                    "action": "unknown.action", "scope": "db.statistics.gather",
                }.get(invalid, "db.session.terminate")
                values["controlled_action_execution"] = (
                    ResourceControlledActionExecution(
                        resource_id=uuid7() if invalid == "resource" else row[1].resource_id,
                        enabled=True,
                        allowed_action_ids=(action_id,),
                    ),
                )
                with self.assertRaises(AIOpsAgentError) as raised:
                    await service.create(CreateAIOpsAgentCommand(**values))
                self.assertEqual(code, raised.exception.code)
                self.assertEqual(0, uow.commit_count)

    async def test_create_freezes_business_system_authorization(self):
        uow = _UnitOfWork()
        service = AIOpsAgentService(uow_factory=lambda: uow)

        result = await service.create(
            CreateAIOpsAgentCommand(
                domain_id=100,
                display_name="统一运维助手",
                allowed_business_system_ids=(uow.system.business_system_id,),
                capabilities=(AIOpsCapability.KNOWLEDGE_QA,),
                knowledge_policy=AgentKnowledgePolicy(
                    knowledge_base_ids=(uow.knowledge_base_id,)
                ),
                actor_id="tester",
            )
        )

        self.assertEqual(
            [str(uow.system.business_system_id)],
            result["allowed_business_system_ids"],
        )
        self.assertEqual(["KNOWLEDGE_QA", "MONITORING_QUERY"], result["capabilities"])
        self.assertEqual(
            [uow.knowledge_base_id],
            next(iter(uow.agents.version_knowledge_bases.values())),
        )
        self.assertNotIn("target_ids", result)
        self.assertEqual(1, uow.commit_count)

    async def test_active_agent_requires_planner_and_composer(self):
        uow = _UnitOfWork()
        service = AIOpsAgentService(uow_factory=lambda: uow)
        with self.assertRaises(AIOpsAgentError) as raised:
            await service.create(
                CreateAIOpsAgentCommand(
                    domain_id=100,
                    display_name="统一运维助手",
                    allowed_business_system_ids=(uow.system.business_system_id,),
                    capabilities=(AIOpsCapability.KNOWLEDGE_QA,),
                    knowledge_policy=AgentKnowledgePolicy(
                        knowledge_base_ids=(uow.knowledge_base_id,)
                    ),
                    status="ACTIVE",
                    actor_id="tester",
                )
            )
        self.assertEqual(
            "AIOPS_AGENT_PLANNER_MODEL_REQUIRED", raised.exception.code
        )

    async def test_diagnosis_model_only_required_for_active_diagnosis(self):
        row = _database_row()
        uow = _UnitOfWork(rows=(row,))
        service = AIOpsAgentService(uow_factory=lambda: uow)
        with self.assertRaises(AIOpsAgentError) as raised:
            await service.create(
                CreateAIOpsAgentCommand(
                    domain_id=100,
                    display_name="诊断助手",
                    allowed_business_system_ids=(uow.system.business_system_id,),
                    capabilities=(AIOpsCapability.ACTIVE_DIAGNOSIS,),
                    knowledge_policy=AgentKnowledgePolicy(enabled=False),
                    models={"planner_llm": uuid7(), "composer_llm": uuid7()},
                    status="ACTIVE",
                    actor_id="tester",
                )
            )
        self.assertEqual(
            "AIOPS_AGENT_DIAGNOSIS_MODEL_REQUIRED", raised.exception.code
        )

    async def test_action_policy_resource_must_belong_to_allowed_system(self):
        uow = _UnitOfWork()
        service = AIOpsAgentService(uow_factory=lambda: uow)
        with self.assertRaises(AIOpsAgentError) as raised:
            await service.create(
                CreateAIOpsAgentCommand(
                    domain_id=100,
                    display_name="变更助手",
                    allowed_business_system_ids=(uow.system.business_system_id,),
                    capabilities=(AIOpsCapability.CONTROLLED_CHANGE,),
                    knowledge_policy=AgentKnowledgePolicy(enabled=False),
                    controlled_action_execution=(
                        ResourceControlledActionExecution(
                            resource_id=uuid7(),
                            enabled=False,
                        ),
                    ),
                    actor_id="tester",
                )
            )
        self.assertEqual(
            "AIOPS_AGENT_ACTION_RESOURCE_NOT_ALLOWED", raised.exception.code
        )

    def test_business_system_authorization_cannot_be_empty(self):
        with self.assertRaises(ValidationError):
            CreateAIOpsAgentCommand(
                domain_id=100,
                display_name="无范围助手",
                allowed_business_system_ids=(),
                capabilities=(AIOpsCapability.KNOWLEDGE_QA,),
                actor_id="tester",
            )


if __name__ == "__main__":
    unittest.main()
