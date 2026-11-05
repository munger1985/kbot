"""Agent 范围变更只收缩继承授权，不转授权或绕过显式权限校验。"""

import unittest
from copy import deepcopy
from types import SimpleNamespace
from uuid import UUID

from aiops_agent.actions import ActionRegistry
from aiops_agent.application.agents import (
    AIOpsAgentError, AIOpsAgentService, CreateAIOpsAgentCommand,
    ResourceControlledActionExecution, UpdateAIOpsAgentCommand,
)
from platform_core.contracts.aiops import AIOpsCapability, AgentKnowledgePolicy
from platform_core.identity import uuid7

from tests.unit.aiops_agent.test_aiops_agent_creation import _UnitOfWork


def _context():
    uow = _UnitOfWork()
    return uow, {"domain_id": 100, "actor_id": "tester"}, uow.agents.resource_policies


class AgentActionAuthorizationScopeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.uow, self.identity, self.policies = _context()
        self.old, self.new, self.shared = [self._row() for _ in range(3)]
        self.old_system, self.new_system = uuid7(), uuid7()
        self.rows = {
            self.old_system: [self.old, self.shared],
            self.new_system: [self.new, self.shared],
        }
        self.systems = {
            key: SimpleNamespace(
                business_system_id=key, system_code=str(key), display_name="测试业务系统",
                status="ENABLED", environment="DEV",
            ) for key in self.rows
        }
        self.uow.business_systems.get_scoped = self._get_system
        self.uow.business_systems.list_system_resource_rows = self._get_rows
        self.uow.targets.get_scoped = self._get_target
        self.service = AIOpsAgentService(
            uow_factory=lambda: self.uow, action_registry=ActionRegistry.load(),
        )

    @staticmethod
    def _row():
        resource_id, target_id = uuid7(), uuid7()
        return (
            SimpleNamespace(status="ACTIVE"),
            SimpleNamespace(resource_id=resource_id, lifecycle_status="ACTIVE"),
            SimpleNamespace(resource_id=resource_id, target_id=target_id),
            SimpleNamespace(
                target_id=target_id, display_name="测试数据库", db_type="ORACLE",
                status="ENABLED", connectivity_status="CONNECTED", importance_level=3,
                readonly_connection_enabled=True, controlled_change_enabled=True,
                execution_credential_id=uuid7(), environment="DEV", version_code="19c",
                capabilities_json={"session_management": True},
            ),
        )

    async def _get_system(self, *, business_system_id, **_scope):
        return self.systems.get(business_system_id)

    async def _get_rows(self, business_system_id):
        return self.rows.get(business_system_id, [])

    async def _get_target(self, *, target_id, **_scope):
        return next((row[3] for row in (self.old, self.new, self.shared) if row[3].target_id == target_id), None)

    def _policy(self, row, *, enabled=True, action="db.session.terminate"):
        return ResourceControlledActionExecution(
            resource_id=row[1].resource_id, enabled=enabled,
            allowed_action_ids=(action,) if enabled else (),
        )

    async def _create(self, **changes):
        values = dict(
            **self.identity, display_name="范围测试 Agent", status="ACTIVE",
            allowed_business_system_ids=(self.old_system,),
            capabilities=(AIOpsCapability.ACTIVE_DIAGNOSIS, AIOpsCapability.CONTROLLED_CHANGE),
            knowledge_policy=AgentKnowledgePolicy(enabled=False),
            models={"planner_llm": uuid7(), "diagnosis_llm": uuid7(), "composer_llm": uuid7()},
            controlled_action_execution=(self._policy(self.old),),
        )
        values.update(changes)
        return await self.service.create(CreateAIOpsAgentCommand(**values))

    async def _update(self, created, **changes):
        return await self.service.update(UpdateAIOpsAgentCommand(
            **self.identity, agent_id=created["agent_id"],
            expected_row_version=created["row_version"], **changes,
        ))

    async def test_switch_system_revokes_old_authorization_without_granting_new(self):
        for explicit_null in (False, True):
            with self.subTest(explicit_null=explicit_null):
                created = await self._create()
                original = deepcopy(self.policies[UUID(created["agent_version_id"])])
                updated = await self._update(
                    created, allowed_business_system_ids=(self.new_system,),
                    **({"controlled_action_execution": None} if explicit_null else {}),
                )
                self.assertEqual([], updated["controlled_action_execution"])
                self.assertEqual([str(self.new_system)], updated["allowed_business_system_ids"])
                self.assertEqual(2, updated["version_no"])
                self.assertEqual(original, self.policies[UUID(created["agent_version_id"])])
                # 切回旧系统也不自动恢复已经从当前版本撤销的权限。
                restored = await self._update(updated, allowed_business_system_ids=(self.old_system,))
                self.assertEqual([], restored["controlled_action_execution"])

    async def test_shrinking_systems_preserves_shared_resource_authorization(self):
        created = await self._create(
            allowed_business_system_ids=(self.old_system, self.new_system),
            controlled_action_execution=(self._policy(self.old), self._policy(self.new), self._policy(self.shared)),
        )
        original = deepcopy(self.policies[UUID(created["agent_version_id"])])
        updated = await self._update(created, allowed_business_system_ids=(self.new_system,))
        self.assertEqual(
            {str(self.new[1].resource_id), str(self.shared[1].resource_id)},
            {item["resource_id"] for item in updated["controlled_action_execution"]},
        )
        for resource_id, policy in self.policies[UUID(updated["agent_version_id"])].items():
            self.assertEqual(original[resource_id], policy)
        self.assertEqual(original, self.policies[UUID(created["agent_version_id"])])

    async def test_same_scope_preserves_policy_and_explicit_empty_revokes_all(self):
        created = await self._create()
        for changes in ({"instruction": "分析等待事件"}, {"controlled_action_execution": None, "instruction": "分析 SQL"}):
            updated = await self._update(created, **changes)
            self.assertEqual(created["controlled_action_execution"], updated["controlled_action_execution"])
            created = updated
        revoked = await self._update(created, controlled_action_execution=())
        self.assertEqual([], revoked["controlled_action_execution"])

    async def test_explicit_out_of_scope_policy_is_rejected_even_when_disabled(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                created = await self._create()
                original = deepcopy(self.policies)
                with self.assertRaises(AIOpsAgentError) as raised:
                    await self._update(
                        created, allowed_business_system_ids=(self.new_system,),
                        controlled_action_execution=(self._policy(self.old, enabled=enabled),),
                    )
                self.assertEqual("AIOPS_AGENT_ACTION_RESOURCE_NOT_ALLOWED", raised.exception.code)
                self.assertEqual(422, raised.exception.status_code)
                self.assertEqual(original, self.policies)
                self.assertEqual(created, await self.service.get(
                    domain_id=self.identity["domain_id"], agent_id=UUID(created["agent_id"]),
                ))

    async def test_inactive_resource_or_link_revokes_only_that_policy_on_metadata_patch(self):
        for stale in ("resource", "link"):
            for changes in ({"display_name": "重命名"}, {"status": "DISABLED"}):
                with self.subTest(stale=stale, changes=changes):
                    self.old[0].status = "ACTIVE"
                    self.old[1].lifecycle_status = "ACTIVE"
                    created = await self._create(controlled_action_execution=(
                        self._policy(self.old), self._policy(self.shared),
                    ))
                    original = deepcopy(self.policies[UUID(created["agent_version_id"])])
                    if stale == "resource":
                        self.old[1].lifecycle_status = "RETIRED"
                    else:
                        self.old[0].status = "RETIRED"
                    updated = await self._update(created, **changes)
                    self.assertEqual(2, updated["version_no"])
                    self.assertNotEqual(created["agent_version_id"], updated["agent_version_id"])
                    self.assertEqual([str(self.shared[1].resource_id)], [
                        item["resource_id"] for item in updated["controlled_action_execution"]
                    ])
                    self.assertEqual(original, self.policies[UUID(created["agent_version_id"])])

    async def test_metadata_patch_without_scope_change_does_not_create_version(self):
        created = await self._create()
        updated = await self._update(created, display_name="正常重命名")
        self.assertEqual(created["agent_version_id"], updated["agent_version_id"])
        self.assertEqual(created["controlled_action_execution"], updated["controlled_action_execution"])

    async def test_shared_resource_survives_one_inactive_system_link(self):
        self.rows[self.new_system][1] = (SimpleNamespace(status="ACTIVE"), *self.shared[1:])
        created = await self._create(
            allowed_business_system_ids=(self.old_system, self.new_system),
            controlled_action_execution=(self._policy(self.shared),),
        )
        self.shared[0].status = "RETIRED"
        updated = await self._update(created, allowed_business_system_ids=(self.new_system,))
        self.assertEqual(created["controlled_action_execution"], updated["controlled_action_execution"])

    async def test_create_does_not_silently_filter_explicit_out_of_scope_policy(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                with self.assertRaises(AIOpsAgentError) as raised:
                    await self._create(controlled_action_execution=(self._policy(self.new, enabled=enabled),))
                self.assertEqual("AIOPS_AGENT_ACTION_RESOURCE_NOT_ALLOWED", raised.exception.code)
                self.assertEqual(0, self.uow.commit_count)
                self.assertEqual({}, self.uow.agents.versions)

    async def test_retained_policy_still_enforces_capability_credential_action_and_scope(self):
        for invalid, code in (
            ("capability", "AIOPS_AGENT_CHANGE_CAPABILITY_REQUIRED"),
            ("target", "AIOPS_AGENT_CHANGE_TARGET_REQUIRED"),
            ("credential", "AIOPS_AGENT_CHANGE_TARGET_REQUIRED"),
            ("action", "AIOPS_AGENT_ACTION_NOT_COMPATIBLE"),
            ("scope", "AIOPS_AGENT_ACTION_SCOPE_REQUIRED"),
        ):
            with self.subTest(invalid=invalid):
                target = self.old[3]
                target.controlled_change_enabled = True
                target.execution_credential_id = uuid7()
                target.capabilities_json = {"session_management": True}
                created = await self._create()
                original = deepcopy(self.policies)
                changes = {"instruction": "不绕过执行门禁"}
                if invalid == "capability":
                    changes["capabilities"] = (AIOpsCapability.ACTIVE_DIAGNOSIS,)
                elif invalid == "target":
                    target.controlled_change_enabled = False
                elif invalid == "credential":
                    target.execution_credential_id = None
                elif invalid == "action":
                    target.capabilities_json = {}
                else:
                    target.capabilities_json = {"dba_catalog_views": True}
                    changes["controlled_action_execution"] = (self._policy(self.old, action="db.statistics.gather"),)
                commits = self.uow.commit_count
                with self.assertRaises(AIOpsAgentError) as raised:
                    await self._update(created, **changes)
                self.assertEqual(code, raised.exception.code)
                self.assertEqual(commits, self.uow.commit_count)
                self.assertEqual(original, self.policies)
                self.assertEqual(created["agent_version_id"], str(self.uow.agents.agents[UUID(created["agent_id"])].current_version_id))

    async def test_disabled_in_scope_policy_does_not_require_change_capability(self):
        created = await self._create(
            capabilities=(AIOpsCapability.ACTIVE_DIAGNOSIS,),
            controlled_action_execution=(self._policy(self.old, enabled=False),),
        )
        updated = await self._update(created, instruction="仅诊断")
        self.assertEqual(created["controlled_action_execution"], updated["controlled_action_execution"])


if __name__ == "__main__":
    unittest.main()
