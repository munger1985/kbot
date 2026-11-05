"""用真实仓储和内存事务验证授权收缩、旧版本保留以及失败回滚。"""

import asyncio
from copy import deepcopy
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aiops_agent.actions import ActionRegistry
from aiops_agent.application.agents import (
    AIOpsAgentError, AIOpsAgentService, CreateAIOpsAgentCommand,
    ResourceControlledActionExecution, UpdateAIOpsAgentCommand,
)
from aiops_agent.entities import AIOpsAgentEntity, AIOpsAgentVersionEntity, PolicyEntity, TargetEntity
from aiops_agent.entities import agent as agent_entities
from aiops_agent.repositories.agent import AIOpsAgentRepository
from aiops_agent.repositories.target import PolicyRepository
from platform_core.contracts.aiops import AIOpsCapability, AgentKnowledgePolicy
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_business_system_publication_lifecycle import (
    LocalUow, SessionBridge, lab, publish,
)


class AgentSessionBridge(SessionBridge):
    async def get(self, *args, **kwargs):
        return self.sync.get(*args, **kwargs)

    async def scalar(self, *args, **kwargs):
        return self.sync.scalar(*args, **kwargs)

    async def scalars(self, *args, **kwargs):
        return self.sync.scalars(*args, **kwargs)


class AgentLocalUow(LocalUow):
    def __init__(self, engine, *, fail_commit=False):
        super().__init__(engine)
        self.agents = AIOpsAgentRepository(AgentSessionBridge(self.sync), lambda: None)
        self.policies = PolicyRepository(AgentSessionBridge(self.sync))
        self.fail_commit = fail_commit

    async def commit(self):
        self.sync.flush()
        if self.fail_commit:
            raise RuntimeError("测试注入：授权版本提交失败")
        await super().commit()


@pytest.mark.parametrize("failure", [None, "explicit_policy", "commit"])
def test_scope_switch_persists_only_new_version_or_rolls_back(lab, failure):
    async def run():
        tables = [
            row.__table__ for name, row in vars(agent_entities).items()
            if name.startswith("AIOpsAgent") and hasattr(row, "__table__")
        ] + [PolicyEntity.__table__]
        AIOpsAgentEntity.metadata.create_all(lab.engine, tables=tables)
        old_system = await publish(lab)
        lab.key = str(uuid7())
        lab.request = lab.request.model_copy(update={
            "system": lab.request.system.model_copy(update={"display_name": "新 OEM 联调系统"}),
        })
        new_system = await publish(lab)
        old_database = old_system.resources[0].resource.database
        old_resource_id = old_system.resources[0].resource.resource_id
        with Session(lab.engine) as session:
            target = session.get(TargetEntity, old_database.target_id)
            target.controlled_change_enabled = True
            target.execution_credential_id = uuid7()
            target.capabilities_json = {"session_management": True}
            session.commit()
        has_tenant = hasattr(lab.scope, "tenant_id")
        identity = {
            "domain_id": lab.scope.domain_id,
            "actor_id": UUID(lab.scope.actor_id) if has_tenant else lab.scope.actor_id,
        }
        if has_tenant:
            identity["tenant_id"] = lab.scope.tenant_id
        fail_commit = False
        service = AIOpsAgentService(
            uow_factory=lambda: AgentLocalUow(lab.engine, fail_commit=fail_commit),
            action_registry=ActionRegistry.load(),
        )
        old_action = ResourceControlledActionExecution(
            resource_id=old_resource_id, enabled=True,
            allowed_action_ids=("db.session.terminate",),
        )
        created = await service.create(CreateAIOpsAgentCommand(
            **identity, display_name="数据库范围事务测试", status="ACTIVE",
            allowed_business_system_ids=(old_system.business_system_id,),
            capabilities=(AIOpsCapability.ACTIVE_DIAGNOSIS, AIOpsCapability.CONTROLLED_CHANGE),
            knowledge_policy=AgentKnowledgePolicy(enabled=False),
            models={"planner_llm": uuid7(), "diagnosis_llm": uuid7(), "composer_llm": uuid7()},
            controlled_action_execution=(old_action,),
        ))
        agent_id, original_version = UUID(created["agent_id"]), UUID(created["agent_version_id"])
        async with AgentLocalUow(lab.engine) as uow:
            original_policies = deepcopy(await uow.agents.version_resource_policies(agent_version_id=original_version))
            original_systems = await uow.agents.version_business_system_ids(agent_version_id=original_version)
        with Session(lab.engine) as session:
            original_rules = deepcopy(session.get(PolicyEntity, UUID(created["policy_id"])).rules_json)
        changes = {"allowed_business_system_ids": (new_system.business_system_id,)}
        if failure == "explicit_policy":
            changes["controlled_action_execution"] = (old_action,)
        fail_commit = failure == "commit"
        command = UpdateAIOpsAgentCommand(
            **identity, agent_id=agent_id, expected_row_version=created["row_version"], **changes,
        )
        if failure:
            with pytest.raises(AIOpsAgentError if failure == "explicit_policy" else RuntimeError) as raised:
                await service.update(command)
            if failure == "explicit_policy":
                assert raised.value.code == "AIOPS_AGENT_ACTION_RESOURCE_NOT_ALLOWED"
        else:
            updated = await service.update(command)
            assert updated["version_no"] == 2
            assert updated["controlled_action_execution"] == []
            assert updated["allowed_business_system_ids"] == [str(new_system.business_system_id)]
            assert updated["agent_version_id"] != created["agent_version_id"]
        async with AgentLocalUow(lab.engine) as uow:
            assert original_policies == await uow.agents.version_resource_policies(agent_version_id=original_version)
            assert original_systems == await uow.agents.version_business_system_ids(agent_version_id=original_version)
        with Session(lab.engine) as session:
            original_policy = session.get(PolicyEntity, UUID(created["policy_id"]))
            assert original_policy.rules_json == original_rules
            assert original_policy.status == ("ACTIVE" if failure else "RETIRED")
            assert session.scalar(select(func.count()).select_from(AIOpsAgentVersionEntity)) == (1 if failure else 2)
            assert session.scalar(select(func.count()).select_from(PolicyEntity)) == (1 if failure else 2)
            agent = session.get(AIOpsAgentEntity, agent_id)
            assert agent.row_version == (created["row_version"] if failure else updated["row_version"])
            assert str(agent.current_version_id) == (created["agent_version_id"] if failure else updated["agent_version_id"])
    asyncio.run(run())
