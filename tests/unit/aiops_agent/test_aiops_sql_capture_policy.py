"""类型化估算策略通过正式Agent合同及版本化策略持久化，不由普通只读权限代替。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from aiops_agent.application.agents import AIOpsAgentService, CreateAIOpsAgentCommand
from platform_core.contracts.aiops.sql_plan_capture import SqlPlanCapturePolicy
from platform_core.identity import uuid7


@pytest.mark.parametrize("invalid", [
    {"enabled": "true"}, {"max_statement_timeout_seconds": True}, {"max_statement_timeout_seconds": 11},
    {"allowed_objects": ["mes.t", "mes.t"]}, {"allowed_objects": [" "]},
    {"allowed_objects": ["mes.t\nother.t"]}, {"sql": "SELECT 1"},
])
def test_policy_rejects_coercion_duplicate_unbounded_or_unregistered_inputs(invalid):
    with pytest.raises(ValidationError):
        SqlPlanCapturePolicy(**invalid)


def test_agent_policy_contract_and_native_versioned_policy_keep_explicit_capture_scope():
    capture = SqlPlanCapturePolicy(enabled=True, allowed_objects=("mes.t",),
        require_distinct_approver=True, max_statement_timeout_seconds=2)
    uow = SimpleNamespace(policies=SimpleNamespace(add=AsyncMock()))
    service = AIOpsAgentService(uow_factory=lambda: uow)
    values = {"domain_id": uuid7()}
    if "execution_policy" in CreateAIOpsAgentCommand.model_fields:
        from aiops_agent.application.agents import AgentExecutionPolicy
        from aiops_agent.api.agents.routes import AgentCreateRequest
        from main_api.api.aiops_agents import AIOpsAgentExecutionPolicyRequest
        assert AIOpsAgentExecutionPolicyRequest(sql_plan_capture=capture).sql_plan_capture == capture
        assert AgentCreateRequest.model_fields["execution_policy"]
        values.update(tenant_id=uuid7(), execution_policy=AgentExecutionPolicy(sql_plan_capture=capture))
    else:
        from aiops_agent.api.agents.routes import AgentCreateRequest
        from main_api.api.aiops_app import AIOpsAgentCreatePayload
        assert AIOpsAgentCreatePayload.model_fields["sql_plan_capture"].annotation is SqlPlanCapturePolicy
        assert AgentCreateRequest.model_fields["sql_plan_capture"].annotation is SqlPlanCapturePolicy
        values.update(domain_id=100, sql_plan_capture=capture.model_dump(mode="json"))
    row = asyncio.run(service._create_policy(uow=uow, agent_id=uuid7(), version_no=2,
        display_name="诊断助手", values=values, actor_id=str(uuid7())))
    assert row.rules_json["sql_plan_capture"] == capture.model_dump(mode="json")
    assert row.rules_json["readonly_database_enabled"] is True
    assert row.version_no == 2
    uow.policies.add.assert_awaited_once_with(row)


@pytest.mark.parametrize("edit", ["OTHER_FIELD", "OPTIONAL_NULL", "EXPLICIT_DISABLE"])
def test_real_agent_create_edit_and_native_policy_roundtrip_preserve_capture_scope(edit):
    from uuid import UUID
    from aiops_agent.application.agents import UpdateAIOpsAgentCommand
    from platform_core.contracts.aiops import AIOpsCapability, AgentKnowledgePolicy
    from tests.unit.aiops_agent import test_aiops_agent_creation as fixtures

    nested = "execution_policy" in CreateAIOpsAgentCommand.model_fields
    capture = SqlPlanCapturePolicy(enabled=True, allowed_objects=("mes.t",),
        require_distinct_approver=True, max_statement_timeout_seconds=2)
    if nested:
        uow = fixtures._UnitOfWork(fixtures._AgentRepository(), uuid7())
        scope = {"tenant_id": fixtures.TENANT_ID, "domain_id": fixtures.DOMAIN_ID, "actor_id": fixtures.ACTOR_ID}
        systems = (uow.business_system_id,)
        specific = {"execution_policy": {"sql_plan_capture": capture, "auto_observe_min_severity": "HIGH"}}
    else:
        uow = fixtures._UnitOfWork(rows=(fixtures._database_row(),))
        scope = {"domain_id": 100, "actor_id": "tester"}
        systems = (uow.system.business_system_id,)
        specific = {"sql_plan_capture": capture}
        # 原夹具只覆盖首版写入；扩展为真实仓储允许的先写新版再换指针，不跳过生产服务。
        async def add_version(row):
            assert uow.agents.agents[row.agent_id].current_version_id != row.agent_version_id
            uow.agents.versions[row.agent_version_id] = row
        uow.agents.add_version = add_version
        uow.agents.next_version_no = AsyncMock(side_effect=lambda **_: 1 + max(
            int(row.version_no) for row in uow.agents.versions.values()))
    service = AIOpsAgentService(uow_factory=lambda: uow)

    async def run():
        created = await service.create(CreateAIOpsAgentCommand(**scope, display_name="范围测试助手",
            allowed_business_system_ids=systems, capabilities=(AIOpsCapability.RESOURCE_CATALOG_QUERY,),
            knowledge_policy=AgentKnowledgePolicy(enabled=False), **specific))
        if edit == "OTHER_FIELD":
            update = {"auto_observe_min_target_level": 2}
            if nested:
                update = {"execution_policy": {"readonly_database_enabled": False}}
        elif edit == "OPTIONAL_NULL":
            update = {"execution_policy" if nested else "sql_plan_capture": None}
        else:
            update = {"execution_policy": {"sql_plan_capture": {"enabled": False}}} if nested else {"sql_plan_capture": {"enabled": False}}
        changed = await service.update(UpdateAIOpsAgentCommand(**scope, agent_id=UUID(created["agent_id"]),
            expected_row_version=created["row_version"], **update))
        result = changed["execution_policy"]["sql_plan_capture"] if nested else changed["sql_plan_capture"]
        assert SqlPlanCapturePolicy.model_validate(result) == (SqlPlanCapturePolicy() if edit == "EXPLICIT_DISABLE" else capture)
        if nested:
            assert changed["execution_policy"]["auto_observe_min_severity"] == "HIGH"
        return changed
    asyncio.run(run())
