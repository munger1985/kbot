"""Agent 管理页面的自动化策略合同、版本冻结与运行门禁。"""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from pydantic import ValidationError

from aiops_agent.api.agents.routes import AgentCreateRequest, AgentUpdateRequest
from aiops_agent.application.agents import (
    AIOpsAgentService, CreateAIOpsAgentCommand, UpdateAIOpsAgentCommand,
)
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from main_api.api.aiops_app import AIOpsAgentCreatePayload, AIOpsAgentUpdatePayload
from platform_core.contracts.aiops import CreateOpsRunCommand
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_agent_creation import _UnitOfWork


@pytest.mark.parametrize("public,internal", [
    (AIOpsAgentCreatePayload, AgentCreateRequest),
    (AIOpsAgentUpdatePayload, AgentUpdateRequest),
])
@pytest.mark.parametrize("seconds", [0, 61, 86400])
def test_public_and_internal_contracts_preserve_exact_seconds(public, internal, seconds):
    fields = {"readonly_database_enabled": False, "allow_agent_execution": True, "alert_cooldown_seconds": seconds}
    required = {"expected_row_version": 1} if public is AIOpsAgentUpdatePayload else {
        "display_name": "诊断助手", "allowed_business_system_ids": [uuid7()], "capabilities": ["RESOURCE_CATALOG_QUERY"],
    }
    wire = public(**required, **fields).model_dump(mode="json", exclude_unset=True)
    result = internal.model_validate(wire).model_dump()
    for name, value in fields.items():
        assert result[name] == value
    for invalid in [-1, 86401]:
        with pytest.raises(ValidationError):
            public(**required, **{**fields, "alert_cooldown_seconds": invalid})


@pytest.mark.parametrize("seconds", [0, 61, 86400])
def test_real_create_edit_freezes_flags_and_exact_cooldown(seconds):
    uow = _UnitOfWork()

    async def add_version(row):
        assert uow.agents.agents[row.agent_id].current_version_id != row.agent_version_id
        uow.agents.versions[row.agent_version_id] = row

    uow.agents.add_version = add_version
    uow.agents.next_version_no = AsyncMock(side_effect=lambda **_: len(uow.agents.versions) + 1)
    service = AIOpsAgentService(uow_factory=lambda: uow)

    async def run():
        created = await service.create(CreateAIOpsAgentCommand(
            domain_id=100, actor_id="tester", display_name="诊断助手",
            allowed_business_system_ids=[uow.system.business_system_id], capabilities=["RESOURCE_CATALOG_QUERY"],
            knowledge_policy={"enabled": False}, readonly_database_enabled=False,
            allow_agent_execution=True, alert_cooldown_seconds=seconds,
        ))
        frozen = dict(uow.policies.rows[UUID(created["policy_id"])].rules_json)
        changed = await service.update(UpdateAIOpsAgentCommand(
            domain_id=100, actor_id="tester", agent_id=UUID(created["agent_id"]), expected_row_version=created["row_version"],
            auto_observe_min_target_level=2, readonly_database_enabled=None, allow_agent_execution=None, alert_cooldown_seconds=None,
        ))
        assert changed["readonly_database_enabled"] is False
        assert changed["allow_agent_execution"] is True
        assert changed["alert_cooldown_seconds"] == seconds
        disabled = await service.update(UpdateAIOpsAgentCommand(
            domain_id=100, actor_id="tester", agent_id=UUID(created["agent_id"]), expected_row_version=changed["row_version"],
            readonly_database_enabled=True, allow_agent_execution=False, alert_cooldown_seconds=0,
        ))
        assert disabled["readonly_database_enabled"] is True
        assert disabled["allow_agent_execution"] is False
        assert disabled["alert_cooldown_seconds"] == 0
        assert disabled["agent_version_id"] != created["agent_version_id"]
        assert uow.policies.rows[UUID(created["policy_id"])].rules_json == frozen

    asyncio.run(run())


def test_auto_alert_toggle_preserves_conditions_and_frozen_policy():
    uow = _UnitOfWork()
    service = AIOpsAgentService(uow_factory=lambda: uow)

    async def run():
        created = await service.create(CreateAIOpsAgentCommand(
            domain_id=100, actor_id="tester", display_name="自动触发配置测试",
            allowed_business_system_ids=[uow.system.business_system_id], capabilities=["RESOURCE_CATALOG_QUERY"],
            knowledge_policy={"enabled": False}, auto_alert_enabled=False,
            auto_observe_min_severity="WARNING", auto_observe_min_target_level=4, alert_cooldown_seconds=137,
        ))
        frozen = dict(uow.policies.rows[UUID(created["policy_id"])].rules_json)
        assert created["auto_alert_enabled"] is False
        changed = await service.update(UpdateAIOpsAgentCommand(
            domain_id=100, actor_id="tester", agent_id=UUID(created["agent_id"]),
            expected_row_version=created["row_version"], auto_alert_enabled=True,
        ))
        assert changed["auto_alert_enabled"] is True
        assert changed["auto_observe_min_severity"] == "WARNING"
        assert changed["auto_observe_min_target_level"] == 4
        assert changed["alert_cooldown_seconds"] == 137
        assert uow.policies.rows[UUID(created["policy_id"])].rules_json == frozen
        assert changed["agent_version_id"] != created["agent_version_id"]

    asyncio.run(run())


@pytest.mark.parametrize("agent_allows,binding_allows,target_allows,expected", [
    (False, True, True, ["AGENT_MUTATION_DISABLED"]),
    (True, False, True, ["BINDING_MUTATION_DISABLED"]),
    (True, True, True, []),
    (True, True, False, ["TARGET_CONTROLLED_CHANGE_DISABLED"]),
])
def test_real_run_creation_keeps_agent_opt_out_and_resource_authorization(agent_allows, binding_allows, target_allows, expected):
    now = datetime.now(UTC)
    agent_id, version_id, target_id, policy_id = uuid7(), uuid7(), uuid7(), uuid7()
    target = SimpleNamespace(target_id=target_id, row_version=1, db_type="ORACLE", domain_id=100, display_name="生产库",
        version_code="19c", environment="PROD", db_role="PRIMARY", deployment_type="SINGLE_INSTANCE", rac_node_number=None,
        connectivity_status="CONNECTED", endpoint_json={}, diagnostic_credential_id=None, execution_credential_id=uuid7(),
        readonly_connection_enabled=True, controlled_change_enabled=target_allows, capabilities_json={})
    binding = SimpleNamespace(binding_id=uuid7(), agent_id=agent_id, policy_id=policy_id, status="ACTIVE", row_version=1,
        allow_mutation=binding_allows, allowed_actions_json=["db.session.terminate"])
    policy = SimpleNamespace(policy_id=policy_id, status="ACTIVE", rules_json={"allow_agent_execution": agent_allows},
        policy_key="agent-policy", version_no=1, policy_hash="a" * 64, row_version=1)

    async def add_run(row):
        row.row_version = 1
        return row

    uow = SimpleNamespace(
        targets=SimpleNamespace(get_scoped=AsyncMock(return_value=target), get_agent_binding=AsyncMock(return_value=binding)),
        agents=SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(status="ACTIVE", current_version_id=version_id, agent_id=agent_id)),
            version=AsyncMock(return_value=SimpleNamespace(agent_version_id=version_id))),
        policies=SimpleNamespace(get_scoped=AsyncMock(return_value=policy)),
        runs=SimpleNamespace(database_now=AsyncMock(return_value=now), get_by_idempotency=AsyncMock(return_value=None),
            add_run=AsyncMock(side_effect=add_run), add_tasks=AsyncMock(), append_event=AsyncMock(return_value=SimpleNamespace(sequence_no=1))),
        outbox=SimpleNamespace(add=AsyncMock()), commit=AsyncMock(),
    )
    context = AsyncMock()
    context.__aenter__.return_value = uow
    blueprints = Mock()
    blueprints.resolve.return_value = SimpleNamespace(blueprint_id="kernel.observe-report", version="1", tasks=[], final_task_key="report")
    runtime = AIOpsRuntimeService(uow_factory=lambda: context, blueprint_registry=blueprints, handler_registry=Mock())
    runtime._management = SimpleNamespace(agent_execution_enabled=True)
    asyncio.run(runtime.create_run(CreateOpsRunCommand(command_id=uuid7(), idempotency_key="policy-test", domain_id=100,
        actor_id="tester", agent_id=agent_id, target_id=target_id, trigger_type="CHAT", input="诊断生产库")))
    snapshot = uow.runs.add_run.await_args.args[0].plan_snapshot_json
    assert snapshot["effective_capabilities"]["mutation_unavailable_reasons"] == expected
    assert snapshot["effective_capabilities"]["mutation_execute"] is (not expected)
    uow.commit.assert_awaited_once()
