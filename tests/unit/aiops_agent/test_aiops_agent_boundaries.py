"""诊断边界配置预览与 Target 权限上限，不连接真实数据库。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiops_agent.application.agents import AIOpsAgentError, AIOpsAgentService, CreateAIOpsAgentCommand, UpdateAIOpsAgentCommand
from aiops_agent.application.credential_issuance import CredentialIssueError, CredentialIssuanceService
from aiops_agent.diagnostics.grants import DiagnosticGrantError
from platform_core.identity import uuid7


def boundary_fixture():
    domain_id = 100 if CreateAIOpsAgentCommand.model_fields["domain_id"].annotation is int else uuid7()
    scope = {"domain_id": domain_id}
    if "tenant_id" in CreateAIOpsAgentCommand.model_fields:
        scope["tenant_id"] = uuid7()
    system = SimpleNamespace(business_system_id=uuid7(), status="ENABLED", tenant_id=scope.get("tenant_id"))
    resource = SimpleNamespace(resource_id=uuid7(), display_name="MES", lifecycle_status="ACTIVE", resource_type_key="POSTGRESQL", domain_id=domain_id, tenant_id=scope.get("tenant_id"))
    target = SimpleNamespace(target_id=uuid7(), status="ENABLED", db_type="POSTGRESQL",
        endpoint_json={"host": "not-returned"}, diagnostic_credential_id=uuid7(), execution_credential_id=uuid7(),
        readonly_connection_enabled=True, controlled_change_enabled=False)
    link = SimpleNamespace(status="ACTIVE")
    database = SimpleNamespace(target_id=target.target_id)
    source = SimpleNamespace(status="ENABLED")
    uow = SimpleNamespace(
        business_systems=SimpleNamespace(get_scoped=AsyncMock(return_value=system),
            list_system_resource_rows=AsyncMock(return_value=[(link, resource, database, target)])),
        targets=SimpleNamespace(get_scoped=AsyncMock(return_value=target),
            list_source_bindings=AsyncMock(return_value=[SimpleNamespace(diagnostic_source_id=uuid7())])),
        diagnostic_sources=SimpleNamespace(get_scoped=AsyncMock(return_value=source)),
        commit=AsyncMock(), managed_credentials=SimpleNamespace(read=AsyncMock()),
    )
    context = AsyncMock()
    context.__aenter__.return_value = uow
    service = AIOpsAgentService(uow_factory=lambda: context)
    return SimpleNamespace(service=service, scope=scope, system=system, resource=resource, target=target,
        source=source, link=link, uow=uow)


@pytest.mark.parametrize("mode", ["readonly", "change", "missing_credential", "missing_endpoint", "target_disabled", "scope_disabled", "source_disabled", "no_source", "no_target", "inactive_resource", "other_space"])
def test_preview_is_configuration_only_and_never_escalates_target(mode):
    f = boundary_fixture()
    if mode == "change": f.target.controlled_change_enabled = True
    if mode == "missing_credential": f.target.diagnostic_credential_id = f.target.execution_credential_id = None
    if mode == "missing_endpoint": f.target.endpoint_json = None
    if mode == "target_disabled": f.target.status = "DISABLED"
    if mode == "scope_disabled": f.system.status = "DISABLED"
    if mode == "source_disabled": f.source.status = "DISABLED"
    if mode == "no_source": f.uow.diagnostic_sources.get_scoped.return_value = None
    if mode == "no_target": f.uow.targets.get_scoped.return_value = None
    if mode == "inactive_resource": f.resource.lifecycle_status = "INACTIVE"
    if mode == "other_space": f.resource.domain_id = uuid7()
    result = asyncio.run(f.service.boundary_resources(**f.scope, business_system_ids=(f.system.business_system_id,)))
    if mode in {"inactive_resource", "other_space"}:
        assert result == {"items": []}
    else:
        row = result["items"][0]
        assert row["change_configured"] is (mode == "change")
        assert row["readonly_configured"] is (mode in {"readonly", "change", "source_disabled", "no_source"})
        assert row["monitoring_configured"] is (mode in {"readonly", "change", "missing_credential", "missing_endpoint"})
        assert not {"target_id", "username", "password", "endpoint_json", "diagnostic_credential_id", "execution_credential_id"} & row.keys()
    f.uow.commit.assert_not_awaited()
    f.uow.managed_credentials.read.assert_not_awaited()


def test_preview_deduplicates_shared_targets_and_scope_ids():
    f = boundary_fixture()
    second_id = uuid7()
    other = SimpleNamespace(business_system_id=second_id, status="ENABLED", tenant_id=f.scope.get("tenant_id"))
    f.uow.business_systems.get_scoped.side_effect = [f.system, other]
    result = asyncio.run(f.service.boundary_resources(**f.scope, business_system_ids=(f.system.business_system_id, f.system.business_system_id, second_id)))
    assert len(result["items"]) == 1
    assert result["items"][0]["business_system_ids"] == [str(f.system.business_system_id), str(second_id)]
    assert f.uow.business_systems.get_scoped.await_count == 2
    f.uow.targets.list_source_bindings.assert_awaited_once()


def test_preview_rejects_inaccessible_business_system_before_reading_resources():
    f = boundary_fixture()
    f.uow.business_systems.get_scoped.return_value = None
    with pytest.raises(AIOpsAgentError) as error:
        asyncio.run(f.service.boundary_resources(**f.scope, business_system_ids=(uuid7(),)))
    assert error.value.status_code == 404
    assert f.uow.business_systems.get_scoped.await_args.kwargs.items() >= f.scope.items()
    f.uow.business_systems.list_system_resource_rows.assert_not_awaited()
    f.uow.targets.get_scoped.assert_not_awaited()


def test_command_defaults_monitor_only_and_update_keeps_omitted_policy():
    scope = boundary_fixture().scope
    actor = "tester" if isinstance(scope["domain_id"], int) else uuid7()
    command = CreateAIOpsAgentCommand(**scope, actor_id=actor, display_name="诊断助手",
        allowed_business_system_ids=(uuid7(),), capabilities=("RESOURCE_CATALOG_QUERY",), knowledge_policy={"enabled": False})
    assert "MONITORING_QUERY" in command.capabilities
    policy = command.model_dump().get("execution_policy", command.model_dump())
    assert policy["readonly_database_enabled"] is False
    assert policy["allow_agent_execution"] is False
    changed = UpdateAIOpsAgentCommand(**scope, actor_id=actor, agent_id=uuid7(), expected_row_version=1,
        capabilities=("RESOURCE_CATALOG_QUERY",))
    assert "MONITORING_QUERY" in changed.capabilities
    assert "execution_policy" not in changed.model_dump(exclude_unset=True)
    assert "readonly_database_enabled" not in changed.model_dump(exclude_unset=True)


@pytest.mark.parametrize("state", ["readonly", "disabled"])
def test_target_can_deny_mutation_credential_even_with_valid_signed_grant(state):
    stamp = datetime.now(UTC)
    target = SimpleNamespace(status="ENABLED", row_version=1, controlled_change_enabled=state != "readonly", execution_credential_id=uuid7())
    if state == "disabled": target.status = "DISABLED"
    grant = SimpleNamespace(audience="test", expires_at=stamp + timedelta(minutes=1), grant_id=uuid7(),
        target_id=uuid7(), domain_id=100, target_version=1, execution_credential_id=target.execution_credential_id)
    diagnostic = SimpleNamespace(verify=Mock(side_effect=DiagnosticGrantError("NOT_DIAGNOSTIC", "非诊断令牌")),
        verify_dynamic=Mock(side_effect=DiagnosticGrantError("NOT_DIAGNOSTIC", "非动态诊断令牌")))
    credentials = SimpleNamespace(read=AsyncMock())
    uow = SimpleNamespace(targets=SimpleNamespace(get_scoped=AsyncMock(return_value=target)),
        inbox=SimpleNamespace(get_by_message=AsyncMock()), managed_credentials=SimpleNamespace())
    context = AsyncMock()
    context.__aenter__.return_value = uow
    service = CredentialIssuanceService(uow_factory=lambda: context, diagnostic_codec=diagnostic,
        mutation_codec=SimpleNamespace(verify=Mock(return_value=grant)), credential_service=credentials, audience="test")
    with pytest.raises(CredentialIssueError):
        asyncio.run(service.issue("signed-execution-token"))
    credentials.read.assert_not_awaited()
    uow.inbox.get_by_message.assert_not_awaited()

def test_internal_preview_route_uses_trusted_scope_and_limits_query(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from aiops_agent.api.agents import routes

    f = boundary_fixture()
    app = FastAPI()
    app.include_router(routes.router)
    app.state.agent_service = SimpleNamespace(boundary_resources=AsyncMock(return_value={"items": []}))
    context = SimpleNamespace(tenant_id=str(f.scope.get("tenant_id", uuid7())))
    app.dependency_overrides[routes.get_aiops_auth_context] = lambda: context
    actor = "tester" if isinstance(f.scope["domain_id"], int) else uuid7()
    monkeypatch.setattr(routes, "_scope", lambda request, auth: (f.scope["domain_id"], actor))
    with TestClient(app) as client:
        path = "/internal/v1/aiops/agents/boundary-resources"
        result = client.get(path, params=[("business_system_ids", str(f.system.business_system_id)), ("domain_id", str(uuid7()))])
        assert result.status_code == 200
        assert result.json() == {"items": []}
        assert app.state.agent_service.boundary_resources.await_args.kwargs == {
            **f.scope, "business_system_ids": (f.system.business_system_id,),
        }
        assert client.get(path, params=[("business_system_ids", str(uuid7())) for _ in range(33)]).status_code == 422
        assert app.state.agent_service.boundary_resources.await_count == 1
