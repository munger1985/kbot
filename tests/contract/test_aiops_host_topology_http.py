"""共享主机公开入口、可信上下文转发和只读合同测试。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from main_api.api import ops
from main_api import app as main_app
from platform_clients.aiops import AIOpsManagementClient
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.identity import uuid7


def route():
    return next(item for item in ops.router.routes if isinstance(item, APIRoute) and item.endpoint.__name__ == "get_host_topology")


def path(system, host):
    return route().path.replace("{domain_code}", "test").replace("{business_system_id}", str(system)).replace("{resource_id}", str(host))


def test_public_host_topology_rejects_unauthenticated_requests():
    factory = getattr(main_app, "create_ammolite_main_api_app", None) or main_app.create_main_api_app
    app = factory(enable_access_log=False)
    downstream = SimpleNamespace(get_host_topology=AsyncMock())
    with patch.object(ops, "_client", return_value=downstream):
        response = TestClient(app).get(path(uuid7(), uuid7()), headers={"X-Ammolite-Entry": "BUSINESS"})
    assert response.status_code == 401, response.text
    downstream.get_host_topology.assert_not_awaited()


def test_proxy_preserves_trusted_context_and_declared_limits():
    system, host = uuid7(), uuid7()
    trusted = AuthContext(principal_kind=PrincipalKind.PORTAL, client_id="test-client", api_key_id="test-key", request_id="host-topology", trace_id="host-topology", app_id="aiops", domain_id=str(uuid7()), tenant_id=str(uuid7()), asserted_user_id="test-user")
    app = FastAPI()
    app.include_router(ops.router)

    async def authorized(request: Request):
        request.state.auth_context = trusted

    # 只在代理单测替代已授权依赖；裸请求拒绝由上一项真实 App 测试验证。
    for dependency in route().dependant.dependencies:
        app.dependency_overrides[dependency.call] = authorized
    downstream = SimpleNamespace(get_host_topology=AsyncMock(return_value={"host_resource_id": str(host), "max_nodes": 7, "resources": [], "relations": [], "truncated": False}))
    with patch.object(ops, "_client", return_value=downstream):
        client = TestClient(app)
        response = client.get(path(system, host) + "?max_nodes=7&max_depth=2&domain_id=untrusted", headers={"X-Domain-ID": "untrusted"})
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        downstream.get_host_topology.assert_awaited_once_with(system, host, max_nodes=7, max_depth=2, auth_context=trusted)
        downstream.get_host_topology.reset_mock()
        for query in ("max_nodes=0", "max_nodes=501", "max_depth=0", "max_depth=7"):
            assert client.get(path(system, host) + "?" + query).status_code == 422
        downstream.get_host_topology.assert_not_awaited()


def test_host_topology_permission_is_registered_in_platform_gate():
    if hasattr(ops, "AIOPS_ROUTE_PERMISSIONS"):
        assert ops.AIOPS_ROUTE_PERMISSIONS["get_host_topology"] == "aiops:use"
    else:
        from platform_core.authorization import APP_AUTHORIZATION_POLICIES
        assert any(policy.route_permission(path=route().path, method="GET") == "aiops:use" for policy in APP_AUTHORIZATION_POLICIES.values())


def test_internal_client_forwards_context_without_caller_scope_parameters():
    client = object.__new__(AIOpsManagementClient)
    client._json = AsyncMock(return_value={})
    system, host, trusted = uuid7(), uuid7(), object()
    asyncio.run(client.get_host_topology(system, host, max_nodes=9, max_depth=3, auth_context=trusted))
    client._json.assert_awaited_once_with("GET", f"{client._CONFIG}/business-systems/{system}/hosts/{host}/topology?max_nodes=9&max_depth=3", auth_context=trusted)


def test_openapi_snapshots_match_host_topology_route_and_projection():
    import json
    from pathlib import Path
    from aiops_agent.bootstrap.openapi import create_internal_contract_app
    from main_api.openapi_contracts import create_aiops_public_contract_app

    root = Path(__file__).resolve().parents[2] / "docs" / "openapi"
    factory = getattr(main_app, "create_ammolite_main_api_app", None) or main_app.create_main_api_app
    main_name = "ammolite_main_api_public_v1.json" if hasattr(main_app, "create_ammolite_main_api_app") else "main_api_public_v1.json"
    apps = {"aiops_internal_v1.json": create_internal_contract_app(), "aiops_public_v1.json": create_aiops_public_contract_app(), main_name: factory(enable_access_log=False)}
    for name, app in apps.items():
        live = app.openapi()
        stored = json.loads((root / name).read_text())
        paths = [key for key in live["paths"] if "/hosts/{resource_id}/topology" in key]
        assert len(paths) == 1
        assert stored["paths"][paths[0]] == live["paths"][paths[0]]
        for key, value in live["components"]["schemas"].items():
            if key.startswith("HostTopology"):
                assert stored["components"]["schemas"][key] == value
