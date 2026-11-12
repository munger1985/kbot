"""Exadata 公开 API 的认证、代理、合同与幂等转发测试。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest
from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from main_api.api import ops
from main_api import app as main_app
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.identity import uuid7


def route(name):
    return next(row for row in ops.router.routes if isinstance(row, APIRoute) and row.endpoint.__name__ == name)


def path(name, system, resource, source, report=None):
    return (route(name).path.replace("{domain_code}", "test")
        .replace("{system_id}", str(system)).replace("{resource_id}", str(resource))
        .replace("{source_id}", str(source)).replace("{report_id}", str(report or uuid7())))


@pytest.mark.parametrize("name,method", [
    ("discover_exadata", "POST"), ("get_exadata_view", "POST"),
    ("diagnose_exadata", "POST"), ("get_exadata_baseline", "POST"), ("get_exadata_report", "GET")])
def test_exadata_public_endpoints_require_authentication(name, method):
    factory = getattr(main_app, "create_ammolite_main_api_app", None) or main_app.create_main_api_app
    app = factory(enable_access_log=False)
    response = TestClient(app).request(method, path(name, uuid7(), uuid7(), uuid7()),
        json={} if method == "POST" else None, headers={"X-Ammolite-Entry": "BUSINESS"})
    assert response.status_code == 401, response.text


@pytest.mark.parametrize("name,method,permission", [
    ("discover_exadata", "POST", "aiops:diagnostic_source_manage"),
    ("get_exadata_view", "POST", "aiops:use"), ("diagnose_exadata", "POST", "aiops:use"),
    ("get_exadata_baseline", "POST", "aiops:use"),
    ("get_exadata_report", "GET", "aiops:use")])
def test_permissions_explicitly_registered(name, method, permission):
    if hasattr(ops, "AIOPS_ROUTE_PERMISSIONS"):
        assert ops.AIOPS_ROUTE_PERMISSIONS[name] == permission
    else:
        from platform_core.authorization import APP_AUTHORIZATION_POLICIES
        assert any(policy.route_permission(path=route(name).path, method=method) == permission
            for policy in APP_AUTHORIZATION_POLICIES.values())


@pytest.mark.parametrize("scene", ["disk", "dbserver"])
def test_proxy_rejects_injected_locator_and_preserves_trusted_context(scene):
    system, resource, source, binding = (uuid7() for _ in range(4))
    trusted = AuthContext(principal_kind=PrincipalKind.PORTAL, client_id="test-client", api_key_id="test-key",
        request_id="exadata-http", trace_id="exadata-http", app_id="aiops", domain_id=str(uuid7()),
        tenant_id=str(uuid7()), asserted_user_id="test-user")
    payload = {"resource_id": str(resource), "binding_id": str(binding), "binding_version": 1, "source_version": 1,
        "catalog_version": "exadata-oem-v1", "catalog_sha256": "a" * 64, "captured_at": "2026-11-11T00:00:00Z",
        "scene": "disk", "members": [], "samples": [], "definitions": [], "gaps": [], "partial": True, "total_queries": 0}
    downstream = SimpleNamespace(exadata_view=AsyncMock(return_value=payload), diagnose_exadata=AsyncMock())
    app = FastAPI()
    app.include_router(ops.router)
    async def authorized(request: Request):
        request.state.auth_context = trusted
    for name in ("get_exadata_view", "diagnose_exadata"):
        for dependency in route(name).dependant.dependencies:
            app.dependency_overrides[dependency.call] = authorized
    with patch.object(ops, "_client", return_value=downstream):
        client = TestClient(app)
        endpoint = path("get_exadata_view", system, resource, source)
        response = client.post(endpoint, json={"scene": scene})
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        downstream.exadata_view.assert_awaited_once_with(system, resource, source,
            {"mode": "LATEST_SNAPSHOT", "window_minutes": 60, "scene": scene, "component_id": None, "offset": 0, "limit": 12}, auth_context=trusted)
        downstream.exadata_view.reset_mock()
        for body in ({"source_locator_key": "untrusted"}, {"target_id": str(uuid7())}, {"sql": "select 1"},
                     {"limit": 25}, {"component_id": "../other"}):
            assert client.post(endpoint, json=body).status_code == 422
        downstream.exadata_view.assert_not_awaited()
        assert client.post(path("diagnose_exadata", system, resource, source), json={}).status_code == 422
        downstream.diagnose_exadata.assert_not_awaited()
