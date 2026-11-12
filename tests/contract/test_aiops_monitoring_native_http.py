"""原生监控公开入口认证、参数白名单、可信上下文与快照合同。"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest

from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from main_api.api import ops
from main_api import app as main_app
from platform_clients.aiops import AIOpsManagementClient, AIOpsClientError
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.identity import uuid7


def route(name):
    return next(item for item in ops.router.routes if isinstance(item, APIRoute) and item.endpoint.__name__ == name)


def endpoint(name, source, instance):
    return route(name).path.replace("{domain_code}", "test").replace("{source_id}", str(source)).replace("{instance_id}", str(instance))


@pytest.mark.parametrize("code", ["SOURCE_PROVIDER_ERROR", "SOURCE_UNREACHABLE", "SOURCE_RESULT_TRUNCATED",
    "SOURCE_RATE_LIMITED", "SOURCE_COLLECTION_CHANGED"])
def test_only_oem_native_routes_preserve_source_failure_classification(code):
    factory = getattr(main_app, "create_ammolite_main_api_app", None) or main_app.create_main_api_app
    app = factory(enable_access_log=False)
    handler = app.exception_handlers[AIOpsClientError]
    error = AIOpsClientError(status_code=502, code=code, message="来源错误原文不应直接暴露")
    for endpoint in (ops.list_monitoring_native_metrics, ops.query_monitoring_native_metric):
        request = Request({"type": "http", "method": "GET", "path": "/test", "headers": [], "endpoint": endpoint})
        response = asyncio.run(handler(request, error))
        payload = json.loads(response.body)
        assert response.status_code == 502 and payload["code"] == code
        assert "来源错误原文" not in payload["detail"]
    request = Request({"type": "http", "method": "GET", "path": "/test", "headers": []})
    response = asyncio.run(handler(request, error))
    assert response.status_code == 503 and json.loads(response.body)["code"] == "AIOPS_UNAVAILABLE"


def test_native_routes_reject_unauthenticated_requests():
    factory = getattr(main_app, "create_ammolite_main_api_app", None) or main_app.create_main_api_app
    client = TestClient(factory(enable_access_log=False))
    for name, method in (("list_monitoring_native_metrics", "GET"), ("query_monitoring_native_metric", "POST")):
        assert client.request(method, endpoint(name, uuid7(), uuid7()), headers={"X-Ammolite-Entry": "BUSINESS"}).status_code == 401


def test_native_catalog_proxy_preserves_context_and_enforces_budgets():
    source, instance = uuid7(), uuid7()
    trusted = AuthContext(principal_kind=PrincipalKind.PORTAL, client_id="native", api_key_id="test-key",
        request_id="native", trace_id="native", app_id="aiops", domain_id=str(uuid7()), tenant_id=str(uuid7()), asserted_user_id="actor")
    app = FastAPI()
    app.include_router(ops.router)
    async def authorized(request: Request):
        request.state.auth_context = trusted
    for item in (route("list_monitoring_native_metrics"), route("query_monitoring_native_metric")):
        for dependency in item.dependant.dependencies:
            app.dependency_overrides[dependency.call] = authorized
    downstream = SimpleNamespace(
        list_monitoring_native_metrics=AsyncMock(return_value={"source_id":str(source), "instance_id":str(instance),
            "entries":[], "discovered_count":0, "truncated":False, "revision":"a"*64}),
        query_monitoring_native_metric=AsyncMock(),
    )
    with patch.object(ops, "_client", return_value=downstream):
        client = TestClient(app)
        path = endpoint("list_monitoring_native_metrics", source, instance)
        response = client.get(path+"?search=CPU&offset=100&limit=20")
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        downstream.list_monitoring_native_metrics.assert_awaited_once_with(source, instance,
            search="CPU", offset=100, limit=20, auth_context=trusted)
        downstream.list_monitoring_native_metrics.reset_mock()
        for query in ("limit=0", "limit=101", "offset=-1", "offset=100001", "search="+"x"*129):
            assert client.get(path+"?"+query).status_code == 422
        downstream.list_monitoring_native_metrics.assert_not_awaited()
        response = client.post(endpoint("query_monitoring_native_metric", source, instance),
            json={"metric_name":"Load/cpu", "source_locator_key":"untrusted"})
        assert response.status_code == 422
        downstream.query_monitoring_native_metric.assert_not_awaited()
        assert client.post(endpoint("query_monitoring_native_metric",source,instance),
            json={"metric_name":"Load/cpu","baseline":"1d"}).status_code == 422


def test_native_baseline_contract_forbids_partial_averages_and_false_windows():
    import pytest
    from platform_core.contracts.aiops.monitoring_native import MonitoringNativeBaseline, MonitoringNativeQuery
    from pydantic import ValidationError
    payload = {"period":"1w","start":"2026-10-29T14:00:00Z","end":"2026-11-05T14:00:00Z",
        "series":[{"dimensions":{"mountPoint":"/"},"average":"9.125","sample_count":12}]}
    assert MonitoringNativeBaseline.model_validate(payload).series[0].average == "9.125"
    for change in ({"truncated":True},{"unavailable_reason":"来源不可用"}, {"start":"2026-11-05T15:00:00Z"},
        {"series":[{"dimensions":{},"average":"NaN","sample_count":12}]}):
        with pytest.raises(ValidationError):
            MonitoringNativeBaseline.model_validate({**payload,**change})
    for period in ("1w","1m"):
        assert MonitoringNativeQuery(metric_name="Load/cpu",baseline=period).baseline == period


def test_native_permissions_and_encoded_client_query():
    for name, method in (("list_monitoring_native_metrics", "GET"), ("query_monitoring_native_metric", "POST")):
        if hasattr(ops, "AIOPS_ROUTE_PERMISSIONS"):
            assert ops.AIOPS_ROUTE_PERMISSIONS[name] == "aiops:use"
        else:
            from platform_core.authorization import APP_AUTHORIZATION_POLICIES
            assert any(policy.route_permission(path=route(name).path, method=method) == "aiops:use" for policy in APP_AUTHORIZATION_POLICIES.values())
    client = object.__new__(AIOpsManagementClient)
    client._json = AsyncMock(return_value={})
    source, instance, trusted = uuid7(), uuid7(), object()
    asyncio.run(client.list_monitoring_native_metrics(source, instance, search="CPU&x=1", offset=100, limit=100, auth_context=trusted))
    client._json.assert_awaited_once_with("GET", f"{client._MONITORING}/sources/{source}/resources/{instance}/metrics?search=CPU%26x%3D1&offset=100&limit=100", auth_context=trusted)


def test_native_openapi_routes_and_models_match_snapshots():
    from aiops_agent.bootstrap.openapi import create_internal_contract_app
    from main_api.openapi_contracts import create_aiops_public_contract_app
    factory = getattr(main_app, "create_ammolite_main_api_app", None) or main_app.create_main_api_app
    main_name = "ammolite_main_api_public_v1.json" if hasattr(main_app, "create_ammolite_main_api_app") else "main_api_public_v1.json"
    apps = {"aiops_internal_v1.json":create_internal_contract_app(), "aiops_public_v1.json":create_aiops_public_contract_app(), main_name:factory(enable_access_log=False)}
    for filename, app in apps.items():
        live = app.openapi()
        stored = json.loads((Path(__file__).resolve().parents[2]/"docs/openapi"/filename).read_text())
        paths = [key for key in live["paths"] if "/resources/{instance_id}/metric" in key]
        assert len(paths) == 2
        for path in paths:
            assert stored["paths"][path] == live["paths"][path]
        for name, model in live["components"]["schemas"].items():
            if name.startswith("MonitoringNative"):
                assert stored["components"]["schemas"][name] == model
