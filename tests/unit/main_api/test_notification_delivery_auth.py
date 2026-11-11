"""KBot 由 AIOps 应用管理员治理通知，不继承平台或其他 App 的管理权。"""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from main_api.api.notification_delivery_auth import external_capabilities, external_context, external_entry_actor
from main_api.api.notification_delivery import router
from main_api.application.access_control import AccessSnapshot
from platform_core.contracts import AuthContext


def request(*, kind="PORTAL", roles=("app_admin",), permissions=("aiops:use",),
            app_id="aiops", token_roles=(), domain="12"):
    context = AuthContext(principal_kind=kind, app_id=app_id, domain_id=domain,
        client_id="fixture", api_key_id="fixture", asserted_user_id="fixture-user",
        request_id="fixture", trace_id="fixture", roles=token_roles)
    snapshot = AccessSnapshot(app_id="aiops", domain_id=12, user_id="fixture-user",
        roles=roles, permissions=frozenset(permissions))
    access = NS(snapshot=AsyncMock(return_value=snapshot), require_platform=AsyncMock(),
        platform_snapshot=AsyncMock())
    authority = NS(actor_active=AsyncMock(return_value=True))
    return NS(state=NS(auth_context=context), app=NS(state=NS(
        access_control_service=access, external_notification_service=NS(authority=authority))))


@pytest.mark.asyncio
@pytest.mark.parametrize("permission", ["channel", "binding", "read", "retry"])
async def test_aiops_app_admin_manages_global_notifications_without_platform_grant(permission):
    req = request()
    assert (await external_capabilities(req))["capabilities"] == ["channel", "binding", "read", "retry"]
    scope, actor = await external_context(req, permission=permission)
    assert scope.domain_id is None and scope.tenant_id is None and actor == "fixture-user"
    req.app.state.access_control_service.snapshot.assert_awaited_once_with(
        app_id="aiops", domain_id=12, user_id="fixture-user")
    req.app.state.access_control_service.require_platform.assert_not_awaited()
    req.app.state.access_control_service.platform_snapshot.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("roles", [(), ("app_user",), ("app_manager",), ("platform_admin",), ("custom_admin",)])
@pytest.mark.parametrize("permission", ["channel", "binding", "read", "retry"])
async def test_other_roles_cannot_manage_even_with_stale_admin_claim(roles, permission):
    req = request(roles=roles, token_roles=("app_admin",))
    assert (await external_capabilities(req))["capabilities"] == []
    with pytest.raises(HTTPException) as error:
        await external_context(req, permission=permission)
    assert error.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["SERVICE", "APP_API_CLIENT", "API_CLIENT"])
async def test_machine_cannot_manage_report_capabilities_or_resolve_personal_entry(kind):
    req = request(kind=kind)
    for operation in [external_capabilities(req), external_context(req, permission="channel"), external_entry_actor(req)]:
        with pytest.raises(HTTPException) as error:
            await operation
        assert error.value.status_code == 403
    req.app.state.access_control_service.snapshot.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("app_id", ["knowledge_retrieval", "media_studio"])
async def test_other_app_admin_cannot_manage_aiops_notifications(app_id):
    req = request(app_id=app_id)
    with pytest.raises(HTTPException) as error:
        await external_context(req, permission="read")
    assert error.value.status_code == 403
    req.app.state.access_control_service.snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_user_member_or_app_has_no_current_access():
    req = request(permissions=())
    with pytest.raises(HTTPException) as error:
        await external_context(req, permission="read")
    assert error.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [True, False])
async def test_personal_settings_only_require_active_user(active):
    req = request(roles=("app_user",))
    req.app.state.external_notification_service.authority.actor_active.return_value = active
    if active:
        scope, actor = await external_context(req, permission="self")
        assert scope.domain_id is None and actor == "fixture-user"
    else:
        with pytest.raises(HTTPException):
            await external_context(req, permission="self")
    req.app.state.access_control_service.snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_permission_is_denied_instead_of_treated_as_personal_access():
    with pytest.raises(HTTPException):
        await external_context(request(), permission="unknown")


@pytest.mark.parametrize("role", ["app_admin", "app_user", "app_manager", "platform_admin", None])
@pytest.mark.parametrize("method,path,body,service_method", [
    ("GET", "/settings", None, "get_settings"),
    ("GET", "/bindings", None, "list"),
    ("GET", "/jobs", None, "list_jobs"),
    ("POST", "/jobs/00000000-0000-0000-0000-000000000001/retry",
     {"expected_row_version": 1}, "retry"),
])
def test_http_governance_uses_real_app_authorization(role, method, path, body, service_method):
    req = request(roles=(role,) if role else (), token_roles=("app_admin",))
    service = NS(get_settings=AsyncMock(return_value={}), list=AsyncMock(return_value=[]),
        list_jobs=AsyncMock(return_value={"items": []}), retry=AsyncMock(return_value={}))
    app = FastAPI()
    app.state.access_control_service = req.app.state.access_control_service
    app.state.external_notification_service = service

    @app.middleware("http")
    async def trusted_session(incoming, call_next):
        # 只替换受信会话入口；路由、角色快照和统一 App 鉴权均执行正式实现。
        incoming.state.auth_context = req.state.auth_context
        return await call_next(incoming)

    app.include_router(router)
    with TestClient(app) as client:
        capabilities = client.get("/api/v1/notification-delivery/capabilities")
        assert capabilities.status_code == 200
        assert capabilities.json()["capabilities"] == (
            ["channel", "binding", "read", "retry"] if role == "app_admin" else [])
        response = client.request(method, "/api/v1/notification-delivery" + path,
            **({"json": body} if body is not None else {}))
    assert response.status_code == (200 if role == "app_admin" else 403)
    if role == "app_admin":
        operation = getattr(service, service_method)
        operation.assert_awaited_once()
        scope = operation.await_args.kwargs["scope"]
        assert scope.tenant_id is None and scope.domain_id is None
    else:
        for operation in vars(service).values():
            operation.assert_not_awaited()
