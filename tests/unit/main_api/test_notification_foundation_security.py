"""通知管理与关注入口必须拒绝横向提权。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from main_api.api.notifications import router
from main_api.application.access_control import AccessDeniedError
from main_api.application.notification_center import NotificationCenterError, NotificationCenterService
from platform_core.identity import uuid7


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/v1/notifications/quarantine"),
    ("POST", "/api/v1/notifications/quarantine/019ffff0-0000-7000-8000-000000000013/retry"),
])
def test_quarantine_requires_notification_management_permission(method, path):
    app = FastAPI()
    app.state.access_control_service = SimpleNamespace(require_platform=AsyncMock(
        side_effect=AccessDeniedError("platform:notification_manage")))
    app.state.notification_center_service = SimpleNamespace(quarantine=AsyncMock(), retry_quarantined=AsyncMock())

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        request.state.auth_context = SimpleNamespace(asserted_user_id="reader", domain_id="73",
            principal_kind="PORTAL")
        return await call_next(request)

    app.include_router(router)
    with TestClient(app) as client:
        assert client.request(method, path).status_code == 403
    app.state.notification_center_service.quarantine.assert_not_awaited()
    app.state.notification_center_service.retry_quarantined.assert_not_awaited()


def test_watching_another_users_operation_does_not_grant_visibility():
    class Uow:
        notifications = SimpleNamespace(get_operation=AsyncMock(return_value=SimpleNamespace(
            initiator_actor_id="other")), watch=AsyncMock())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    async def run():
        with pytest.raises(NotificationCenterError) as caught:
            await NotificationCenterService(uow_factory=Uow).watch_operation(
                operation_id=uuid7(), domain_id=73, actor_id="reader", notify_terminal=True)
        assert caught.value.status_code == 404
        Uow.notifications.watch.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("kind", ["SERVICE", "API_CLIENT"])
def test_management_rejects_nonhuman_principal_even_if_permission_would_be_granted(kind):
    app = FastAPI()
    permission = AsyncMock()
    quarantine = AsyncMock()
    app.state.access_control_service = SimpleNamespace(require_platform=permission)
    app.state.notification_center_service = SimpleNamespace(quarantine=quarantine)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        request.state.auth_context = SimpleNamespace(asserted_user_id="manager", domain_id="73",
            principal_kind=kind)
        return await call_next(request)

    app.include_router(router)
    with TestClient(app) as client:
        assert client.get("/api/v1/notifications/quarantine").status_code == 403
    permission.assert_not_awaited()
    quarantine.assert_not_awaited()


def test_quarantine_exposes_metadata_not_envelope_or_recipient_list(monkeypatch):
    async def run():
        serialized = {"outbox_id": str(uuid7()), "event_type": "aiops.proposal.review_required",
            "domain_id": 73, "last_error_code": "NOTIFICATION_APPROVER_UNAVAILABLE",
            "payload_json": {"summary": "业务正文"}, "recipient_actor_ids": ["reviewer"]}
        repository = SimpleNamespace(quarantine=AsyncMock(return_value=[object()]))
        context = AsyncMock()
        context.__aenter__.return_value = SimpleNamespace(notifications=repository)
        monkeypatch.setattr("main_api.application.notification_center.serialize_entity", lambda row: serialized)
        result = await NotificationCenterService(uow_factory=lambda: context).quarantine(domain_id=73, limit=10)
        assert result == [{key: value for key, value in serialized.items()
            if key not in {"payload_json", "recipient_actor_ids"}}]
    asyncio.run(run())


@pytest.mark.parametrize("audit_fails", [False, True])
def test_retry_records_real_manager_in_same_transaction_and_never_commits_without_audit(audit_fails):
    async def run():
        outbox_id = uuid7()
        retry = AsyncMock(return_value=True)
        audit = AsyncMock(side_effect=RuntimeError("审计不可用") if audit_fails else None)
        commit = AsyncMock()
        context = AsyncMock()
        context.__aenter__.return_value = SimpleNamespace(
            notifications=SimpleNamespace(retry_quarantined=retry),
            access=SimpleNamespace(add_access_audit_event=audit), commit=commit)
        service = NotificationCenterService(uow_factory=lambda: context)
        if audit_fails:
            with pytest.raises(RuntimeError, match="审计不可用"):
                await service.retry_quarantined(domain_id=73, outbox_id=outbox_id, actor_id="manager")
            commit.assert_not_awaited()
            assert context.__aexit__.await_args.args[0] is RuntimeError
        else:
            await service.retry_quarantined(domain_id=73, outbox_id=outbox_id, actor_id="manager")
            commit.assert_awaited_once()
        retry.assert_awaited_once_with(domain_id=73, outbox_id=outbox_id)
        event = audit.await_args.args[0]
        assert (event.actor_id, event.action, event.resource_id) == ("manager", "NOTIFICATION_RETRY", str(outbox_id))
    asyncio.run(run())
