"""投递列表分页、时间合同及租户／资源范围回归。"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Column, DateTime, MetaData, String, Table, Uuid, insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from main_api.api.notification_delivery import router
from main_api.application.notification_delivery import ExternalNotificationService
from main_api.application.notification_delivery import ExternalDeliveryError
from platform_core.notifications.external_config import ExternalDeliveryConfig
from main_api.repositories.notification_delivery import ExternalDeliveryRepository
from platform_core.notifications.delivery import DeliveryScope


@pytest.mark.asyncio
@pytest.mark.parametrize("count,has_more", [(0, False), (20, False), (21, True)])
async def test_page_uses_lookahead_and_public_projection(count, has_more):
    repo = SimpleNamespace(list=AsyncMock(return_value=[
        {"id": uuid4(), "status": "ACCEPTED", "content": {"private": "hidden"}, "credential_ref": "hidden"}
        for _ in range(count)
    ]))

    @asynccontextmanager
    async def factory():
        yield SimpleNamespace(external_delivery=repo)

    service = ExternalNotificationService(uow_factory=factory, config=None, vault=None, sender=None, authority=None)
    scope = DeliveryScope()
    result = await service.list_jobs(scope=scope, limit=20, offset=40, event_type="connection.test")
    assert result["has_more"] is has_more
    assert len(result["items"]) == min(count, 20)
    assert result["limit"] == 20 and result["offset"] == 40
    assert all("content" not in row and "credential_ref" not in row for row in result["items"])
    repo.list.assert_awaited_once_with("job", scope=scope, limit=21, offset=40, event_type="connection.test")


@pytest.fixture
def http(monkeypatch):
    service = SimpleNamespace(list_jobs=AsyncMock(return_value={"items": [], "has_more": False, "limit": 20, "offset": 0}))
    context = AsyncMock(return_value=(DeliveryScope(), "admin"))
    monkeypatch.setattr("main_api.api.notification_delivery.external_context", context)
    app = FastAPI()
    app.state.external_notification_service = service
    app.include_router(router)
    return TestClient(app), service, context


def test_http_defaults_and_no_domain_partition(http):
    client, service, context = http
    assert client.get("/api/v1/notification-delivery/jobs").json()["has_more"] is False
    context.assert_awaited_once()
    assert context.await_args.kwargs == {"permission": "read"}
    assert service.list_jobs.await_args.kwargs["scope"].domain_id is None
    assert service.list_jobs.await_args.kwargs["limit"] == 20


def test_capabilities_only_report_permissions_without_deployment_gate(http, monkeypatch):
    client, service, _ = http
    monkeypatch.setattr("main_api.api.notification_delivery.external_capabilities",
        AsyncMock(return_value={"capabilities": ["channel"]}))
    response = client.get("/api/v1/notification-delivery/capabilities")
    assert response.status_code == 200
    assert response.json() == {"capabilities": ["channel"]}


@pytest.fixture
def channel_configuration():
    row = {"id": uuid4(), "name": "旧名称", "provider": "FEISHU", "app_id": "cli_fixture",
        "credential_ref": "managed://old", "credential_version": 1, "row_version": 2,
        "enabled": True, "health": "UNCHECKED"}
    repo = SimpleNamespace(lock_configuration=AsyncMock(),
        connection_for_provider=AsyncMock(return_value=None),
        add=AsyncMock(side_effect=lambda kind, **kwargs: {**kwargs["values"], "row_version": 1}),
        get=AsyncMock(return_value=row), change=AsyncMock(return_value=True), audit=AsyncMock(),
        claim=AsyncMock())
    commit = AsyncMock()

    @asynccontextmanager
    async def factory():
        yield SimpleNamespace(external_delivery=repo, commit=commit)

    vault = SimpleNamespace(write=AsyncMock(return_value="managed://new"), read=AsyncMock(), delete=AsyncMock())
    from platform_core.notifications.delivery import SendResult, DeliveryStatus
    sender = SimpleNamespace(send=AsyncMock(), check=AsyncMock(return_value=SendResult(DeliveryStatus.ACCEPTED)))
    vault.read.return_value = {"app_secret": "fixture-only-secret"}
    service = ExternalNotificationService(uow_factory=factory, config=ExternalDeliveryConfig(),
        vault=vault, sender=sender, authority=None)
    return service, repo, vault, sender, commit, row


@pytest.mark.parametrize("channel_enabled", [False, True])
def test_configuration_creation_needs_no_portal_or_network(http, channel_configuration, channel_enabled):
    client, adapter, _ = http
    service, repo, vault, sender, commit, _ = channel_configuration
    adapter.create_connection = service.create_connection
    response = client.post("/api/v1/notification-delivery/connections",
        json={"name": "测试连接", "app_id": "cli_fixture", "app_secret": "fixture-only-secret",
            "enabled": channel_enabled})
    assert response.status_code == 201
    result = response.json()
    assert result["enabled"] is channel_enabled
    assert result["health"] == "UNCHECKED"
    assert not {"app_secret", "credential_ref", "credential_version"} & result.keys()
    vault.write.assert_awaited_once()
    assert repo.audit.await_args.kwargs["action"] == "connection.create"
    commit.assert_awaited_once()
    assert "enabled" not in type(service.config).model_fields and "public_origin" not in type(service.config).model_fields
    sender.send.assert_not_awaited()
    sender.check.assert_not_awaited()
    repo.claim.assert_not_awaited()


@pytest.mark.parametrize("rotate_secret", [False, True])
def test_configuration_update_and_rotation_need_no_portal(http, channel_configuration, rotate_secret):
    client, adapter, _ = http
    service, repo, vault, sender, commit, row = channel_configuration
    adapter.update_connection = service.update_connection
    response = client.put(f"/api/v1/notification-delivery/connections/{row['id']}",
        json={"name": "新名称", "enabled": False, "expected_row_version": 2,
            **({"app_secret": "rotated-fixture-secret"} if rotate_secret else {})})
    assert response.status_code == 200
    result = response.json()
    assert result["name"] == "新名称" and result["enabled"] is False and result["row_version"] == 3
    assert not {"app_secret", "credential_ref", "credential_version"} & result.keys()
    assert vault.write.await_count == int(rotate_secret)
    assert repo.change.await_args.kwargs["expected_version"] == 2
    assert repo.audit.await_args.kwargs["action"] == "connection.update"
    commit.assert_awaited_once()
    assert "enabled" not in type(service.config).model_fields
    sender.send.assert_not_awaited()
    sender.check.assert_not_awaited()


@pytest.mark.asyncio
async def test_connection_check_needs_no_portal_or_deployment_switch(channel_configuration):
    service, repo, vault, sender, _, row = channel_configuration
    result = await service.check_connection(scope=DeliveryScope(), actor_id="admin", object_id=row["id"])
    assert result["health"] == "TOKEN_VALID"
    vault.read.assert_awaited_once()
    sender.check.assert_awaited_once()
    sender.send.assert_not_awaited()



def test_configuration_preserves_connection_singleton_and_credential_cleanup(http, channel_configuration):
    client, adapter, _ = http
    service, repo, vault, sender, commit, row = channel_configuration
    repo.connection_for_provider.return_value = row
    adapter.create_connection = service.create_connection
    response = client.post("/api/v1/notification-delivery/connections",
        json={"name": "重复连接", "app_id": "cli_fixture", "app_secret": "fixture-only-secret"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "NOTIFICATION_CONNECTION_ALREADY_CONFIGURED"
    repo.add.assert_not_awaited()
    commit.assert_not_awaited()
    vault.delete.assert_awaited_once()
    sender.send.assert_not_awaited()


@pytest.mark.parametrize("rotate_secret", [False, True])
def test_configuration_preserves_optimistic_version_control(http, channel_configuration, rotate_secret):
    client, adapter, _ = http
    service, repo, vault, sender, commit, row = channel_configuration
    repo.change.return_value = False
    adapter.update_connection = service.update_connection
    response = client.put(f"/api/v1/notification-delivery/connections/{row['id']}",
        json={"name": "冲突配置", "enabled": False, "expected_row_version": 2,
            **({"app_secret": "rotated-fixture-secret"} if rotate_secret else {})})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "NOTIFICATION_VERSION_CONFLICT"
    commit.assert_not_awaited()
    assert vault.delete.await_count == int(rotate_secret)
    sender.send.assert_not_awaited()


@pytest.mark.parametrize("query", [
    "limit=0", "limit=101", "offset=-1", "status=INVALID", "connection_id=wrong",
    "created_from=2026-11-01T00:00:00",
    "created_from=2026-11-02T00:00:00Z&created_to=2026-11-01T00:00:00Z",
    "created_from=2026-11-01T00:00:00Z&created_to=2026-11-01T00:00:00Z",
])
def test_http_rejects_invalid_page_or_time_range(http, query):
    client, service, _ = http
    assert client.get("/api/v1/notification-delivery/jobs?" + query).status_code == 422
    service.list_jobs.assert_not_awaited()


def test_http_forwards_valid_filters(http):
    client, service, _ = http
    connection_id = uuid4()
    response = client.get("/api/v1/notification-delivery/jobs", params={
        "limit": 50, "offset": 50, "status": "UNKNOWN", "event_type": "connection.test",
        "connection_id": str(connection_id), "recipient_actor_id": "user",
        "created_from": "2026-11-01T00:00:00Z", "created_to": "2026-11-02T00:00:00Z",
    })
    assert response.status_code == 200
    filters = service.list_jobs.await_args.kwargs
    assert filters["connection_id"] == connection_id and filters["recipient_actor_id"] == "user"
    assert filters["created_from"].tzinfo is not None


@pytest.mark.asyncio
async def test_repository_combines_filters_and_keeps_tenant_scope(tmp_path):
    metadata = MetaData()
    connection = Table("test_connection", metadata, Column("id", Uuid(), primary_key=True), Column("tenant_id", Uuid()))
    job = Table("test_job", metadata, Column("id", Uuid(), primary_key=True), Column("tenant_id", Uuid()),
        Column("domain_id", Uuid()), Column("connection_id", Uuid()), Column("event_type", String()),
        Column("recipient_actor_id", String()), Column("status", String()), Column("created_at", DateTime(timezone=True)))
    engine = create_async_engine("sqlite+aiosqlite:///" + str(tmp_path / "pages.db"))
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    tenant_id, other_tenant, domain_a, domain_b, connection_id = [uuid4() for _ in range(5)]
    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(7):
        rows.append({"id": uuid4(), "tenant_id": other_tenant if index == 6 else tenant_id,
            "domain_id": domain_a if index % 2 == 0 else domain_b,
            "connection_id": connection_id, "event_type": "connection.test",
            "recipient_actor_id": "user", "status": "UNKNOWN", "created_at": start + timedelta(hours=index)})
    rows[4]["status"] = "ACCEPTED"
    rows[5]["event_type"] = "identity.verification"
    factory = async_sessionmaker(engine)
    try:
        async with factory() as session:
            await session.execute(insert(job), rows)
            await session.commit()
            repo = ExternalDeliveryRepository(session)
            repo.tables = {"connection": connection, "job": job}
            filters = dict(connection_id=connection_id, event_type="connection.test", recipient_actor_id="user",
                status="UNKNOWN", created_from=start + timedelta(hours=1), created_to=start + timedelta(hours=4))
            result = await repo.list("job", scope=DeliveryScope(tenant_id=tenant_id), **filters)
            assert [row["id"] for row in result] == [rows[3]["id"], rows[2]["id"], rows[1]["id"]]
            assert len(await repo.list("job", scope=DeliveryScope(tenant_id=tenant_id), **filters, limit=2)) == 2
            assert len(await repo.list("job", scope=DeliveryScope(tenant_id=tenant_id), **filters, offset=2)) == 1
            restricted = await repo.list("job", scope=DeliveryScope(tenant_id=tenant_id, domain_id=str(domain_a)), **filters)
            assert [row["id"] for row in restricted] == [rows[2]["id"]]
    finally:
        await engine.dispose()
