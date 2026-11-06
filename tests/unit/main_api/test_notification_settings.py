"""通知配置持久化、热读取、无门户试发和部署网关的回归。"""

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from datetime import datetime, timezone

import tomllib

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from pydantic import ValidationError
from sqlalchemy import DateTime, JSON, MetaData, Numeric, Uuid, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.types import TypeDecorator

from main_api.api.notification_delivery import router
from main_api.application.notification_delivery import ExternalNotificationService, ExternalDeliveryError
from main_api.repositories.notification_delivery import ExternalDeliveryRepository
from platform_core.notifications.delivery import (ConnectionCreate, DeliveryScope, DeliveryStatus,
    ExternalPolicyCreate, NotificationSettingsUpdate, SendResult, notification_settings_id)
from platform_core.notifications.external_config import ExternalDeliveryConfig
from platform_core.notifications.feishu import FeishuSender
from platform_core.persistence.notification_delivery import define_external_tables

ROOT = Path(__file__).resolve().parents[3]
ORACLE = (ROOT / "database/oracle/main_api").exists()


class UTCDateTime(TypeDecorator):
    impl = DateTime
    cache_ok = True

    def process_result_value(self, value, dialect):
        return value.replace(tzinfo=timezone.utc) if value and value.tzinfo is None else value


@pytest_asyncio.fixture
async def configured(tmp_path):
    tables = define_external_tables(metadata=MetaData(), prefix="notify_", id_type=Uuid(),
        domain_type=Numeric(38, 0), json_type=JSON(), timestamp_type=UTCDateTime(),
        tenant_type=None if ORACLE else Uuid())
    engine = create_async_engine("sqlite+aiosqlite:///" + str(tmp_path / "settings.db"))
    async with engine.begin() as conn:
        await conn.run_sync(tables["settings"].metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def uow():
        async with factory() as session:
            repo = ExternalDeliveryRepository(session)
            repo.tables = tables
            yield SimpleNamespace(external_delivery=repo, commit=session.commit)

    async with factory() as session:
        await session.execute(tables["throttle"].insert().values(id="external", next_send_at=datetime.now(timezone.utc)))
        await session.commit()
    service = ExternalNotificationService(uow_factory=uow, config=ExternalDeliveryConfig(),
        vault=SimpleNamespace(write=AsyncMock(return_value="managed://fixture"),
            read=AsyncMock(return_value={"app_secret": "fixture-only"}), delete=AsyncMock()),
        sender=SimpleNamespace(check=AsyncMock(return_value=SendResult(DeliveryStatus.ACCEPTED)), send=AsyncMock()),
        authority=SimpleNamespace(actor_active=AsyncMock(return_value=True)))
    scope = DeliveryScope() if ORACLE else DeliveryScope(tenant_id=uuid4())
    connection = await service.create_connection(scope=scope, actor_id="admin", body=ConnectionCreate(
        name="飞书", app_id="cli_fixture", app_secret="fixture-only", enabled=True))
    binding_id = uuid4()
    async with uow() as work:
        await work.external_delivery.add("binding", scope=scope, actor_id="admin",
            now=datetime.now(timezone.utc), values={"id":binding_id,"connection_id":connection["id"],
                "actor_id":"member","target_type":"open_id","external_id":"ou_fixture",
                "status":"VERIFIED","challenge_attempts":0})
        await work.commit()
    yield service, scope, connection, binding_id, uow, tables
    await engine.dispose()


@pytest.mark.asyncio
async def test_settings_persist_versions_isolate_tenant_and_reload_without_restart(configured):
    service, scope, _, _, uow, tables = configured
    assert await service.get_settings(scope=scope, actor_id="admin") == {"public_origin":"", "row_version":0}
    for version, origin in enumerate(("https://first.example", "https://second.example"), 1):
        result = await service.update_settings(scope=scope, actor_id="admin",
            body=NotificationSettingsUpdate(public_origin=origin + "/", expected_row_version=version - 1))
        assert result == {"public_origin":origin, "row_version":version}
        assert await service.get_settings(scope=scope, actor_id="admin") == result
    with pytest.raises(ExternalDeliveryError, match="NOTIFICATION_VERSION_CONFLICT"):
        await service.update_settings(scope=scope, actor_id="admin",
            body=NotificationSettingsUpdate(public_origin="", expected_row_version=1))
    same_scope = scope.model_copy(update={"domain_id":"12" if ORACLE else str(uuid4())})
    assert (await service.get_settings(scope=same_scope, actor_id="admin"))["row_version"] == 2
    assert notification_settings_id(scope) == notification_settings_id(same_scope)
    if not ORACLE:
        other = DeliveryScope(tenant_id=uuid4())
        assert (await service.get_settings(scope=other, actor_id="admin"))["row_version"] == 0
        await service.update_settings(scope=other, actor_id="admin",
            body=NotificationSettingsUpdate(public_origin="https://other.example", expected_row_version=0))
        assert (await service.get_settings(scope=scope, actor_id="admin"))["public_origin"] == "https://second.example"
    async with uow() as work:
        audits = await work.external_delivery.list("audit", scope=scope)
        assert sum(item["action"] == "settings.update" for item in audits) == 2


@pytest.mark.asyncio
async def test_check_and_pure_test_need_no_portal_or_restart(configured):
    service, scope, connection, binding_id, uow, tables = configured
    service.sender.send.assert_not_awaited()
    assert (await service.check_connection(scope=scope, actor_id="admin", object_id=connection["id"]))["health"] == "TOKEN_VALID"
    result = await service.queue_test(scope=scope, actor_id="admin", object_id=connection["id"], binding_id=binding_id)
    async with uow() as work:
        job = await work.external_delivery.get("job", scope=scope, object_id=result["job_id"])
        assert job["purpose"] == "TEST" and job["domain_id"] is None and job["entry_id"] is None
    service.sender.send.assert_not_awaited()
    async with uow() as work:
        await work.external_delivery.change("connection", scope=scope, object_id=connection["id"],
            expected_version=2, values={"enabled":False}, actor_id="admin", now=datetime.now(timezone.utc))
        await work.commit()
    with pytest.raises(ExternalDeliveryError, match="NOTIFICATION_TEST_TARGET_INVALID"):
        await service.queue_test(scope=scope, actor_id="admin", object_id=connection["id"], binding_id=binding_id)


@pytest.mark.asyncio
async def test_business_rule_requires_portal_and_active_rule_prevents_clearing(configured):
    service, scope, connection, binding_id, uow, _ = configured
    body = ExternalPolicyCreate(connection_id=connection["id"], binding_ids=[binding_id],
        event_types=["aiops.situation.detected"], enabled=True)
    with pytest.raises(ExternalDeliveryError, match="NOTIFICATION_PUBLIC_ORIGIN_REQUIRED"):
        await service.create_policy(scope=scope, actor_id="admin", body=body)
    await service.update_settings(scope=scope, actor_id="admin",
        body=NotificationSettingsUpdate(public_origin="https://portal.example", expected_row_version=0))
    rule = await service.create_policy(scope=scope, actor_id="admin", body=body)
    with pytest.raises(ExternalDeliveryError, match="NOTIFICATION_PUBLIC_ORIGIN_IN_USE"):
        await service.update_settings(scope=scope, actor_id="admin",
            body=NotificationSettingsUpdate(public_origin="", expected_row_version=1))
    await service.set_policy_enabled(scope=scope, actor_id="admin", object_id=rule["id"], expected_version=1, enabled=False)
    assert (await service.update_settings(scope=scope, actor_id="admin",
        body=NotificationSettingsUpdate(public_origin="", expected_row_version=1)))["row_version"] == 2


@pytest.mark.asyncio
async def test_http_settings_use_channel_permission_and_reject_endpoint_injection(configured):
    service, scope, _, _, _, _ = configured
    app = FastAPI()
    app.include_router(router)
    app.state.external_notification_service = service
    identity = AsyncMock(return_value=(scope, "admin"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        with patch("main_api.api.notification_delivery.external_context", identity):
            assert (await client.get("/api/v1/notification-delivery/settings")).json()["row_version"] == 0
            assert identity.await_args.kwargs == {"permission":"channel"}
            response = await client.put("/api/v1/notification-delivery/settings",
                json={"public_origin":"https://portal.example", "expected_row_version":0})
            assert response.status_code == 200
            response = await client.put("/api/v1/notification-delivery/settings",
                json={"public_origin":"", "expected_row_version":0})
            assert response.status_code == 409
            assert (await client.put("/api/v1/notification-delivery/settings",
                json={"public_origin":"", "expected_row_version":1, "token_url":"https://untrusted.example"})).status_code == 422


@pytest.mark.parametrize("origin", ["http://portal.example", "https://user:secret@portal.example",
    "https://portal.example/path", "https://portal.example?x=1", "https://portal.example#x",
    "https://portal.example:", "https://portal.example:99999", "https://portal .example"])
def test_database_origin_contract_rejects_unsafe_values(origin):
    with pytest.raises(ValidationError):
        NotificationSettingsUpdate(public_origin=origin, expected_row_version=0)


def test_deployment_template_has_only_technical_notification_config():
    filename = "kbot.toml.example" if ORACLE else "ammolite.toml.example"
    values = tomllib.loads((ROOT / "configuration" / filename).read_text())["external_delivery"]
    assert "enabled" not in values and "public_origin" not in values
    config = ExternalDeliveryConfig(**values)
    config.validate_runtime(production=True)
    assert config.token_url == "https://accounts.feishu.cn/oauth/v3/token"
    assert config.message_origin == "https://open.feishu.cn"


@pytest.mark.asyncio
@pytest.mark.parametrize("redirect", [None, "token", "message"])
async def test_deployment_gateway_routes_both_endpoints_and_never_follows_redirects(redirect):
    requests = []
    def transport(request):
        requests.append(request)
        token = request.url.host == "oauth.gateway.internal"
        if (token and redirect == "token") or (not token and redirect == "message"):
            return httpx.Response(307, headers={"Location":"https://unexpected.example/steal"})
        if token:
            return httpx.Response(200, json={"access_token":"fixture-token","expires_in":3600,"token_type":"Bearer"})
        return httpx.Response(200, json={"code":0,"data":{"message_id":"om_gateway"}})
    config = ExternalDeliveryConfig(token_url="https://oauth.gateway.internal/proxy/oauth/v3/token",
        message_origin="https://message.gateway.internal")
    config.validate_runtime(production=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport), follow_redirects=True) as client:
        result = await FeishuSender(config=config, client=client).send(
            connection={"id":"fixture","app_id":"cli_fixture","credential_version":1},
            secret="fixture-only", target_type="open_id", external_id="ou_fixture",
            content={}, provider_uuid="fixture")
    assert all(request.url.host in {"oauth.gateway.internal","message.gateway.internal"} for request in requests)
    assert (result.status == DeliveryStatus.ACCEPTED) == (redirect is None)
    if redirect is None:
        assert requests[0].url.path == "/proxy/oauth/v3/token"
        assert requests[1].url.path == "/open-apis/im/v1/messages"
