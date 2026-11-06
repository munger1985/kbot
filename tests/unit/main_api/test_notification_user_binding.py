"""用户可选绑定与渠道 owner 范围回归；只使用隔离 SQLite，不访问部署数据库。"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import DateTime, JSON, MetaData, Numeric, Uuid
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from main_api.application.notification_user_binding import personal_binding, set_personal_binding
from main_api.application.notification_delivery import ExternalDeliveryError
from main_api.repositories.notification_delivery import ExternalDeliveryRepository
from platform_core.notifications.delivery import ConnectionCreate, DeliveryScope
from platform_core.persistence.notification_delivery import define_external_tables


@pytest_asyncio.fixture(params=[False, True], ids=["global", "tenant"])
async def owner_database(request, tmp_path):
    tenant = uuid4() if request.param else None
    tables = define_external_tables(metadata=MetaData(), prefix="test_", id_type=Uuid(),
        domain_type=Uuid() if tenant else Numeric(38, 0), tenant_type=Uuid() if tenant else None,
        json_type=JSON(), timestamp_type=DateTime(timezone=True))
    engine = create_async_engine("sqlite+aiosqlite:///" + str(tmp_path / "users.db"))
    async with engine.begin() as connection:
        await connection.run_sync(tables["connection"].metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory, tables, DeliveryScope(tenant_id=tenant)
    await engine.dispose()


def repo_for(session, tables):
    repo = ExternalDeliveryRepository(session)
    repo.tables = tables
    return repo


async def connection(repo, scope):
    return await repo.add("connection", scope=scope, actor_id="admin", now=datetime.now(timezone.utc),
        values={"id": uuid4(), "name": "tenant channel", "provider": "FEISHU", "app_id": "cli_fixture",
            "credential_ref": "opaque", "credential_version": 1, "enabled": True, "health": "UNCHECKED"})


@pytest.mark.asyncio
async def test_optional_empty_create_does_not_require_channel_or_external_tables():
    repo = Mock()
    assert await set_personal_binding(repo, scope=DeliveryScope(), user_id="user", actor_id="admin",
        open_id=None, creating=True) is None
    assert not repo.mock_calls


@pytest.mark.asyncio
async def test_binding_reused_across_domains_and_separate_tenants(owner_database):
    factory, tables, scope = owner_database
    async with factory() as session:
        repo = repo_for(session, tables)
        conn = await connection(repo, scope)
        created = await set_personal_binding(repo, scope=scope, user_id="user", actor_id="admin",
            open_id="ou_fixture", creating=True)
        assert created["status"] == "UNVERIFIED"
        for domain in ([str(uuid4()), str(uuid4())] if scope.tenant_id else ["12", "99"]):
            resource_scope = scope.model_copy(update={"domain_id": domain})
            assert (await repo.get("connection", scope=resource_scope, object_id=conn["id"]))["id"] == conn["id"]
            assert (await personal_binding(repo, scope=resource_scope, user_id="user"))["id"] == created["id"]
        if scope.tenant_id:
            other = DeliveryScope(tenant_id=uuid4())
            assert await repo.get("connection", scope=other, object_id=conn["id"]) is None
            assert await personal_binding(repo, scope=other, user_id="user") is None
        assert all("domain_id" not in tables[k].c for k in ("connection","binding","policy","audit"))
        assert not tables["entry"].c.domain_id.nullable
        assert tables["job"].c.domain_id.nullable


@pytest.mark.asyncio
async def test_rebind_and_unbind_preserve_history_and_versions(owner_database):
    factory, tables, scope = owner_database
    async with factory() as session:
        repo = repo_for(session, tables)
        await connection(repo, scope)
        old = await set_personal_binding(repo, scope=scope, user_id="user", actor_id="admin",
            open_id="ou_first", creating=True)
        with pytest.raises(ExternalDeliveryError, match="VERSION_CONFLICT"):
            await set_personal_binding(repo, scope=scope, user_id="user", actor_id="admin", open_id=None)
        new = await set_personal_binding(repo, scope=scope, user_id="user", actor_id="admin", open_id="ou_second",
            expected_binding_id=old["id"], expected_version=old["row_version"])
        assert new["id"] != old["id"] and new["status"] == "UNVERIFIED"
        previous = await repo.get("binding", scope=scope, object_id=UUID(old["id"]))
        assert previous["status"] == "DISABLED" and previous["row_version"] > old["row_version"]
        assert await set_personal_binding(repo, scope=scope, user_id="user", actor_id="admin", open_id=None,
            expected_binding_id=new["id"], expected_version=new["row_version"]) is None
        assert await personal_binding(repo, scope=scope, user_id="user") is None
        assert len(await repo.list("binding", scope=scope)) == 2


@pytest.mark.asyncio
async def test_same_feishu_identity_cannot_belong_to_two_users(owner_database):
    factory, tables, scope = owner_database
    async with factory() as session:
        repo = repo_for(session, tables)
        await connection(repo, scope)
        await set_personal_binding(repo, scope=scope, user_id="first", actor_id="admin",
            open_id="ou_shared", creating=True)
        with pytest.raises(ExternalDeliveryError, match="BINDING_CONFLICT"):
            await set_personal_binding(repo, scope=scope, user_id="second", actor_id="admin",
                open_id="ou_shared", creating=True)


def test_wecom_is_not_an_enabled_provider():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ConnectionCreate(name="not available", provider="WECOM", app_id="cli_fixture",
            app_secret="fixture", enabled=True)
