"""在隔离 SQLite 文件上验证真实事务竞争、主键幂等和旧租约写回。"""

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import DateTime, JSON, MetaData, Numeric, Uuid, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.types import TypeDecorator

from main_api.repositories.notification_delivery import ExternalDeliveryRepository
from platform_core.persistence.notification_delivery import define_external_tables
from platform_core.notifications.delivery import notification_settings_id, DeliveryScope, DeliveryStatus as Status, SendResult, payload_hash, stable_id
from platform_core.notifications.external_config import ExternalDeliveryConfig
from platform_core.notifications.external_projection import enqueue_external_event


class UTCTimestamp(TypeDecorator):
    impl = DateTime
    cache_ok = True
    def process_result_value(self, value, dialect):
        return value.replace(tzinfo=timezone.utc) if value and value.tzinfo is None else value


@pytest_asyncio.fixture
async def database(tmp_path):
    metadata = MetaData()
    tables = define_external_tables(metadata=metadata, prefix="notify_", id_type=Uuid(),
        domain_type=Numeric(38, 0), json_type=JSON(), timestamp_type=UTCTimestamp())
    engine = create_async_engine("sqlite+aiosqlite:///" + str(tmp_path / "delivery.db"))
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory, tables
    await engine.dispose()


def repository(session, tables):
    repo = ExternalDeliveryRepository(session)
    repo.tables = tables
    return repo


async def seed(factory, tables, *, network=False):
    scope, now = DeliveryScope(domain_id="12"), datetime.now(timezone.utc)
    connection_id, binding_id, job_id = uuid4(), uuid4(), uuid4()
    async with factory() as session:
        repo = repository(session, tables)
        await repo.add("connection", scope=scope, actor_id="admin", now=now, values={
            "id": connection_id, "name": "test", "provider": "FEISHU", "app_id": "cli_test",
            "credential_ref": "opaque", "credential_version": 1, "enabled": True, "health": "UNCHECKED"})
        await repo.add("binding", scope=scope, actor_id="admin", now=now, values={
            "id": binding_id, "connection_id": connection_id, "actor_id": "B", "target_type": "open_id",
            "external_id": "ou_test", "status": "VERIFIED", "challenge_attempts": 0})
        await repo.add("job", scope=scope, actor_id="admin", now=now, values={
            "id": job_id, "source_event_id": uuid4(), "event_type": "aiops.execution.succeeded",
            "connection_id": connection_id, "connection_version": 1, "binding_id": binding_id,
            "binding_version": 1, "recipient_actor_id": "B", "target_type": "open_id", "external_id": "ou_test",
            "purpose": "BUSINESS", "template_version": 1, "generation": 0, "content": {}, "content_hash": payload_hash({}),
            "provider_uuid": str(job_id), "status": Status.PENDING, "attempt_count": 0, "retry_count": 0,
            "available_at": now, "expires_at": now + timedelta(hours=1)})
        await session.commit()
    return scope, job_id, now


@pytest.mark.asyncio
async def test_parallel_workers_only_one_claim(database):
    factory, tables = database
    scope, job_id, now = await seed(factory, tables)
    async def claim():
        async with factory() as session:
            result = await repository(session, tables).claim(now=now, lease_seconds=60, dedupe_seconds=3300)
            await session.commit()
            return result
    results = await asyncio.gather(claim(), claim())
    assert sum(result is not None for result in results) == 1


@pytest.mark.asyncio
async def test_stale_lease_cannot_overwrite_and_crash_is_unknown(database):
    factory, tables = database
    scope, job_id, now = await seed(factory, tables)
    async with factory() as session:
        repo = repository(session, tables)
        job = await repo.claim(now=now, lease_seconds=60, dedupe_seconds=3300)
        assert await repo.mark_network(job=job, now=now)
        await session.commit()
    async with factory() as session:
        repo = repository(session, tables)
        assert await repo.claim(now=now + timedelta(seconds=61), lease_seconds=60, dedupe_seconds=3300) is None
        await session.commit()
    async with factory() as session:
        repo = repository(session, tables)
        assert not await repo.finish(job=job, result=SendResult(Status.ACCEPTED, message_id="om_old"),
            now=now + timedelta(seconds=62), available_at=now)
        row = await repo.get("job", scope=scope, object_id=job_id)
        assert row["status"] == Status.UNKNOWN and row["message_id"] is None
        attempts = await repo.attempts(scope=scope, job_id=job_id)
        assert attempts[0]["status"] == Status.UNKNOWN
        assert await repo.get("job", scope=DeliveryScope(domain_id="99"), object_id=job_id) is None


@pytest.mark.asyncio
async def test_same_primary_key_creation_is_concurrently_idempotent(database):
    factory, tables = database
    scope, job_id, now = await seed(factory, tables)
    object_id = stable_id("event", "connection", "target")
    async def create():
        async with factory() as session:
            result = await repository(session, tables).add_once("entry", scope=scope, actor_id="admin", now=now,
                values={"id": object_id, "recipient_actor_id": "B", "audience": "PERSONAL",
                    "event_type": "aiops.execution.succeeded", "route_key": "aiops.run.detail",
                    "route_params": {"domain_id": "12"}, "expires_at": now + timedelta(days=1)})
            await session.commit()
            return result
    assert sorted(await asyncio.gather(create(), create())) == [False, True]


@pytest.mark.asyncio
async def test_expired_job_cannot_enter_network_even_with_live_lease(database):
    factory, tables = database
    scope, job_id, now = await seed(factory, tables)
    async with factory() as session:
        await session.execute(update(tables["job"]).where(tables["job"].c.id == job_id)
            .values(expires_at=now+timedelta(seconds=2)))
        repo = repository(session, tables)
        job = await repo.claim(now=now, lease_seconds=60, dedupe_seconds=3300)
        assert not await repo.mark_network(job=job, now=now+timedelta(seconds=3))


@pytest.mark.asyncio
async def test_add_once_never_swallows_not_null_even_if_same_primary_key_exists(database):
    factory, tables = database
    scope, job_id, now = await seed(factory, tables)
    async with factory() as session:
        repo = repository(session, tables)
        # 已存在主键不代表缺少必填事实的写入可以被当作幂等成功。
        with pytest.raises(IntegrityError):
            await repo.add_once("job", scope=scope, actor_id="test", now=now, values={"id":job_id})


@pytest.mark.asyncio
async def test_delivery_history_filters_and_pages_preserve_scope(database):
    factory, tables = database
    scope, job_id, now = await seed(factory, tables)
    async with factory() as session:
        repo = repository(session, tables)
        assert len(await repo.list("job", scope=scope, status=Status.PENDING)) == 1
        assert await repo.list("job", scope=scope, status=Status.UNKNOWN) == []
        assert await repo.list("job", scope=scope, offset=1) == []
        assert await repo.list("job", scope=DeliveryScope(domain_id="99")) == []


@pytest.mark.asyncio
async def test_coalescing_only_selects_unsent_jobs_in_current_scope(database):
    factory, tables = database
    scope, job_id, now = await seed(factory, tables)
    async with factory() as session:
        repo = repository(session, tables)
        original = await repo.get("job", scope=scope, object_id=job_id)
        template = {key:value for key,value in original.items()
            if key not in {"created_at","updated_at","created_by","updated_by","row_version","domain_id"}}
        accepted_ids = set()
        for status, sent, other_scope in [
            (Status.PENDING,False,False), (Status.RETRY_WAIT,False,False),
            (Status.RETRY_WAIT,True,False), (Status.SENDING,False,False),
            (Status.UNKNOWN,True,False), (Status.PENDING,False,True),
        ]:
            object_id = uuid4()
            await repo.add("job", scope=DeliveryScope(domain_id="99") if other_scope else scope,
                actor_id="test", now=now, values=template | {"id":object_id,"coalesce_key":"same-window",
                    "status":status,"first_sent_at":now if sent else None})
            if not sent and not other_scope and status in {Status.PENDING,Status.RETRY_WAIT}:
                accepted_ids.add(object_id)
        selected = await repo.coalesce_candidates(scope=scope,coalesce_key="same-window",exclude_id=job_id)
        assert {row["id"] for row in selected} == accepted_ids


@pytest.mark.asyncio
async def test_concurrent_event_coalescing_preserves_both_jobs_and_single_pending(database,monkeypatch):
    factory,tables=database
    scope,job_id,now=await seed(factory,tables)
    now=now.replace(second=1,microsecond=0)
    monkeypatch.setattr("platform_core.notifications.external_projection.now_utc",lambda:now)
    async with factory() as session:
        repo=repository(session,tables)
        job=await repo.get("job",scope=scope,object_id=job_id)
        await repo.add("settings",scope=scope,actor_id="admin",now=now,values={
            "id":notification_settings_id(scope),"public_origin":"https://portal.test"})
        await repo.add("policy",scope=scope,actor_id="admin",now=now,values={
            "id":uuid4(),"connection_id":job["connection_id"],"enabled":True,
            "event_types":["aiops.situation.detected"],"binding_ids":[str(job["binding_id"])],"actor_ids":[]})
        await session.commit()
    route={"route_key":"aiops.situation.detail","route_params":{"domain_id":"12","situation_id":str(uuid4())}}
    event_ids=[uuid4(),uuid4()]
    async def enqueue(event_id):
        async with factory() as session:
            await enqueue_external_event(repo=repository(session,tables),scope=scope,
                config=ExternalDeliveryConfig(coalesce_seconds=60),
                source_event_id=event_id,event_type="aiops.situation.detected",severity="WARNING",
                route=route,legitimate_actors={"B"},occurred_at=now)
            await session.commit()
    await asyncio.gather(*(enqueue(event_id) for event_id in event_ids))
    async with factory() as session:
        repo=repository(session,tables)
        jobs=[row for row in await repo.list("job",scope=scope) if row["event_type"]=="aiops.situation.detected"]
        assert len(jobs)==2
        assert sum(row["status"]==Status.PENDING for row in jobs)==1
        assert sum(row["last_error_code"]=="NOTIFICATION_COALESCED" for row in jobs)==1
        assert len(await repo.list("entry",scope=scope))==2
        assert len(await repo.list("audit",scope=scope))==1
