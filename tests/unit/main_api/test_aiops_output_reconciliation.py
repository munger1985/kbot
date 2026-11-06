"""通知乱序、真实授权对账与最小状态合同；离线替身不代表生产验收。"""

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from sqlalchemy.dialects import oracle, postgresql

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.output_projection import OutputProjectionService
from main_api.application.aiops_output_reconciliation import AIOpsOutputReconciliation
from main_api.application.output_authority import OutputAuthorityDenied, OutputAuthorityService
from platform_core.contracts import PrincipalKind
from platform_core.contracts.aiops.output_projection import OutputProjectionRequest, OutputProjectionSnapshot
from platform_core.notifications.source_order import output_event_version, source_version_decision
from tests.unit.main_api.test_aiops_output_authority import authority_request, Uow as AuthorityUow, ORACLE
from tests.unit.aiops_agent.test_aiops_output_job_api import api, human_context
from tests.unit.aiops_agent.test_aiops_output_jobs import Store, MemoryUow, NOW
from platform_core.identity import uuid7
from platform_core.security import create_service_auth_context


def snapshot(status="RUNNING", version=8):
    authority = authority_request(job_type="CASE_EXTRACTION")
    request = OutputProjectionRequest(**authority.model_dump(include={
        "job_id", "job_type", "actor_id", "domain_id", "tenant_id"}))
    return OutputProjectionSnapshot(request=request, authority=authority, status=status,
        stage={"RUNNING": "INDEXING", "FAILED": "FAILED", "SUCCEEDED": "COMPLETED"}[status],
        source_version=version, source_updated_at=NOW, source_security_level=2)


@pytest.mark.parametrize("current,incoming,old,new,decision", [
    (None, 1, None, "RUNNING", "ADVANCE"),
    (2, 3, "FAILED", "RUNNING", "ADVANCE"),
    (9, 8, "RUNNING", "FAILED", "IGNORE"),
    (9, 8, "SUCCEEDED", "RUNNING", "IGNORE"),
    (9, 9, "SUCCEEDED", "SUCCEEDED", "DELIVER"),
    (9, 9, "SUCCEEDED", "FAILED", "IGNORE"),
    (9, 10, "FAILED", "SUCCEEDED", "ADVANCE"),
    (9, 10, "SUCCEEDED", "FAILED", "ADVANCE"),
])
def test_version_order_overrules_clock_and_keeps_same_terminal_notification(current, incoming, old, new, decision):
    assert source_version_decision(current=current, incoming=incoming,
        current_status=old, incoming_status=new) == decision


@pytest.mark.parametrize("bad", [None, 0, -1, True, "8", 8.1])
def test_output_event_requires_integer_business_version(bad):
    with pytest.raises(ValueError):
        output_event_version(producer="aiops", event_type="aiops.case.draft_ready",
            params={}, data={"source_version": bad})


def test_non_output_events_do_not_inherit_output_ordering():
    assert output_event_version(producer="aiops", event_type="aiops.report.ready", params={}, data={}) is None


@pytest.mark.parametrize("field,value", [("job_id", uuid7()), ("actor_id", "other"), ("domain_id", "99")])
def test_snapshot_rejects_identity_mismatch(field, value):
    original = snapshot()
    with pytest.raises(ValidationError):
        OutputProjectionSnapshot.model_validate({**original.model_dump(),
            "request": {**original.request.model_dump(), field: value}})


@pytest.mark.parametrize("changed", [None, "user", "domain", "permission", "security", "response"])
def test_reconcile_uses_real_service_and_current_authority(changed):
    async def run():
        original = snapshot()
        authority_uow = AuthorityUow()
        client = SimpleNamespace(get_output_projection=AsyncMock(return_value=original.model_dump(mode="json")))
        resolver = AIOpsOutputReconciliation(client=client,
            authority=OutputAuthorityService(uow_factory=lambda: authority_uow), caller_service="main-api")
        if changed == "user":
            authority_uow.authentication.background_actor_is_active.return_value = False
            authority_uow.access.get_user.return_value = SimpleNamespace(status="DISABLED", max_security_level=2)
        elif changed == "domain":
            authority_uow.app_domains.output_job_binding.return_value = None
            authority_uow.domains.exists_active.return_value = False
        elif changed == "permission":
            authority_uow.app_domains.list_user_app_permissions.return_value = ()
            authority_uow.access.permissions_for.return_value = ()
        elif changed == "security":
            original = original.model_copy(update={"source_security_level": 3})
            client.get_output_projection.return_value = original.model_dump(mode="json")
            authority_uow.access.get_user.return_value.max_security_level = 1
        elif changed == "response":
            client.get_output_projection.return_value = snapshot().model_dump(mode="json")
        if changed == "security" and not ORACLE:
            # Ammolite 当前平台无独立最高密级；不得伪造一个 KBot 专属限制。
            assert await resolver.resolve(original.request) == original
        elif changed:
            with pytest.raises((OutputAuthorityDenied, PermissionError, ValueError)):
                await resolver.resolve(original.request)
        else:
            assert await resolver.resolve(original.request) == original
        context = client.get_output_projection.await_args.kwargs["auth_context"]
        assert context.principal_kind == PrincipalKind.SERVICE
        assert not getattr(context, "asserted_user_id", None)
        assert not getattr(context, "permissions", ())
    asyncio.run(run())


@pytest.mark.parametrize("changed", [None, "actor", "domain", "tenant", "type"])
def test_projection_read_is_scoped_and_never_returns_frozen_body(changed):
    async def run():
        store = Store()
        receipt = await store.submit()
        job = store.jobs[receipt.job_id]
        job.input_snapshot_json["parameters"]["authority"] = {
            "origin": "HUMAN", "source_agent_ids": [str(uuid7())]}
        request = OutputProjectionRequest(job_id=job.job_id, job_type=job.job_type,
            actor_id=str(store.actor), domain_id=str(store.domain), tenant_id=store.tenant)
        updates = {}
        if changed == "actor": updates["actor_id"] = str(uuid7()) if store.tenant else "other"
        if changed == "domain": updates["domain_id"] = str(uuid7()) if store.tenant else "2"
        if changed == "tenant": updates["tenant_id"] = uuid7()
        if changed == "type": updates["job_type"] = "CASE_EXTRACTION"
        request = request.model_copy(update=updates)
        service = OutputProjectionService(uow_factory=lambda: MemoryUow(store), tenant_scoped=bool(store.tenant))
        if changed:
            with pytest.raises(AIOpsApplicationError):
                await service.read(request)
        else:
            value = await service.read(request)
            assert value.request == request and value.source_version == 1
            assert value.status == "QUEUED"
            assert not {"result", "input_snapshot_json", "artifacts", "parameters", "error_message",
                "processing_ref_json", "lease_token", "token"} & value.model_dump().keys()
        assert len(store.jobs) == len(store.events) == len(store.notifications) == 1
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["service", "human", "scope"])
def test_projection_endpoint_requires_service_and_narrow_scope(mode):
    async def run():
        store = Store()
        receipt = await store.submit()
        job = store.jobs[receipt.job_id]
        job.input_snapshot_json["parameters"]["authority"] = {"origin": "HUMAN", "source_agent_ids": [str(uuid7())]}
        request = OutputProjectionRequest(job_id=job.job_id, job_type=job.job_type,
            actor_id=str(store.actor), domain_id=str(store.domain), tenant_id=store.tenant)
        context = create_service_auth_context(caller_service="main-api", trace_id="test")
        if mode == "human": context = human_context(store)
        app = api(store, context=context, scopes=("aiops.run",) if mode == "scope" else ("aiops.output.reconcile",))
        app.state.output_projection_service = OutputProjectionService(
            uow_factory=lambda: MemoryUow(store), tenant_scoped=bool(store.tenant))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://aiops.test") as client:
            response = await client.post("/internal/v1/aiops/output-jobs:projection",
                json=request.model_dump(mode="json"))
            assert response.status_code == (200 if mode == "service" else 403)
    asyncio.run(run())


@pytest.mark.parametrize("changed", [False, True])
def test_repository_cas_fences_source_version_row_version_and_scope(changed):
    async def run():
        session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(rowcount=int(changed))))
        operation_id, tenant = uuid7(), uuid7()
        if ORACLE:
            from main_api.repositories.notifications import NotificationRepository
            applied = await NotificationRepository(session).reconcile_output_operation(
                operation_id=operation_id, domain_id=7, source_operation_id=str(uuid7()),
                expected_row_version=19, source_version=24, source_updated_at=NOW,
                status="RUNNING", summary="处理中", error_code=None)
            statement = session.execute.await_args.args[0]
            compiled = statement.compile(dialect=oracle.dialect())
        else:
            from main_api.repositories.notification_projection import update_notification_operation_status
            applied = await update_notification_operation_status(session, operation_id=operation_id,
                tenant_id=tenant, source_service="aiops", source_operation_id=uuid7(),
                target_status="RUNNING", expected_row_version=19, source_version=24,
                source_updated_at=NOW)
            statement = session.execute.await_args.args[0]
            compiled = statement.compile(dialect=postgresql.dialect())
        assert applied == changed
        sql = str(compiled).lower()
        where = sql.split(" where ")[1]
        assert "row_version =" in where and "source_version <=" in where
        assert "source_operation_id =" in where and ("domain_id =" in where if ORACLE else "tenant_id =" in where)
        assert 19 in compiled.params.values() and 24 in compiled.params.values()
    asyncio.run(run())


def test_projection_rejects_older_terminal_and_allows_new_retry_even_with_older_clock():
    async def run():
        job_id, domain, actor, tenant = uuid7(), (7 if ORACLE else uuid7()), ("operator" if ORACLE else uuid7()), uuid7()
        def event(event_type, version, offset=0):
            if ORACLE:
                from platform_core.notifications import NotificationEnvelope
                return SimpleNamespace(outbox_id=uuid7(), domain_id=domain, producer_service="aiops",
                    payload_json=NotificationEnvelope(domain_id=domain, event_type=event_type,
                        resource_type="aiops_output_job", resource_id=str(job_id), operation_id=str(job_id),
                        initiator_actor_id=actor, recipient_actor_ids=[actor], summary="安全摘要",
                        correlation_id="trace", occurred_at=NOW + timedelta(seconds=offset),
                        safe_data={"job_id": str(job_id), "job_type": "CASE_EXTRACTION",
                            "source_version": version}).model_dump(mode="json"))
            return dict(event_type=event_type, operation_id=str(job_id), resource_type="aiops_output_job",
                resource_id=str(job_id), initiator_id=str(actor), tenant_id=str(tenant),
                occurred_at=(NOW + timedelta(seconds=offset)).isoformat(),
                route_key="aiops.output_job.detail", route_params={"job_id": str(job_id),
                    "domain_id": str(domain), "job_type": "CASE_EXTRACTION"},
                safe_payload={"source_version": version})

        if ORACLE:
            from tests.unit.main_api.test_notification_center_s6 import _Repository, _Uow
            from main_api.application.notification_projection import NotificationProjectionService
            repo = _Repository()
            events = {}
            repo.operation_event_payload = AsyncMock(side_effect=lambda *, outbox_id: events[outbox_id])
            async def project(type_, version, offset=0):
                value = event(type_, version, offset)
                events[value.outbox_id] = value.payload_json
                await NotificationProjectionService().project(uow=_Uow(repo), outbox=value)
                return next(iter(repo.operations.values())).status
            assert await project("aiops.output_job.failed", 5) == "FAILED"
            assert await project("aiops.output_job.started", 8, -10) == "RUNNING"
            assert await project("aiops.output_job.failed", 6, 50) == "RUNNING"
            assert len(repo.inbox) == 1
            assert await project("aiops.case.draft_ready", 10) == "SUCCEEDED"
            assert len(repo.inbox) == 2
            assert await project("aiops.output_job.started", 9, 99) == "SUCCEEDED"
        else:
            from main_api.repositories.notification_projection import _upsert_operation
            existing = dict(id=uuid7(), status="FAILED", source_updated_at=NOW,
                source_version=5, initiator_id=actor,
                route_params=event("aiops.output_job.started", 1)["route_params"])
            async def project(type_, version, target, offset=0):
                statements = []
                async def execute(statement, parameters):
                    statements.append(str(statement))
                    return SimpleNamespace(mappings=lambda: SimpleNamespace(one_or_none=lambda: existing),
                        scalar_one_or_none=lambda: existing["id"])
                session = SimpleNamespace(execute=execute)
                delivered = await _upsert_operation(session, event_record_id=uuid7(), producer="aiops",
                    envelope=event(type_, version, offset), projection={"operation": {
                        "status": target, "stage_code": "test", "stage_label": "测试"}},
                    title="测试", summary="安全摘要")
                return delivered, statements
            deliver, statements = await project("aiops.output_job.started", 8, "RUNNING", -10)
            assert deliver and "ON CONFLICT" in statements[1]
            assert "EXCLUDED.source_version > platform_background_operation.source_version" in statements[1]
            existing.update(status="RUNNING", source_version=8)
            deliver, statements = await project("aiops.output_job.failed", 6, "FAILED", 99)
            assert not deliver and len(statements) == 1
            existing.update(status="SUCCEEDED", source_version=10)
            deliver, statements = await project("aiops.case.draft_ready", 10, "SUCCEEDED")
            assert deliver and len(statements) == 1
            deliver, statements = await project("aiops.output_job.failed", 10, "FAILED")
            assert not deliver and len(statements) == 1
    asyncio.run(run())

@pytest.mark.parametrize("scenario", ["apply", "slow", "revoked", "corrupt"])
def test_reconciliation_tick_uses_fresh_write_uow_and_never_overwrites_newer_retry(scenario):
    async def run():
        value = snapshot(status="SUCCEEDED", version=20)
        stop = asyncio.Event()
        state = SimpleNamespace(row_version=9, status="RUNNING", writes=0, active=False, entries=0)
        resolver = SimpleNamespace(resolve=AsyncMock())
        async def resolve(request):
            assert not state.active
            if scenario == "revoked": raise OutputAuthorityDenied()
            if scenario == "slow":
                state.row_version, state.status = 10, "RUNNING"
            return value
        resolver.resolve.side_effect = resolve
        async def write(**kwargs):
            stop.set()
            state.writes += 1
            if state.row_version != kwargs["expected_row_version"]: return False
            state.status = kwargs.get("status", kwargs.get("target_status"))
            return True
        async def defer(**kwargs):
            stop.set()
        repo = SimpleNamespace(
            reconcile_output_operation=AsyncMock(side_effect=write),
            update_operation_status=AsyncMock(side_effect=write),
            defer_output_reconciliation=AsyncMock(side_effect=defer),
            defer_operation_reconciliation=AsyncMock(side_effect=defer),
            list_work_item_candidates=AsyncMock(return_value=[]),
            apply_retention=AsyncMock())
        if ORACLE:
            from platform_core.notifications import NotificationEnvelope
            from main_api.application.notification_projection import reconcile_output_operations
            row = dict(operation_id=uuid7(), domain_id=int(value.request.domain_id),
                source_operation_id=str(value.request.job_id), initiator_actor_id=value.request.actor_id,
                status="RUNNING", source_version=8, row_version=9,
                envelope=NotificationEnvelope(domain_id=int(value.request.domain_id),
                    event_type="aiops.output_job.started", resource_type="aiops_output_job",
                    resource_id=str(value.request.job_id), operation_id=str(value.request.job_id),
                    initiator_actor_id=value.request.actor_id, summary="处理中", correlation_id="trace",
                    safe_data={"job_id": str(value.request.job_id), "job_type": value.request.job_type,
                        "source_version": 8}).model_dump(mode="json"))
            if scenario == "corrupt": row["envelope"]["domain_id"] += 1
            repo.output_operation_candidates = AsyncMock(return_value=[row])
        else:
            from main_api.application.notification_projection import run_notification_reconciler
            row = dict(id=uuid7(), tenant_id=value.request.tenant_id, source_service="aiops",
                source_operation_id=value.request.job_id, initiator_id=UUID(value.request.actor_id),
                status="RUNNING", source_version=8, row_version=9,
                route_params={"job_id": str(value.request.job_id), "domain_id": value.request.domain_id,
                    "job_type": value.request.job_type})
            if scenario == "corrupt": row["route_params"]["job_id"] = str(uuid7())
            repo.list_operation_candidates = AsyncMock(return_value=[row])
        class Unit:
            def __init__(self):
                self.notifications = self.notification_projection = repo
            async def __aenter__(self):
                assert not state.active
                state.active = True
                state.entries += 1
                return self
            async def __aexit__(self, *_): state.active = False
            async def commit(self): pass
        if ORACLE:
            await reconcile_output_operations(uow_factory=Unit, resolver=resolver)
        else:
            from unittest.mock import patch
            app = SimpleNamespace(state=SimpleNamespace(output_reconciliation=resolver))
            with patch("main_api.application.notification_projection._reconcile_context") as fake_user:
                await run_notification_reconciler(app, Unit, stop=stop, interval_seconds=0.01)
                fake_user.assert_not_called()
        assert state.entries == 2
        assert state.writes == (1 if scenario in {"apply", "slow"} else 0)
        assert state.status == ("SUCCEEDED" if scenario == "apply" else "RUNNING")
    asyncio.run(run())

@pytest.mark.parametrize("terminal", ["SUCCEEDED", "FAILED"])
def test_terminal_outbox_version_matches_real_orm_flush(terminal):
    """SQLite 只验证真实 ORM 刷新版本语义，不冒充 PostgreSQL／Oracle 并发验收。"""
    async def run():
        from sqlalchemy import create_engine, select
        from sqlalchemy.orm import Session
        from aiops_agent.entities import OpsOutputJobEntity
        from aiops_agent.application.output_finalization import OutputJobFinalizer
        from aiops_agent.application.errors import AIOpsApplicationError
        from aiops_agent.domain.output_jobs import start_attempt
        from aiops_agent.workers.output_jobs import OutputJobWorker, OutputExecution, OutputAdvance
        from platform_core.contracts.aiops.output_jobs import FrozenOutputInput, OutputJobResult

        store = Store()
        receipt = await store.submit()
        job = store.jobs[receipt.job_id]
        job.job_type = "CASE_EXTRACTION"
        frozen = FrozenOutputInput.model_validate(job.input_snapshot_json)
        engine = create_engine("sqlite:///:memory:")
        try:
            # 只替换隔离 SQLite 的 DDL 类型拼写，不修改生产模型或 ORM 版本机制。
            from unittest.mock import patch
            with patch.object(engine.dialect.type_compiler_instance, "visit_JSONB",
                return_value="JSON", create=True):
                OpsOutputJobEntity.__table__.create(engine)
            with Session(engine, expire_on_commit=False) as session:
                session.add(job)
                session.commit()
                token = uuid7()
                start_attempt(job, now=NOW, owner="worker", token=token, until=NOW + timedelta(minutes=1))
                session.commit()
                initial_version = job.row_version
                notifications = []
                async def notify(**kwargs):
                    database_version = session.scalar(select(OpsOutputJobEntity.row_version).where(
                        OpsOutputJobEntity.job_id == job.job_id))
                    assert job.status == terminal
                    assert job.row_version == database_version > initial_version
                    notifications.append((job.status, job.row_version))
                class Unit:
                    def __init__(self):
                        self.output_jobs = SimpleNamespace(get=AsyncMock(return_value=job), finish_attempt=AsyncMock())
                        self.runs = SimpleNamespace(database_now=AsyncMock(return_value=NOW))
                        self.platform_notifications = SimpleNamespace(emit_output_job_event=notify)
                    async def __aenter__(self): return self
                    async def __aexit__(self, *args):
                        if args[0]: session.rollback()
                    async def flush(self): session.flush()
                    async def commit(self): session.commit()
                execution = OutputExecution(job.job_id, job.job_type, store.scope, {}, token, "INDEXING", frozen)
                finalizer = OutputJobFinalizer()
                result = OutputJobResult(asset_id=uuid7(), asset_version_id=uuid7())
                combined = SimpleNamespace(succeeded=AsyncMock(return_value=result),
                    notify_succeeded=finalizer.notify_succeeded, failed=finalizer.failed)
                worker = OutputJobWorker(uow_factory=Unit,
                    authority=SimpleNamespace(resolve=AsyncMock(return_value=object())),
                    processor=SimpleNamespace(advance=AsyncMock(return_value=OutputAdvance(
                        stage="FINALIZING", processing_ref={}, finalization={"test": True}))),
                    finalizer=combined, worker_id="worker")
                if terminal == "SUCCEEDED":
                    await worker._advance(execution)
                else:
                    await worker._record_failure(execution,
                        AIOpsApplicationError(code="SOURCE_REVOKED", message="来源失效",
                            status_code=403, retryable=False), previous_failures=0)
                assert notifications == [(terminal, job.row_version)]
        finally:
            engine.dispose()
    asyncio.run(run())

@pytest.mark.parametrize("mode", ["service", "human", "scope", "caller", "unsigned"])
def test_real_aiops_boundary_limits_projection_to_main_service(mode):
    async def run():
        from aiops_agent.bootstrap.api import create_aiops_api
        from aiops_agent.config import AIOpsSettings
        from platform_core.security import build_scoped_internal_auth_headers
        from platform_core.security import create_auth_context_codec, create_service_identity_codec
        settings = AIOpsSettings()
        app = create_aiops_api(settings)
        # 不启动真实数据库生命周期；生产中间件使用真实签验器。
        app.state.auth_context_codec = create_auth_context_codec()
        app.state.service_identity_codec = create_service_identity_codec()
        app.state.business_license_service = SimpleNamespace(is_allowed=AsyncMock(return_value=True))
        value = snapshot()
        app.state.output_projection_service = SimpleNamespace(read=AsyncMock(return_value=value))
        caller = ("kbot-" if ORACLE else "ammolite-") + ("aiops-worker" if mode == "caller" else "main-api")
        context = create_service_auth_context(caller_service=caller, trace_id="boundary")
        if mode == "human": context = human_context(Store())
        headers = {} if mode == "unsigned" else build_scoped_internal_auth_headers(
            audience=settings.api.service_name, caller_service=caller,
            scopes=("aiops.run",) if mode == "scope" else ("aiops.output.reconcile",), context=context)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://aiops.test") as client:
            response = await client.post("/internal/v1/aiops/output-jobs:projection",
                headers=headers, json=value.request.model_dump(mode="json"))
        if mode == "service":
            assert response.status_code == 200, response.text
            app.state.output_projection_service.read.assert_awaited_once()
        else:
            assert response.status_code in {401, 403}
            app.state.output_projection_service.read.assert_not_awaited()
    asyncio.run(run())
