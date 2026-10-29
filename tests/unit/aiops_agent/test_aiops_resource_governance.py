"""持久化配额协议、执行围栏和来源调用的本地回归；不冒充真实多进程验收。"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.dialects import oracle, postgresql

from aiops_agent.application.credential_issuance import CredentialIssuanceService, CredentialIssueError
from aiops_agent.application.resource_governance import (
    ResourceAdmissionError, ResourceGovernanceService, reserve_readonly_resource, token_hash,
)
from aiops_agent.adapters.aiops_execution_client import AIOpsExecutionClientError
from aiops_agent.entities import ResourcePoolEntity, ResourceLeaseEntity
from aiops_agent.executor.drivers import DiagnosticDriverError
from aiops_agent.executor.resource_admission import readonly_credentials
from aiops_agent.repositories.resource_governance import ResourceGovernanceRepository
from aiops_agent.workers.resource_admission import ResourceBoundSourceHandler
from platform_core.contracts.aiops.executor import CredentialIssueResponse, DiagnosticExecutionGrant, ReadonlyExecutionFenceReceipt
from platform_core.identity import uuid7
from tests.support.aiops_resources import resource_receipt, resource_repository
import aiops_agent.resource_admission as admission_module


@pytest.fixture(autouse=True)
def no_real_admission_sleep(monkeypatch):
    # 准入退避的完整序列另有专项，其他拒绝门禁不重复消耗墙钟时间。
    monkeypatch.setattr(admission_module, "ADMISSION_RETRY_DELAYS", ())


def now():
    return datetime.now(timezone.utc)


class MemoryStore:
    """显式模拟共享行锁与事务，仅验证应用协议，不模拟真实数据库隔离级别。"""
    def __init__(self, **limits):
        self.clock = now()
        self.pool = SimpleNamespace(global_limit=8, target_limit=2, source_limit=4,
            cleanup_grace_seconds=30, **{})
        for key, value in limits.items():
            setattr(self.pool, key, value)
        self.lock = asyncio.Lock()
        self.leases = {}
        self.commits = 0

    def factory(self):
        return MemoryUow(self)


class MemoryUow:
    def __init__(self, store):
        self.store = store
        self.resources = self
        self.owned = False
        self.added = []
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        if not self.committed:
            for key in self.added:
                self.store.leases.pop(key, None)
        if self.owned:
            self.store.lock.release()

    async def commit(self):
        self.committed = True
        self.store.commits += 1

    async def lock_pool(self, _key):
        if self.store.lock.locked():
            return None
        await self.store.lock.acquire()
        self.owned = True
        return self.store.pool

    async def database_now(self):
        return self.store.clock

    async def get_lease(self, lease_id, *, lock=False):
        return self.store.leases.get(lease_id)

    async def active_counts(self, *, pool_key, target_id, source_kind, source_id, now):
        assert self.owned
        # 刻意切换任务，暴露未持池锁就 count/insert 的错误实现。
        await asyncio.sleep(0)
        active = [item for item in self.store.leases.values()
                  if item.pool_key == pool_key and item.released_at is None and item.held_until > now]
        return (len(active), sum(item.target_id == target_id for item in active),
                sum(item.source_kind == source_kind and item.source_id == source_id for item in active))

    async def add_lease(self, entity):
        assert self.owned and entity.lease_id not in self.store.leases
        self.store.leases[entity.lease_id] = entity
        self.added.append(entity.lease_id)


async def reserve(store, *, target=None, source=None, kind="DATABASE", lease_id=None, commit=True, seconds=20):
    async with store.factory() as uow:
        receipt = await reserve_readonly_resource(uow=uow, lease_id=lease_id or uuid7(), run_id=uuid7(),
            target_id=target or uuid7(), source_kind=kind, source_id=source or uuid7(),
            deadline_at=store.clock + timedelta(seconds=seconds))
        if commit:
            await uow.commit()
        return receipt


@pytest.mark.parametrize("dimension", ["GLOBAL", "TARGET", "SOURCE"])
def test_global_target_source_admission_counts_share_a_single_pool(dimension):
    async def run():
        store = MemoryStore(**{f"{dimension.lower()}_limit": 1})
        target, source = uuid7(), uuid7()
        await reserve(store, target=target, source=source)
        with pytest.raises(ResourceAdmissionError, match=f"RESOURCE_QUOTA_{dimension}_EXHAUSTED"):
            await reserve(store, target=target if dimension == "TARGET" else uuid7(),
                          source=source if dimension == "SOURCE" else uuid7())
        assert len(store.leases) == store.commits == 1
    asyncio.run(run())


def test_concurrent_uows_cannot_multiply_admission_limit():
    async def run():
        store = MemoryStore(global_limit=3)
        async def caller():
            for _ in range(30):
                try:
                    return await reserve(store)
                except ResourceAdmissionError as exc:
                    if exc.code != "RESOURCE_QUOTA_UNAVAILABLE":
                        return exc.code
                    await asyncio.sleep(0)
            return "BUSY"
        results = await asyncio.gather(*(caller() for _ in range(20)))
        assert len(store.leases) == 3
        assert sum(not isinstance(item, str) for item in results) == 3
        assert not store.lock.locked()
    asyncio.run(run())


def test_rollback_does_not_consume_a_grant_or_capacity():
    async def run():
        store = MemoryStore(global_limit=1)
        key = uuid7()
        await reserve(store, lease_id=key, commit=False)
        assert not store.leases
        await reserve(store, lease_id=key)
        assert len(store.leases) == 1
    asyncio.run(run())


def test_failed_or_crashed_operation_keeps_capacity_through_cleanup_grace():
    async def run():
        store = MemoryStore(global_limit=1)
        receipt = await reserve(store, seconds=10)
        store.clock += timedelta(seconds=11)
        with pytest.raises(ResourceAdmissionError, match="RESOURCE_QUOTA_GLOBAL_EXHAUSTED"):
            await reserve(store)
        store.clock = receipt.held_until
        await reserve(store)
        with pytest.raises(ResourceAdmissionError, match="RESOURCE_LEASE_ALREADY_CONSUMED"):
            await reserve(store, lease_id=receipt.lease_id)
    asyncio.run(run())


def test_release_is_token_fenced_idempotent_and_does_not_release_another_lease():
    async def run():
        store = MemoryStore()
        receipt = await reserve(store)
        service = ResourceGovernanceService(uow_factory=store.factory)
        with pytest.raises(ResourceAdmissionError, match="RESOURCE_LEASE_RELEASE_INVALID"):
            await service.release(receipt.model_copy(update={"release_token": "x" * 43}))
        assert store.leases[receipt.lease_id].released_at is None
        await service.release(receipt)
        released = store.leases[receipt.lease_id].released_at
        await service.release(receipt)
        assert store.leases[receipt.lease_id].released_at == released
        later = await reserve(store)
        await service.release(receipt)
        assert store.leases[later.lease_id].released_at is None
        assert token_hash(receipt.release_token) == store.leases[receipt.lease_id].release_token_hash
        assert receipt.release_token not in repr(receipt)
    asyncio.run(run())


def test_database_and_monitor_source_namespaces_do_not_collide():
    async def run():
        store = MemoryStore(source_limit=1)
        source = uuid7()
        await reserve(store, source=source, kind="DATABASE")
        await reserve(store, source=source, kind="MONITOR")
        assert len(store.leases) == 2
    asyncio.run(run())


@pytest.mark.parametrize("value", [0, -1, 1025, 1.5, True, None, Decimal("NaN")])
def test_invalid_persisted_limits_fail_closed(value):
    async def run():
        store = MemoryStore(global_limit=value)
        with pytest.raises(ResourceAdmissionError, match="RESOURCE_QUOTA_POLICY_INVALID"):
            await reserve(store)
        assert not store.leases
    asyncio.run(run())


def test_oracle_decimal_policy_and_expired_grant_are_handled_explicitly():
    async def run():
        store = MemoryStore(global_limit=Decimal(8), target_limit=Decimal(2), source_limit=Decimal(4))
        await reserve(store)
        with pytest.raises(ResourceAdmissionError, match="EXECUTION_DEADLINE_EXCEEDED"):
            await reserve(store, seconds=0)
    asyncio.run(run())


def issuance_fixture():
    stamp = now()
    task = SimpleNamespace(ops_task_id=uuid7(), ops_run_id=uuid7(), status="RUNNING",
        lease_token=uuid7(), lease_until=stamp + timedelta(seconds=25),
        started_at=stamp, timeout_seconds=20)
    grant = SimpleNamespace(grant_id=uuid7(), task_id=task.ops_task_id, run_id=task.ops_run_id,
        audience="test-executor", expires_at=stamp + timedelta(seconds=30),
        tenant_id=uuid7(), domain_id=101 if DiagnosticExecutionGrant.model_fields["domain_id"].annotation is int else uuid7(),
        target_id=uuid7(), target_row_version=1, diagnostic_credential_id=uuid7(),
        lease_token_hash=hashlib.sha256(str(task.lease_token).encode()).hexdigest())
    run = SimpleNamespace(ops_run_id=grant.run_id, tenant_id=grant.tenant_id, domain_id=grant.domain_id,
        target_id=grant.target_id, cancel_requested_at=None, status="RUNNING",
        deadline_at=stamp + timedelta(seconds=60), workflow_kind="CHAT_TURN")
    target = SimpleNamespace(row_version=1, status="ENABLED", readonly_connection_enabled=True, diagnostic_credential_id=grant.diagnostic_credential_id)
    uow = SimpleNamespace(resources=resource_repository(stamp),
        runs=SimpleNamespace(get_run=AsyncMock(return_value=run), get_task=AsyncMock(return_value=task)),
        targets=SimpleNamespace(get_fully_scoped=AsyncMock(return_value=target), get_scoped=AsyncMock(return_value=target)),
        managed_credentials=SimpleNamespace(), inbox=SimpleNamespace(
            get_by_message=AsyncMock(return_value=None), add=AsyncMock()), commit=AsyncMock())

    class Factory:
        def __call__(self): return self
        async def __aenter__(self): return uow
        async def __aexit__(self, *_args): return None

    credentials = SimpleNamespace(read=AsyncMock(return_value={"username": "read-only", "password": "test-secret"}))
    codec = SimpleNamespace(verify=Mock(return_value=grant), verify_dynamic=Mock(return_value=grant))
    service = CredentialIssuanceService(uow_factory=Factory(), diagnostic_codec=codec,
        mutation_codec=Mock(), credential_service=credentials, audience="test-executor")
    return SimpleNamespace(service=service, grant=grant, run=run, task=task, target=target,
        uow=uow, credentials=credentials, codec=codec, factory=Factory())


@pytest.mark.parametrize("source", ["DATABASE", "MONITOR"])
def test_resumed_task_uses_checkpoint_deadline_but_never_extends_grant_or_lease(source):
    async def run():
        f = issuance_fixture()
        # 保留首次执行时间，实际可用期限必须来自审批恢复后的检查点。
        f.task.started_at -= timedelta(hours=1)
        f.task.execution_deadline_at = now() + timedelta(seconds=8)
        if source == "DATABASE":
            receipt = (await f.service.issue("signed-token")).resource_lease
        else:
            source_id = uuid7()
            f.run.plan_snapshot_json = {"monitoring": {"bindings": [{"source": {"source_id": str(source_id)}}]}}
            ctx = SimpleNamespace(run_id=f.run.ops_run_id, task_id=f.task.ops_task_id,
                target_id=f.run.target_id, lease_token=str(f.task.lease_token))
            receipt = await ResourceGovernanceService(uow_factory=f.factory).acquire_monitor(
                context=ctx, source_id=source_id, lease_id=uuid7())
        assert receipt.execution_deadline_at == f.task.execution_deadline_at
        assert receipt.execution_deadline_at <= f.task.lease_until
        assert receipt.execution_deadline_at <= f.grant.expires_at
    asyncio.run(run())


def test_resumed_task_cannot_reactivate_an_expired_signed_grant():
    async def run():
        f = issuance_fixture()
        f.task.execution_deadline_at = now() + timedelta(seconds=8)
        f.grant.expires_at = now() - timedelta(seconds=1)
        with pytest.raises(CredentialIssueError):
            await f.service.issue("signed-token")
        f.credentials.read.assert_not_awaited()
        f.uow.resources.add_lease.assert_not_awaited()
    asyncio.run(run())


def test_credential_issuance_reserves_in_same_transaction_before_reading_secrets():
    async def run():
        f = issuance_fixture()
        result = await f.service.issue("signed-token")
        assert result.resource_lease.lease_id == f.grant.grant_id
        assert result.resource_lease.execution_deadline_at == f.task.started_at + timedelta(seconds=20)
        f.uow.commit.assert_awaited_once()
        f.credentials.read.assert_awaited_once()
        f.uow.inbox.add.assert_awaited_once()
        assert f.uow.resources.add_lease.await_args.args[0].lease_id == f.grant.grant_id
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["cancel", "terminal", "expired_task", "task_reassigned", "task_run", "task_status", "target_version", "credential", "replay", "quota"])
def test_rejected_execution_cannot_read_secret_or_consume_new_credential(mutation):
    async def run():
        f = issuance_fixture()
        expected = CredentialIssueError
        if mutation == "cancel": f.run.cancel_requested_at = now()
        if mutation == "terminal": f.run.status = "COMPLETED"
        if mutation == "expired_task": f.task.lease_until = now() - timedelta(seconds=1)
        if mutation == "task_reassigned": f.task.lease_token = uuid7()
        if mutation == "task_run": f.task.ops_run_id = uuid7()
        if mutation == "task_status": f.task.status = "READY"
        if mutation == "target_version": f.target.row_version = 2
        if mutation == "credential": f.target.diagnostic_credential_id = uuid7()
        if mutation == "replay": f.uow.inbox.get_by_message.return_value = object()
        if mutation == "quota":
            expected = ResourceAdmissionError
            f.uow.resources.active_counts.return_value = (8, 0, 0)
        with pytest.raises(expected):
            await f.service.issue("signed-token")
        f.credentials.read.assert_not_awaited()
        f.uow.inbox.add.assert_not_awaited()
        f.uow.commit.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["none", "cancel", "lease", "source", "timeout"])
def test_monitor_admission_checks_current_run_task_source_and_deadline(mutation):
    async def run():
        f = issuance_fixture()
        source_id = uuid7()
        f.run.plan_snapshot_json = {"monitoring": {"bindings": [{"source": {"source_id": str(source_id)}}]}}
        ctx = SimpleNamespace(run_id=f.run.ops_run_id, task_id=f.task.ops_task_id,
            target_id=f.run.target_id, lease_token=str(f.task.lease_token))
        if mutation == "cancel": f.run.cancel_requested_at = now()
        if mutation == "lease": ctx.lease_token = str(uuid7())
        if mutation == "source": source_id = uuid7()
        if mutation == "timeout": f.task.started_at -= timedelta(seconds=60)
        service = ResourceGovernanceService(uow_factory=f.factory)
        if mutation == "none":
            receipt = await service.acquire_monitor(context=ctx, source_id=source_id, lease_id=uuid7())
            assert receipt.execution_deadline_at == f.task.started_at + timedelta(seconds=20)
            assert f.uow.resources.add_lease.await_args.args[0].source_kind == "MONITOR"
        else:
            with pytest.raises(ResourceAdmissionError):
                await service.acquire_monitor(context=ctx, source_id=source_id, lease_id=uuid7())
            f.uow.commit.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["success", "quota", "missing", "different_grant", "expired", "failure", "cancel"])
def test_executor_never_queries_without_receipt_or_releases_uncertain_execution(mode):
    async def run():
        grant = SimpleNamespace(grant_id=uuid7(), trace_id="test", db_type="POSTGRESQL", expires_at=now() + timedelta(seconds=10))
        receipt = resource_receipt(grant)
        if mode == "different_grant": receipt = receipt.model_copy(update={"lease_id": uuid7()})
        if mode == "expired": receipt = receipt.model_copy(update={"execution_deadline_at": now() - timedelta(seconds=1)})
        control = SimpleNamespace(issue_credential=AsyncMock(return_value=CredentialIssueResponse(
            username="readonly", password="hidden", resource_lease=None if mode == "missing" else receipt)),
            release_resource=AsyncMock(), report_readonly_termination=AsyncMock(), check_readonly_execution=AsyncMock(return_value=ReadonlyExecutionFenceReceipt(
                allowed=True, checked_at=now(), execution_deadline_at=grant.expires_at)))
        if mode == "quota":
            control.issue_credential.side_effect = AIOpsExecutionClientError("RESOURCE_QUOTA_TARGET_EXHAUSTED")
        driver = AsyncMock()
        async def execute():
            async with readonly_credentials(control, token="signed", grant=grant):
                await driver()
                if mode == "failure": raise RuntimeError("查询失败")
                if mode == "cancel": raise asyncio.CancelledError()
        if mode == "success":
            await execute()
            control.release_resource.assert_awaited_once()
        elif mode in {"failure", "cancel"}:
            with pytest.raises(RuntimeError if mode == "failure" else asyncio.CancelledError):
                await execute()
            control.release_resource.assert_not_awaited()
        else:
            with pytest.raises(DiagnosticDriverError):
                await execute()
            driver.assert_not_awaited()
            control.release_resource.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("prefix", ["observe:", "log:"])
@pytest.mark.parametrize("mode", ["success", "quota", "cancel"])
def test_monitor_log_wrapper_never_turns_resource_denial_into_empty_success(prefix, mode):
    async def run():
        source_id, binding_id = uuid7(), str(uuid7())
        grant = SimpleNamespace(grant_id=uuid7(), expires_at=now() + timedelta(seconds=10))
        governance = SimpleNamespace(acquire_monitor=AsyncMock(return_value=resource_receipt(grant)), release=AsyncMock())
        handler = SimpleNamespace(execute=AsyncMock(return_value=object()))
        if mode == "quota": governance.acquire_monitor.side_effect = ResourceAdmissionError("RESOURCE_QUOTA_SOURCE_EXHAUSTED")
        if mode == "cancel": handler.execute.side_effect = asyncio.CancelledError()
        context = SimpleNamespace(task_key=f"{prefix}{binding_id}", target_id=str(uuid7()),
            plan_snapshot={"monitoring": {"bindings": [{"binding_id": binding_id, "source": {"source_id": str(source_id)}}],
                "window": {"start": now().isoformat(), "end": (now() + timedelta(seconds=15)).isoformat()}}})
        wrapped = ResourceBoundSourceHandler(handler=handler, governance=governance, prefix=prefix)
        if mode == "cancel":
            with pytest.raises(asyncio.CancelledError): await wrapped.execute(context)
            governance.release.assert_not_awaited()
        else:
            result = await wrapped.execute(context)
            if mode == "success":
                assert result is handler.execute.return_value
                governance.release.assert_awaited_once()
            else:
                assert result.gaps[0].code == "RESOURCE_QUOTA_SOURCE_EXHAUSTED"
                handler.execute.assert_not_awaited()
                governance.release.assert_not_awaited()
    asyncio.run(run())


def test_pool_lock_sql_is_skip_locked_on_both_control_plane_databases():
    async def run():
        session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: None)))
        repository = ResourceGovernanceRepository(session)
        await repository.lock_pool("READONLY")
        statement = session.execute.await_args.args[0]
        for dialect in (oracle.dialect(), postgresql.dialect()):
            sql = str(statement.compile(dialect=dialect)).upper()
            assert "FOR UPDATE SKIP LOCKED" in sql
        assert ResourcePoolEntity.__table__.primary_key.columns.keys() == ["pool_key"]
        assert ResourceLeaseEntity.__table__.primary_key.columns.keys() == ["lease_id"]
    asyncio.run(run())


def test_production_bootstrap_enforces_both_database_and_source_resource_paths():
    root = Path(__file__).resolve().parents[3]
    service_root = root / "services/aiops_agent/src/aiops_agent"
    assert "ResourceGovernanceService(uow_factory=runtime.uow_factory)" in (service_root / "bootstrap/worker.py").read_text()
    assert "ResourceGovernanceService(uow_factory=runtime.uow_factory)" in (service_root / "bootstrap/api.py").read_text()
    assert '"aiops.resources.release"' in (service_root / "bootstrap/api.py").read_text()
    for module in ("service.py", "dynamic_service.py"):
        content = (service_root / "executor" / module).read_text()
        assert "readonly_credentials(" in content
        assert "._control_plane.issue_credential(" not in content


def test_resource_schema_has_no_business_unique_or_check_constraints():
    from sqlalchemy import CheckConstraint, UniqueConstraint
    for entity in (ResourcePoolEntity, ResourceLeaseEntity):
        assert not any(isinstance(item, (CheckConstraint, UniqueConstraint)) for item in entity.__table__.constraints)
        assert not any(item.unique for item in entity.__table__.indexes)
    assert set(ResourceLeaseEntity.__table__.columns.keys()) == {
        "lease_id", "pool_key", "run_id", "target_id", "source_kind", "source_id",
        "release_token_hash", "acquired_at", "execution_deadline_at", "held_until", "released_at",
    }


def test_canonical_deployment_initializes_shared_pool_but_runtime_never_creates_it():
    import json
    root = Path(__file__).resolve().parents[3]
    if ResourcePoolEntity.__tablename__ == "KBOT_OPS_RESOURCE_POOL":
        folder = root / "database/oracle/aiops_agent"
        ddl = (folder / "013_ops_resource_governance.sql").read_text()
        manifest = json.loads((folder / "schema_manifest.json").read_text())
        assert manifest["schema_version"] == 52
        item = next(item for item in manifest["scripts"] if item["name"] == "013_ops_resource_governance.sql")
        assert item["sha256"] == hashlib.sha256(ddl.encode()).hexdigest()
        assert ddl in (root / "database/oracle/operations/apply_aiops_schema_51.sql").read_text()
        seed = (root / "database/oracle/bootstrap/aiops/resource_pool.sql").read_text()
        assert seed in (root / "database/oracle/operations/apply_aiops_schema_51.sql").read_text()
        for name in ("rebuild_aiops_schema.sql", "rebuild_aiops_preserve_sources_targets.sql"):
            assert ddl in (root / "database/oracle/generated/aiops_agent" / name).read_text()
            assert seed in (root / "database/oracle/generated/aiops_agent" / name).read_text()
    else:
        source = (root / "scripts/deployment/init_postgresql.py").read_text()
        assert source.count("await _seed_aiops_resource_pool(connection, schema=schema)") == 3
        assert "--align-aiops-resource-governance-only" in source
        assert "ON CONFLICT (pool_key) DO NOTHING" in source
    service_root = root / "services/aiops_agent/src/aiops_agent"
    for name in ("application/resource_governance.py", "repositories/resource_governance.py",
                 "executor/resource_admission.py", "workers/resource_admission.py"):
        content = (service_root / name).read_text().lower()
        assert "create_all" not in content and "create table" not in content


@pytest.mark.parametrize("code,retryable", [
    ("RESOURCE_QUOTA_TARGET_EXHAUSTED", True),
    ("RESOURCE_QUOTA_POLICY_INVALID", False),
    ("CREDENTIAL_GRANT_INVALID", False),
])
def test_only_transient_resource_denials_are_retryable(code, retryable):
    async def run():
        control = SimpleNamespace(issue_credential=AsyncMock(side_effect=AIOpsExecutionClientError(code)))
        grant = SimpleNamespace(trace_id="test")
        with pytest.raises(DiagnosticDriverError) as error:
            async with readonly_credentials(control, token="signed", grant=grant):
                pytest.fail("未取得资源不得进入执行")
        assert error.value.code == code and error.value.retryable is retryable
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["available", "saturated", "revoked", "cancel"])
def test_admission_retries_only_busy_acquisition_and_is_bounded(monkeypatch, mode):
    async def run():
        delays = (0.01, 0.02, 0.03)
        monkeypatch.setattr(admission_module, "ADMISSION_RETRY_DELAYS", delays)
        pauses = []
        async def pause(seconds): pauses.append(seconds)
        monkeypatch.setattr(admission_module.asyncio, "sleep", pause)
        busy = ResourceAdmissionError("RESOURCE_QUOTA_TARGET_EXHAUSTED")
        operation = AsyncMock(side_effect=[busy, busy, "lease"] if mode == "available" else
            ResourceAdmissionError("RESOURCE_EXECUTION_FENCE_INVALID") if mode == "revoked" else
            asyncio.CancelledError() if mode == "cancel" else busy)
        if mode == "available":
            assert await admission_module.wait_for_resource(operation, retry_errors=(ResourceAdmissionError,)) == "lease"
            assert pauses == list(delays[:2]) and operation.await_count == 3
        else:
            with pytest.raises(asyncio.CancelledError if mode == "cancel" else ResourceAdmissionError):
                await admission_module.wait_for_resource(operation, retry_errors=(ResourceAdmissionError,))
            assert operation.await_count == (4 if mode == "saturated" else 1)
            assert pauses == (list(delays) if mode == "saturated" else [])
    asyncio.run(run())
