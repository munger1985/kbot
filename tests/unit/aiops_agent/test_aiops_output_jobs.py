"""独立输出任务协议回归；内存事务不冒充真实数据库并发验收。"""

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import oracle, postgresql

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.output_jobs import OutputJobScope, OutputJobService, job_receipt
from aiops_agent.domain.output_jobs import (
    OutputJobConflictError, can_claim, complete_job, fail_job, require_lease, retry_job, start_attempt,
)
from aiops_agent.entities import OpsOutputJobEntity, OpsOutputJobAttemptEntity
from aiops_agent.persistence import AIOpsUnitOfWork
from aiops_agent.repositories.output_jobs import OutputJobRepository
from platform_core.contracts.aiops.output_jobs import (
    FrozenOutputInput, OutputJobReceipt, OutputJobResult, OutputJobSource, OutputSourceArtifact,
)
from platform_core.identity import uuid7


NOW = datetime(2026, 11, 3, 1, tzinfo=timezone.utc)


def fixture_job(**changes):
    source = OutputJobSource(kind="CONVERSATION", source_id=uuid7())
    values = dict(
        job_id=uuid7(), domain_id=1, actor_id="actor", job_type="REPORT_GENERATION",
        source_kind=source.kind, source_id=source.source_id, source_json=source.model_dump(mode="json"),
        conversation_id=source.source_id, anchor_run_id=None, idempotency_key_hash="a" * 64,
        input_hash="b" * 64, input_snapshot_json={}, status="QUEUED", stage="ACCEPTED",
        attempt_count=0, available_at=NOW, started_at=None, completed_at=None,
        lease_owner=None, lease_token=None, lease_until=None, heartbeat_at=None, updated_at=NOW,
        result_json=None, error_code=None, error_message=None, retryable=False, failure_stage=None,
        row_version=1, trace_id="trace", submitted_at=NOW,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def claim(job, *, token=None, now=NOW):
    token = token or uuid7()
    start_attempt(job, now=now, owner="worker-a", token=token, until=now + timedelta(seconds=30))
    return token


class Store:
    def __init__(self):
        self.jobs, self.events, self.notifications = {}, [], []
        self.lock = asyncio.Lock()
        self.domain = uuid7() if "tenant_id" in OpsOutputJobEntity.__table__.c else 1
        self.tenant = uuid7() if "tenant_id" in OpsOutputJobEntity.__table__.c else None
        self.actor = uuid7() if self.tenant else "actor"
        self.scope = OutputJobScope(domain_id=self.domain, actor_id=self.actor, tenant_id=self.tenant, trace_id="trace")
        self.source = OutputJobSource(kind="CONVERSATION", source_id=uuid7())
        self.root = SimpleNamespace(tenant_id=self.tenant, created_by=self.actor)
        self.fail_event = False
        self.fail_notification = self.fail_commit = False
        self.freeze_calls = 0
        self.fingerprint = "a" * 64
        self.artifact = OutputSourceArtifact(
            run_id=uuid7(), artifact_id=uuid7(), schema_version="AIOPS_TURN_RESULT.v1",
            content_hash="b" * 64, security_level=2, turn_id=uuid7(),
        )
        self.service = OutputJobService(uow_factory=lambda: MemoryUow(self))

    async def freeze(self, _uow, root):
        assert root.created_by == self.scope.actor_id
        self.freeze_calls += 1
        return FrozenOutputInput(
            source=self.source, artifacts=(self.artifact,), source_fingerprint=self.fingerprint,
            frozen_at=NOW + timedelta(seconds=self.freeze_calls),
        )

    async def submit(self, key="request-1", **kwargs):
        return await self.service.submit(
            scope=self.scope, source=self.source, job_type="REPORT_GENERATION",
            idempotency_key=key, freeze=self.freeze, **kwargs,
        )


class MemoryUow:
    def __init__(self, store):
        self.store = store
        self.pending_jobs, self.pending_events = {}, []
        self.pending_notifications = []
        self.output_jobs = self.outbox = self
        self.platform_notifications = self
        self.runs = SimpleNamespace(database_now=AsyncMock(return_value=NOW))
        self.owned = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        if self.owned:
            self.store.lock.release()

    async def lock_source(self, **kwargs):
        await self.store.lock.acquire()
        self.owned = True
        assert kwargs["domain_id"] == self.store.domain
        return self.store.root if kwargs["source_id"] == self.store.source.source_id else None

    async def find_submission(self, **kwargs):
        assert self.owned
        await asyncio.sleep(0)
        return [row for row in self.store.jobs.values()
            if row.domain_id == kwargs["domain_id"] and row.actor_id == kwargs["actor_id"]
            and row.job_type == kwargs["job_type"] and row.source_id == kwargs["source_id"]
            and (row.idempotency_key_hash == kwargs["key_hash"] or row.input_hash == kwargs["input_hash"])]

    async def add(self, entity):
        if isinstance(entity, OpsOutputJobEntity):
            self.pending_jobs[entity.job_id] = entity
        else:
            if self.store.fail_event:
                raise RuntimeError("模拟事务事件写入失败")
            self.pending_events.append(entity)
        return entity

    async def get(self, *, domain_id, actor_id, job_id, lock=False):
        row = self.store.jobs.get(job_id)
        if not row or row.domain_id != domain_id or row.actor_id != actor_id:
            return None
        if lock:
            row = deepcopy(row)
            self.pending_jobs[job_id] = row
        return row

    async def emit_output_job_event(self, *, job, event_type, now):
        if self.store.fail_notification:
            raise RuntimeError("模拟通知 Outbox 写入失败")
        self.pending_notifications.append((job.job_id, event_type, now))

    async def list(self, *, domain_id, actor_id, limit, **_):
        return [row for row in self.store.jobs.values()
            if row.domain_id == domain_id and row.actor_id == actor_id][:limit + 1]

    async def flush(self):
        return None

    async def commit(self):
        if self.store.fail_commit:
            raise RuntimeError("模拟提交失败")
        self.store.jobs.update(self.pending_jobs)
        self.store.events.extend(self.pending_events)
        self.store.notifications.extend(self.pending_notifications)


def test_acceptance_freezes_inputs_and_atomically_enqueues_without_generation():
    async def run():
        store = Store()
        receipt = await store.submit()
        assert receipt.status == "QUEUED" and receipt.result is None
        assert receipt.job_id != store.source.source_id
        job = store.jobs[receipt.job_id]
        assert job.input_snapshot_json["artifacts"][0]["artifact_id"] == str(store.artifact.artifact_id)
        assert len(store.events) == 1
        event = store.events[0]
        assert event.aggregate_id == job.job_id and event.event_type == "OPS_OUTPUT_JOB_REQUESTED"
        assert event.payload_json["job_id"] == str(job.job_id)
        assert store.notifications == [(job.job_id, "aiops.output_job.started", NOW)]
        public = receipt.model_dump(mode="json")
        assert not {"input_snapshot_json", "lease_token", "processing_ref_json"} & public.keys()
        assert "request-1" not in str(vars(job))
    asyncio.run(run())


def test_parallel_duplicate_acceptance_uses_one_job_and_rechecks_sources():
    async def run():
        store = Store()
        receipts = await asyncio.gather(*(store.submit() for _ in range(8)))
        assert len({row.job_id for row in receipts}) == 1
        assert len(store.jobs) == len(store.events) == 1
        assert len(store.notifications) == 1
        assert store.freeze_calls == 8
        assert (await store.submit(key="another-click")).job_id == receipts[0].job_id
    asyncio.run(run())


def test_same_key_changed_input_is_rejected_and_different_key_creates_new_job():
    async def run():
        store = Store()
        first = await store.submit()
        store.fingerprint = "c" * 64
        with pytest.raises(AIOpsApplicationError, match="相同幂等键"):
            await store.submit()
        second = await store.submit(key="new-request")
        assert first.job_id != second.job_id and len(store.jobs) == 2
    asyncio.run(run())


def test_enqueue_failure_rolls_back_job_and_receipt():
    async def run():
        store = Store()
        store.fail_event = True
        with pytest.raises(RuntimeError):
            await store.submit()
        assert not store.jobs and not store.events
        store.fail_event = False
        assert (await store.submit()).status == "QUEUED"
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["fail_notification", "fail_commit"])
def test_notification_or_commit_failure_rolls_back_entire_submission(failure):
    async def run():
        store = Store()
        setattr(store, failure, True)
        with pytest.raises(RuntimeError):
            await store.submit()
        assert not store.jobs and not store.events and not store.notifications
        setattr(store, failure, False)
        await store.submit()
        assert len(store.jobs) == len(store.events) == len(store.notifications) == 1
    asyncio.run(run())


def test_retry_notification_failure_does_not_requeue_committed_job():
    async def run():
        store = Store()
        receipt = await store.submit()
        job = store.jobs[receipt.job_id]
        token = claim(job)
        fail_job(job, code="TEMPORARY", message="临时失败", retryable=True,
            now=NOW, owner="worker-a", token=token)
        store.fail_notification = True
        with pytest.raises(RuntimeError):
            await store.service.retry(scope=store.scope, job_id=job.job_id,
                expected_row_version=job.row_version, authorize=AsyncMock())
        assert store.jobs[job.job_id].status == "FAILED"
        assert len(store.events) == len(store.notifications) == 1
    asyncio.run(run())


def test_idempotency_replay_does_not_bypass_current_authorization():
    async def run():
        store = Store()
        await store.submit()
        async def withdrawn(_uow, _root):
            raise AIOpsApplicationError(code="SOURCE_REVOKED", message="来源权限已撤销", status_code=403)
        store.freeze = withdrawn
        with pytest.raises(AIOpsApplicationError, match="撤销"):
            await store.submit()
        assert len(store.jobs) == 1
    asyncio.run(run())


def test_job_reads_are_actor_scoped_and_reauthorize_sources():
    async def run():
        store = Store()
        first = await store.submit()
        authorize = AsyncMock()
        result = await store.service.get(scope=store.scope, job_id=first.job_id, authorize=authorize)
        assert result.job_id == first.job_id
        authorize.assert_awaited_once()
        alien = OutputJobScope(domain_id=store.domain, actor_id="another", tenant_id=store.tenant, trace_id="other")
        with pytest.raises(AIOpsApplicationError):
            await store.service.get(scope=alien, job_id=first.job_id, authorize=authorize)
        assert authorize.await_count == 1
        page = await store.service.list(scope=store.scope, authorize=authorize, limit=1)
        assert len(page.items) == 1 and page.next_cursor is None
    asyncio.run(run())


@pytest.mark.parametrize("value", ["", "x" * 257])
def test_invalid_idempotency_key_is_rejected(value):
    async def run():
        store = Store()
        with pytest.raises(AIOpsApplicationError):
            await store.submit(key=value)
        assert not store.jobs
    asyncio.run(run())


def test_lease_expiry_recovery_fences_out_previous_worker():
    job = fixture_job()
    old = claim(job)
    assert not can_claim(job, now=NOW + timedelta(seconds=1))
    new = uuid7()
    start_attempt(job, now=NOW + timedelta(seconds=31), owner="worker-b",
        token=new, until=NOW + timedelta(seconds=61))
    assert job.attempt_count == 2
    with pytest.raises(OutputJobConflictError):
        require_lease(job, now=NOW + timedelta(seconds=32), owner="worker-a", token=old)
    require_lease(job, now=NOW + timedelta(seconds=32), owner="worker-b", token=new)


def test_success_is_terminal_and_does_not_claim_problem_resolution():
    job = fixture_job()
    token = claim(job)
    complete_job(job, result=OutputJobResult(report_id=uuid7()), now=NOW,
        owner="worker-a", token=token)
    receipt = job_receipt(job)
    assert receipt.status == "SUCCEEDED" and receipt.result.report_id
    assert "health" not in receipt.model_dump() and "resolved" not in receipt.model_dump()
    assert not can_claim(job, now=NOW + timedelta(days=1))
    with pytest.raises(OutputJobConflictError):
        complete_job(job, result=OutputJobResult(report_id=uuid7()), now=NOW,
            owner="worker-a", token=token)


def test_case_success_requires_exact_version_but_does_not_publish():
    job = fixture_job(job_type="CASE_EXTRACTION")
    token = claim(job)
    result = OutputJobResult(asset_id=uuid7(), asset_version_id=uuid7())
    complete_job(job, result=result, now=NOW, owner="worker-a", token=token)
    receipt = job_receipt(job)
    assert receipt.result.asset_version_id == result.asset_version_id
    assert "publication_status" not in receipt.model_dump()


def test_failure_and_retry_preserve_frozen_input_and_attempt_history():
    job = fixture_job(input_snapshot_json={"frozen": "unchanged"})
    token = claim(job)
    job.stage = "INGESTING"
    fail_job(job, code="KC_UNAVAILABLE", message="案例处理暂不可用", retryable=True,
        now=NOW, owner="worker-a", token=token)
    assert job_receipt(job).retryable
    retry_job(job, expected_version=1, now=NOW)
    assert job.status == "QUEUED" and job.stage == "INGESTING"
    assert job.attempt_count == 1 and job.input_snapshot_json == {"frozen": "unchanged"}
    assert job.completed_at is None and job.lease_token is None


@pytest.mark.parametrize("change", [{"retryable": False}, {"row_version": 2}, {"status": "SUCCEEDED"}])
def test_illegal_retries_do_not_change_state(change):
    job = fixture_job(status="FAILED", stage="FAILED", retryable=True, **{
        key: val for key, val in change.items() if key not in {"status", "retryable"}
    })
    for key, value in change.items():
        setattr(job, key, value)
    with pytest.raises(OutputJobConflictError):
        retry_job(job, expected_version=1, now=NOW)


@pytest.mark.parametrize("data", [
    {"asset_id": uuid7()}, {"asset_version_id": uuid7()},
    {"report_id": uuid7(), "asset_id": uuid7(), "asset_version_id": uuid7()}, {},
])
def test_result_locator_rejects_incomplete_or_mixed_outputs(data):
    with pytest.raises(ValidationError):
        OutputJobResult.model_validate(data)


def test_contract_rejects_success_without_result_or_wrong_result_type():
    receipt = job_receipt(fixture_job()).model_dump()
    receipt.update(status="SUCCEEDED", stage="COMPLETED", completed_at=NOW)
    with pytest.raises(ValidationError):
        OutputJobReceipt.model_validate(receipt)
    receipt["result"] = {"asset_id": uuid7(), "asset_version_id": uuid7()}
    with pytest.raises(ValidationError):
        OutputJobReceipt.model_validate(receipt)


def test_entities_have_no_business_uniqueness_and_attempts_reference_job():
    for entity in (OpsOutputJobEntity, OpsOutputJobAttemptEntity):
        assert not any(constraint.__class__.__name__ in {"UniqueConstraint", "CheckConstraint"}
            for constraint in entity.__table__.constraints)
        assert not any(index.unique for index in entity.__table__.indexes)
    foreign_key = next(iter(OpsOutputJobAttemptEntity.__table__.c.job_id.foreign_keys))
    assert foreign_key.column is OpsOutputJobEntity.__table__.c.job_id


def test_repository_queries_keep_domain_actor_filters_and_do_not_commit():
    async def run():
        session = SimpleNamespace(
            execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: None)),
            scalars=AsyncMock(return_value=SimpleNamespace(all=lambda: [])),
        )
        guard = Mock()
        repo = OutputJobRepository(session, guard)
        await repo.get(domain_id=1, actor_id="actor", job_id=uuid7(), lock=True)
        statement = session.execute.call_args.args[0]
        for dialect in (oracle.dialect(), postgresql.dialect()):
            sql = str(statement.compile(dialect=dialect)).lower()
            assert "domain_id =" in sql and "actor_id =" in sql and "for update" in sql
        await repo.list(domain_id=1, actor_id="actor", limit=5, status="FAILED", before_id=uuid7())
        values = session.scalars.call_args.args[0].compile().params.values()
        assert "FAILED" in values and "actor" in values and 6 in values
        guard.assert_called()
    asyncio.run(run())


def test_output_repository_is_owned_and_closed_by_single_commit_uow():
    async def run():
        transaction = SimpleNamespace(commit=AsyncMock())
        session = SimpleNamespace(
            begin=AsyncMock(return_value=transaction), flush=AsyncMock(),
            rollback=AsyncMock(), close=AsyncMock(),
        )
        uow = AIOpsUnitOfWork(lambda: session)
        assert uow.output_jobs is None
        async with uow:
            assert isinstance(uow.output_jobs, OutputJobRepository)
            await uow.commit()
        assert uow.output_jobs is None
        transaction.commit.assert_awaited_once()
        session.close.assert_awaited_once()
    asyncio.run(run())
