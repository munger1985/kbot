"""输出 Worker 协议回归；内存事务用于故障注入，不代表实库恢复验收。"""

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.domain.output_jobs import can_claim, start_attempt
from aiops_agent.playbooks import canonical_hash
from aiops_agent.workers.outbox_dispatcher import AIOpsDomainOutboxSink
from aiops_agent.workers.output_jobs import OutputAdvance, OutputJobWorker
from platform_core.contracts.aiops.output_jobs import FrozenOutputInput, OutputJobResult, OutputJobSource, OutputSourceArtifact
from platform_core.identity import uuid7


NOW = datetime(2026, 11, 3, 2, tzinfo=timezone.utc)


class Store:
    def __init__(self):
        self.now = NOW
        self.source = OutputJobSource(kind="CONVERSATION", source_id=uuid7())
        frozen = FrozenOutputInput(source=self.source, source_fingerprint="a" * 64,
            artifacts=(OutputSourceArtifact(run_id=uuid7(), artifact_id=uuid7(),
                schema_version="AIOPS_TURN_RESULT.v1", content_hash="b" * 64, security_level=2),),
            parameters={"title": "提交时标题"}, frozen_at=NOW)
        snapshot = frozen.model_dump(mode="json")
        self.job = SimpleNamespace(
            job_id=uuid7(), job_type="REPORT_GENERATION", domain_id=7, tenant_id=None, actor_id="author",
            source_json=self.source.model_dump(mode="json"), input_snapshot_json=snapshot,
            input_hash=canonical_hash({k: v for k, v in snapshot.items() if k != "frozen_at"}),
            trace_id="trace", status="QUEUED", stage="ACCEPTED", processing_ref_json=None,
            result_json=None, available_at=NOW, submitted_at=NOW, attempt_count=0,
            started_at=None, completed_at=None, updated_at=NOW, heartbeat_at=None,
            lease_token=None, lease_until=None, lease_owner=None, error_code=None,
            error_message=None, retryable=False, failure_stage=None,
            row_version=1,
        )
        self.lock = asyncio.Lock()
        self.attempts, self.outputs, self.notifications = [], [], []
        self.active_uows = 0
        self.authority = SimpleNamespace(resolve=AsyncMock(side_effect=self.authorize))
        self.processor = SimpleNamespace(advance=AsyncMock(side_effect=self.advance))
        self.finalizer = SimpleNamespace(succeeded=AsyncMock(side_effect=self.succeeded),
            failed=AsyncMock(side_effect=self.failed),
            notify_succeeded=AsyncMock(side_effect=self.notify_succeeded))
        self.report_id = uuid7()

    def worker(self, **changes):
        return OutputJobWorker(uow_factory=lambda: MemoryUow(self), authority=self.authority,
            processor=self.processor, finalizer=self.finalizer, worker_id="worker-a", **changes)

    async def authorize(self, execution):
        assert self.active_uows == 0
        return {"actor_id": execution.scope.actor_id, "permissions": ["aiops:report:generate"]}

    async def advance(self, execution, *, authority, checkpoint, record_receipt):
        assert self.active_uows == 0 and authority["actor_id"] == "author"
        assert execution.frozen.parameters["title"] == "提交时标题"
        return OutputAdvance(stage="FINALIZING", processing_ref={}, finalization={"title": "已完成"})

    async def succeeded(self, *, uow, job, execution, authority, material, now):
        assert self.active_uows == 1
        uow.outputs.append(material)
        return OutputJobResult(report_id=self.report_id, report_version="1")

    async def notify_succeeded(self, *, uow, job, result, now):
        assert job.status == "SUCCEEDED" and job.result_json == result.model_dump(mode="json")
        uow.notifications.append(("ready", job.job_id))

    async def failed(self, *, uow, job, execution, now):
        assert self.active_uows == 1
        uow.notifications.append(("failed", job.job_id))


class MemoryUow:
    def __init__(self, store):
        self.store = store
        self.runs = SimpleNamespace(database_now=AsyncMock(side_effect=lambda: self.store.now))
        self.output_jobs = self

    async def __aenter__(self):
        await self.store.lock.acquire()
        self.store.active_uows += 1
        self.job = deepcopy(self.store.job)
        self.attempts = deepcopy(self.store.attempts)
        self.outputs = deepcopy(self.store.outputs)
        self.notifications = deepcopy(self.store.notifications)
        return self

    async def __aexit__(self, *_):
        self.store.active_uows -= 1
        self.store.lock.release()

    async def commit(self):
        self.store.job = deepcopy(self.job)
        self.store.attempts = deepcopy(self.attempts)
        self.store.outputs = deepcopy(self.outputs)
        self.store.notifications = deepcopy(self.notifications)

    async def flush(self):
        self.job.row_version += 1

    async def claim(self, *, now, owner, token, until):
        if not can_claim(self.job, now=now):
            return None
        if self.job.lease_token:
            await self.finish_attempt(job_id=self.job.job_id, token=self.job.lease_token,
                status="EXPIRED", stage=self.job.stage, now=now)
        start_attempt(self.job, now=now, owner=owner, token=token, until=until)
        self.attempts.append(SimpleNamespace(lease_token=token, status="RUNNING", stage=self.job.stage))
        return self.job

    async def get_for_worker(self, *, job_id):
        return self.job if self.job.job_id == job_id else None

    async def get(self, *, job_id, domain_id, actor_id, **_):
        if job_id == self.job.job_id and domain_id == self.job.domain_id and actor_id == self.job.actor_id:
            return self.job

    async def list_attempts(self, **_):
        return self.attempts

    async def finish_attempt(self, *, token, status, stage, **_):
        for attempt in self.attempts:
            if attempt.lease_token == token and attempt.status == "RUNNING":
                attempt.status, attempt.stage = status, stage


def test_terminal_result_and_notification_are_atomic_and_replay_does_not_duplicate():
    async def run():
        store = Store()
        worker = store.worker()
        payload = {"job_id": str(store.job.job_id), "domain_id": "7", "actor_id": "author"}
        await worker.accept_requested(payload)
        assert await worker.run_once()
        assert store.job.status == "SUCCEEDED" and store.job.lease_token is None
        assert store.job.result_json["report_id"] == str(store.report_id)
        assert store.outputs == [{"title": "已完成"}]
        assert store.notifications == [("ready", store.job.job_id)]
        assert [a.status for a in store.attempts] == ["SUCCEEDED"]
        assert store.authority.resolve.await_count == 2
        await worker.accept_requested(payload)
        assert not await worker.run_once()
        assert len(store.outputs) == len(store.notifications) == 1
    asyncio.run(run())


def test_waiting_index_is_not_success_and_resumes_exact_external_revision():
    async def run():
        store = Store()
        store.job.job_type = "CASE_EXTRACTION"
        revision_id, version_id, asset_id = str(uuid7()), uuid7(), uuid7()
        async def advance(execution, *, authority, checkpoint, record_receipt):
            if not execution.processing_ref:
                await checkpoint(stage="INGESTING", processing_ref={"revision_id": revision_id})
                return OutputAdvance(stage="INDEXING", processing_ref={"revision_id": revision_id}, check_after_seconds=10)
            assert execution.processing_ref == {"revision_id": revision_id}
            return OutputAdvance(stage="FINALIZING", processing_ref=execution.processing_ref,
                finalization={"asset_id": str(asset_id), "version_id": str(version_id)})
        async def finish(**kwargs):
            kwargs["uow"].notifications.append(("draft", kwargs["job"].job_id))
            return OutputJobResult(asset_id=asset_id, asset_version_id=version_id)
        store.processor.advance.side_effect = advance
        store.finalizer.succeeded.side_effect = finish
        worker = store.worker()
        await worker.run_once()
        assert store.job.status == "RUNNING" and store.job.stage == "INDEXING"
        assert store.job.result_json is None and not store.notifications
        assert store.job.processing_ref_json == {"revision_id": revision_id}
        assert store.job.lease_until is None and not await worker.run_once()
        store.now += timedelta(seconds=10)
        await worker.run_once()
        assert store.job.status == "SUCCEEDED" and store.job.result_json["asset_version_id"] == str(version_id)
        assert [a.status for a in store.attempts] == ["WAITING", "SUCCEEDED"]
    asyncio.run(run())


@pytest.mark.parametrize("when", ["before", "after"])
def test_current_authority_revocation_fails_closed_without_leaking_raw_error(when):
    async def run():
        store = Store()
        denied = AIOpsApplicationError(code="APP_PERMISSION_DENIED", message="private raw upstream secret",
            status_code=403, retryable=False)
        store.authority.resolve.side_effect = denied if when == "before" else [{"actor_id": "author"}, denied]
        await store.worker().run_once()
        assert store.job.status == "FAILED" and not store.job.retryable
        assert store.job.error_code == "APP_PERMISSION_DENIED"
        assert "private" not in store.job.error_message
        assert not store.outputs and store.notifications == [("failed", store.job.job_id)]
        assert store.processor.advance.await_count == (0 if when == "before" else 1)
        store.finalizer.succeeded.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("corruption", ["hash", "shape", "source"])
def test_corrupted_frozen_input_never_reaches_models_or_authority(corruption):
    async def run():
        store = Store()
        if corruption == "hash":
            store.job.input_snapshot_json["parameters"]["title"] = "被修改"
        elif corruption == "shape":
            store.job.input_snapshot_json["artifacts"] = []
        else:
            store.job.source_json["source_id"] = str(uuid7())
        await store.worker().run_once()
        assert store.job.status == "FAILED" and not store.job.retryable
        assert store.job.error_code == "AIOPS_OUTPUT_JOB_INPUT_INVALID"
        store.authority.resolve.assert_not_awaited()
        store.processor.advance.assert_not_awaited()
    asyncio.run(run())


def test_transient_failures_are_bounded_and_keep_checkpointed_recovery_locator():
    async def run():
        store = Store()
        revision = str(uuid7())
        async def broken(execution, *, authority, checkpoint, record_receipt):
            await checkpoint(stage="INDEXING", processing_ref={"revision_id": revision})
            raise TimeoutError("不可记录的完整连接串")
        store.processor.advance.side_effect = broken
        worker = store.worker(max_failures=3)
        for index in range(3):
            await worker.run_once()
            assert store.job.processing_ref_json == {"revision_id": revision}
            assert store.job.status == ("FAILED" if index == 2 else "RUNNING")
            store.now += timedelta(seconds=60)
        assert store.job.retryable and store.job.failure_stage == "INDEXING"
        assert [a.status for a in store.attempts] == ["RETRY_SCHEDULED", "RETRY_SCHEDULED", "FAILED"]
        assert store.notifications == [("failed", store.job.job_id)]
        assert "连接串" not in store.job.error_message and worker.healthy
    asyncio.run(run())


def test_notification_failure_rolls_back_result_and_success_before_retry():
    async def run():
        store = Store()
        async def fail(**kwargs):
            kwargs["uow"].outputs.append({"should": "rollback"})
            raise RuntimeError("模拟通知 Outbox 事务失败")
        store.finalizer.succeeded.side_effect = fail
        await store.worker().run_once()
        assert store.job.status == "RUNNING" and store.job.result_json is None
        assert not store.outputs and not store.notifications
        assert store.attempts[-1].status == "RETRY_SCHEDULED"
    asyncio.run(run())


def test_expired_worker_is_cancelled_and_cannot_publish_after_lease_takeover():
    async def run():
        store = Store()
        entered, cancelled = asyncio.Event(), asyncio.Event()
        async def slow(*_, **__):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        store.processor.advance.side_effect = slow
        worker = store.worker(lease_seconds=0.2, heartbeat_seconds=0.01)
        task = asyncio.create_task(worker.run_once())
        await asyncio.wait_for(entered.wait(), timeout=1)
        async with MemoryUow(store) as uow:
            uow.job.lease_token = uuid7()
            uow.job.lease_owner = "worker-b"
            await uow.commit()
        await asyncio.wait_for(task, timeout=1)
        assert cancelled.is_set() and not store.outputs and not store.notifications
        assert store.job.status == "RUNNING" and store.job.lease_owner == "worker-b"
    asyncio.run(run())


def test_process_cancellation_preserves_claim_and_next_worker_recovers_after_expiry():
    async def run():
        store = Store()
        entered = asyncio.Event()
        async def slow(*_, **__):
            entered.set()
            await asyncio.Event().wait()
        store.processor.advance.side_effect = slow
        worker = store.worker()
        task = asyncio.create_task(worker.run_once())
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert store.job.status == "RUNNING" and not store.notifications
        assert store.attempts[-1].status == "RUNNING"
        store.now += timedelta(seconds=121)
        store.processor.advance.side_effect = store.advance
        await store.worker().run_once()
        assert store.job.status == "SUCCEEDED"
        assert [a.status for a in store.attempts] == ["EXPIRED", "SUCCEEDED"]
    asyncio.run(run())


def test_index_wait_timeout_finishes_without_holding_external_processing_forever():
    async def run():
        store = Store()
        store.now += timedelta(days=2)
        await store.worker().run_once()
        assert store.job.status == "FAILED" and not store.job.retryable
        assert store.job.error_code == "AIOPS_OUTPUT_JOB_RECOVERY_EXHAUSTED"
        store.processor.advance.assert_not_awaited()
    asyncio.run(run())


def test_external_stage_is_cancelled_when_total_execution_deadline_expires():
    async def run():
        store = Store()
        cancelled = asyncio.Event()
        async def slow(*_, **__):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        store.processor.advance.side_effect = slow
        await store.worker(max_age_seconds=0.01).run_once()
        assert cancelled.is_set() and store.job.status == "FAILED"
        assert store.job.error_code == "AIOPS_OUTPUT_JOB_RECOVERY_EXHAUSTED"
        assert not store.job.retryable and not store.outputs
    asyncio.run(run())


@pytest.mark.parametrize("changes", [
    {"job_id": str(uuid7())}, {"domain_id": "8"}, {"actor_id": "another"}, {"tenant_id": str(uuid7())},
])
def test_outbox_wakeup_cannot_substitute_foreign_job_scope(changes):
    async def run():
        store = Store()
        payload = {"job_id": str(store.job.job_id), "domain_id": "7", "actor_id": "author", **changes}
        with pytest.raises(ValueError):
            await store.worker().accept_requested(payload)
        assert store.job.status == "QUEUED" and not store.outputs
    asyncio.run(run())


def test_missing_output_consumer_never_falls_back_to_logging_success():
    async def run():
        fallback = SimpleNamespace(publish=AsyncMock())
        sink = AIOpsDomainOutboxSink(runtime_service=None, fallback=fallback)
        with pytest.raises(RuntimeError, match="消费者未配置"):
            await sink.publish("OPS_OUTPUT_JOB_REQUESTED", {"job_id": str(uuid7())})
        fallback.publish.assert_not_awaited()
        consumer = SimpleNamespace(accept_requested=AsyncMock())
        sink = AIOpsDomainOutboxSink(runtime_service=None, fallback=fallback, output_job_worker=consumer)
        payload = {"job_id": str(uuid7())}
        await sink.publish("OPS_OUTPUT_JOB_REQUESTED", payload)
        consumer.accept_requested.assert_awaited_once_with(payload)
        fallback.publish.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("changes", [{"authority": None}, {"processor": None}, {"finalizer": None},
    {"heartbeat_seconds": 120}, {"interval_seconds": 120}])
def test_worker_cannot_start_with_missing_dependencies_or_invalid_lease_policy(changes):
    store = Store()
    config = dict(uow_factory=lambda: MemoryUow(store), authority=store.authority,
        processor=store.processor, finalizer=store.finalizer, worker_id="worker-a", **{})
    config.update(changes)
    with pytest.raises(ValueError):
        OutputJobWorker(**config)

def test_external_receipt_is_durable_even_when_current_authority_is_lost():
    async def run():
        store = Store()
        async def process(execution, *, authority, checkpoint, record_receipt):
            await checkpoint(stage="INGESTING", processing_ref={"body_hash": "a" * 64})
            store.authority.resolve.side_effect = AIOpsApplicationError(
                code="AUTHORITY_REVOKED", message="撤权", status_code=403, retryable=False)
            await record_receipt(processing_ref={"body_hash": "a" * 64, "revision_id": "exact"})
            return OutputAdvance(stage="INDEXING", processing_ref={"revision_id": "exact"})
        store.processor.advance.side_effect = process
        await store.worker().run_once()
        assert store.job.status == "FAILED" and store.job.result_json is None
        assert store.job.processing_ref_json["revision_id"] == "exact"
        assert store.notifications == [("failed", store.job.job_id)]
        store.finalizer.succeeded.assert_not_awaited()
    asyncio.run(run())


def test_receipt_recording_still_rejects_an_expired_worker_lease():
    async def run():
        store = Store()
        async def process(execution, *, authority, checkpoint, record_receipt):
            store.job.lease_token = uuid7()
            await record_receipt(processing_ref={"revision_id": "must-not-write"})
            raise AssertionError("过期租约不应继续")
        store.processor.advance.side_effect = process
        await store.worker().run_once()
        assert store.job.processing_ref_json is None
        assert not store.notifications
        store.finalizer.succeeded.assert_not_awaited()
    asyncio.run(run())
