"""输出任务终态与真实案例处理组合回归；替身事务不冒充实库验收。"""

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.output_finalization import OutputJobFinalizer
from aiops_agent.playbooks import canonical_hash
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_conversation_case_output import fixture, advance, execute


NOW = datetime(2026, 11, 3, 2, tzinfo=timezone.utc)


def job_for(f, *, retryable=False):
    snapshot = f.frozen.model_dump(mode="json")
    return SimpleNamespace(
        job_id=f.execution.job_id, job_type="CASE_EXTRACTION", domain_id=f.domain,
        actor_id=f.actor, tenant_id=f.scope.tenant_id, retryable=retryable,
        processing_ref_json=deepcopy(f.ref), input_snapshot_json=snapshot,
        input_hash=canonical_hash({key: value for key, value in snapshot.items() if key != "frozen_at"}),
        source_json=snapshot["source"],
    )


async def ready():
    f = await fixture()
    await advance(f)
    f.phase = "READY"
    output = await advance(f)
    f.material = output.finalization
    f.job = job_for(f)
    f.finalizer = OutputJobFinalizer()
    f.uow.platform_notifications.emit_output_job_event = AsyncMock()
    f.uow.flush = AsyncMock()
    f.uow.commit.reset_mock()
    return f


async def succeed(f):
    result = await f.finalizer.succeeded(
        uow=f.uow, job=f.job, execution=f.execution, authority=f.authority,
        material=f.material, now=NOW)
    f.uow.platform_notifications.emit_output_job_event.assert_not_awaited()
    f.job.status = "SUCCEEDED"
    await f.finalizer.notify_succeeded(uow=f.uow, job=f.job, result=result, now=NOW)
    return result


def test_success_rechecks_exact_case_and_only_emits_draft_notification_without_commit():
    async def run():
        f = await ready()
        result = await succeed(f)
        version = next(iter(f.state.versions.values()))
        asset = f.state.assets[version.asset_id]
        assert result.asset_version_id == version.asset_version_id
        assert version.status == "DRAFT" and asset.current_version_id is None
        args = f.uow.platform_notifications.emit_output_job_event.await_args.kwargs
        assert args["event_type"] == "aiops.case.draft_ready" and args["now"] == NOW
        assert args["result"] == result
        assert f.uow.operations_knowledge.get_version.await_args.kwargs["for_update"] is True
        assert f.uow.operations_knowledge.get_asset.await_args.kwargs["for_update"] is True
        f.uow.commit.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("mutation", [
    "report_result", "version_changed", "asset_changed", "index_changed",
    "projection_changed", "source_changed", "withdrawn", "source_revoked", "retired",
])
def test_success_refuses_stale_or_invalid_result_without_notification(mutation):
    async def run():
        f = await ready()
        version = next(iter(f.state.versions.values()))
        asset = f.state.assets[version.asset_id]
        index = f.state.relations[version.asset_version_id]["indexes"][0]
        if mutation == "report_result":
            f.material["case_result"] = {"report_id": str(uuid7())}
        if mutation == "version_changed":
            version.row_version += 1
        if mutation == "asset_changed":
            asset.row_version += 1
        if mutation == "index_changed":
            index.document_version_id = uuid7()
        if mutation == "projection_changed":
            f.material["index_stamp"]["body_sha256"] = "f" * 64
        if mutation == "source_changed":
            version.profile_json["source"]["source_fingerprint"] = "f" * 64
        if mutation == "withdrawn":
            f.uow.outbox.get_by_idempotency.return_value = SimpleNamespace(status="PENDING")
        if mutation == "source_revoked":
            f.uow.agents.version_has_target.return_value = False
        if mutation == "retired":
            version.status = "RETIRED"
        with pytest.raises(AIOpsApplicationError):
            await succeed(f)
        f.uow.platform_notifications.emit_output_job_event.assert_not_awaited()
        f.uow.commit.assert_not_awaited()
    asyncio.run(run())


async def accepted_before_registry(*, retryable=False):
    f = await fixture()
    saved = {}
    async def crash(recovery):
        saved.update(recovery.model_dump(mode="json"))
        raise RuntimeError("模拟外部已受理而 Registry 尚未登记")
    with pytest.raises(RuntimeError):
        await execute(f, record_receipt=crash)
    assert len(f.state.versions) == 1
    f.ref = {"closure_artifact_id": str(f.prepared.closure_artifact_id), "intake": saved}
    f.job = job_for(f, retryable=retryable)
    # 执行快照早于检查点，失败必须读取任务当前恢复定位。
    assert not f.execution.processing_ref
    f.finalizer = OutputJobFinalizer()
    f.uow.platform_notifications.emit_output_job_event = AsyncMock()
    f.uow.flush = AsyncMock()
    f.uow.commit.reset_mock()
    return f


@pytest.mark.parametrize("retryable", [False, True])
def test_failure_records_accepted_revision_and_distinguishes_revocation_from_transient_exhaustion(retryable):
    async def run():
        f = await accepted_before_registry(retryable=retryable)
        f.uow.agents.version_has_target.return_value = False
        await f.finalizer.failed(uow=f.uow, job=f.job, execution=f.execution, now=NOW)
        assert len(f.state.versions) == 1
        version = next(iter(f.state.versions.values()))
        relations = f.state.relations[version.asset_version_id]
        assert str(relations["indexes"][0].document_version_id) == str(next(iter(f.documents.records)))
        assert version.status == ("PROCESSING" if retryable else "REVIEW_REQUIRED")
        events = [call.args[0] for call in f.uow.outbox.add.await_args_list]
        assert any(row.event_type == "OPS_CASE_WITHDRAWAL_REQUESTED" for row in events) is not retryable
        assert f.uow.operations_knowledge.get_version.await_args.kwargs["for_update"] is True
        assert f.uow.operations_knowledge.get_asset.await_args.kwargs["for_update"] is True
        assert f.uow.platform_notifications.emit_output_job_event.await_args.kwargs["event_type"] == "aiops.output_job.failed"
        f.documents.publish.assert_not_awaited()
        f.uow.commit.assert_not_awaited()
    asyncio.run(run())


def test_failure_cleanup_cannot_withdraw_a_version_published_before_lock_acquisition():
    async def run():
        f = await accepted_before_registry()
        original = f.uow.operations_knowledge.get_version.side_effect
        def publish_then_lock(**kwargs):
            version = original(**kwargs)
            assert kwargs["for_update"] is True
            version.status = "PUBLISHED"
            f.state.assets[version.asset_id].current_version_id = version.asset_version_id
            return version
        f.uow.operations_knowledge.get_version.side_effect = publish_then_lock
        await f.finalizer.failed(uow=f.uow, job=f.job, execution=f.execution, now=NOW)
        version = next(iter(f.state.versions.values()))
        assert version.status == "PUBLISHED"
        assert not any(call.args[0].event_type == "OPS_CASE_WITHDRAWAL_REQUESTED"
            for call in f.uow.outbox.add.await_args_list)
        f.uow.platform_notifications.emit_output_job_event.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize("corruption", ["seal", "receipt", "artifact_id"])
def test_failure_with_corrupt_recovery_still_notifies_but_never_creates_registry(corruption):
    async def run():
        f = await accepted_before_registry()
        before_versions = set(f.state.versions)
        if corruption == "seal":
            f.job.input_hash = "f" * 64
        if corruption == "receipt":
            f.job.processing_ref_json["intake"] = {"invalid": True}
        if corruption == "artifact_id":
            f.job.processing_ref_json["closure_artifact_id"] = "invalid"
        await f.finalizer.failed(uow=f.uow, job=f.job, execution=f.execution, now=NOW)
        assert set(f.state.versions) == before_versions
        f.uow.platform_notifications.emit_output_job_event.assert_awaited_once()
    asyncio.run(run())


def test_real_finalizer_notification_failure_rolls_back_worker_terminal_state():
    from aiops_agent.workers.output_jobs import OutputJobWorker
    from tests.unit.aiops_agent.test_aiops_output_job_worker import Store, MemoryUow

    async def run():
        f = await ready()
        store = Store()
        store.job.job_id = f.execution.job_id
        store.job.job_type, store.job.domain_id, store.job.actor_id, store.job.tenant_id = (
            "CASE_EXTRACTION", f.domain, f.actor, f.scope.tenant_id)
        store.job.source_json = f.job.input_snapshot_json["source"]
        store.job.input_snapshot_json = f.job.input_snapshot_json
        store.job.input_hash = f.job.input_hash
        store.job.processing_ref_json = deepcopy(f.ref)
        store.authority.resolve.side_effect = lambda _: f.authority
        store.processor.advance.side_effect = None
        from aiops_agent.workers.output_jobs import OutputAdvance
        store.processor.advance.return_value = OutputAdvance(
            stage="FINALIZING", processing_ref=f.ref, finalization=f.material)
        broken = True

        class TransactionUow(MemoryUow):
            def __init__(self, owner):
                super().__init__(owner)
                self.operations_knowledge = f.uow.operations_knowledge
                self.agents, self.targets, self.turns = f.uow.agents, f.uow.targets, f.uow.turns
                self.conversations = f.uow.conversations
                self.outbox = f.uow.outbox
                self.runs = SimpleNamespace(**vars(f.uow.runs))
                self.runs.database_now = AsyncMock(side_effect=lambda: store.now)
                self.platform_notifications = SimpleNamespace(emit_output_job_event=self.notify)

            async def notify(self, *, event_type, job, **_):
                if broken:
                    raise RuntimeError("模拟通知 Outbox 不可写")
                self.notifications.append((event_type, job.job_id))

        worker = OutputJobWorker(uow_factory=lambda: TransactionUow(store),
            authority=store.authority, processor=store.processor,
            finalizer=f.finalizer, worker_id="worker-a")
        await worker.run_once()
        assert store.job.status == "RUNNING" and store.job.result_json is None
        assert not store.notifications
        broken = False
        store.now = store.job.available_at
        await worker.run_once()
        assert store.job.status == "SUCCEEDED"
        assert store.notifications == [("aiops.case.draft_ready", store.job.job_id)]
        assert not await worker.run_once()
        assert len(f.state.versions) == 1
    asyncio.run(run())
