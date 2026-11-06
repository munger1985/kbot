"""报告案例真实异步入口、KC 恢复和确切草稿终态；不冒充实库验收。"""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from aiops_agent.api import operations_knowledge as knowledge_api
from aiops_agent.application.conversation_case_output import CaseExecutionOptions, CaseOutputProcessor
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.output_finalization import OutputJobFinalizer
from aiops_agent.application.output_job_sources import OutputJobSourceValidator
from aiops_agent.application.output_jobs import OutputJobScope, OutputJobService
from aiops_agent.application.report_case_output import (
    execute_prepared_report_case, freeze_report_case_input, prepare_report_case,
    report_case_from_frozen_input, submit_report_case, validate_prepared_report_case,
)
from aiops_agent.playbooks import canonical_hash
from aiops_agent.workers.output_jobs import OutputExecution
from platform_core.contracts.aiops.output_jobs import OutputJobSource
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_output_job_api import api, human_context
from tests.unit.aiops_agent.test_aiops_output_jobs import Store, MemoryUow
from tests.unit.aiops_agent.test_aiops_report_source_revision import revision_fixture, edit


async def fixture():
    f = await revision_fixture()
    f.store = Store()
    if not f.store.tenant:
        f.domain, f.actor = f.store.domain, str(f.actor)
        f.run.domain_id, f.run.actor_id, f.report.domain_id = f.domain, f.actor, f.domain
    f.store.domain, f.store.actor = f.domain, f.actor
    f.scope = f.store.scope = OutputJobScope(domain_id=f.domain, actor_id=f.actor,
        tenant_id=f.store.tenant, trace_id="report-case")
    f.store.source = OutputJobSource(kind="REPORT", source_id=f.report.report_id)
    f.store.root = f.report
    if f.store.tenant:
        f.report.tenant_id = f.store.tenant
    f.uow.platform_notifications.emit_output_job_event = AsyncMock()
    f.uow.flush = AsyncMock()
    def unit():
        value = MemoryUow(f.store)
        for name in ("runs", "targets", "agents", "turns", "conversations", "inspections", "operations_knowledge"):
            setattr(value, name, getattr(f.uow, name))
        return value
    f.jobs = f.store.service = OutputJobService(uow_factory=unit)
    repo = f.uow.operations_knowledge
    def find_hash(**kw):
        rows = [(f.state.assets[v.asset_id], v) for v in f.state.versions.values()
            if v.source_hash == kw["source_hash"] and v.status != "RETIRED"]
        return max(rows, key=lambda row: row[1].version_no) if rows else None
    def find_report(**kw):
        rows = [(f.state.assets[v.asset_id], v) for v in f.state.versions.values()
            if any(row.source_kind == "REPORT" and row.source_report_id == kw["report_id"]
                for row in f.state.relations[v.asset_version_id]["sources"])]
        return max(rows, key=lambda row: row[1].version_no) if rows else None
    def find_revision(**kw):
        return next(((f.state.assets[v.asset_id], v) for v in f.state.versions.values()
            if any(index.document_version_id == kw["revision_id"]
                for index in f.state.relations[v.asset_version_id]["indexes"])), None)
    repo.find_by_source_hash.side_effect = find_hash
    repo.find_case_by_report_id.side_effect = find_report
    repo.find_case_by_intake_revision.side_effect = find_revision
    original = repo.add_asset_version.side_effect
    async def store(**kw):
        await original(**kw)
        f.state.relations[kw["version"].asset_version_id].update(scopes=kw["scopes"], reviews=[])
    repo.add_asset_version.side_effect = store
    async def add_version(**kw):
        await store(asset=f.state.assets[kw["version"].asset_id], **kw)
    repo.add_version = AsyncMock(side_effect=add_version)
    f.options = CaseExecutionOptions(context=SimpleNamespace(principal_kind="SERVICE", principal_id=uuid7()),
        entity_scope={"tenant_id": f.store.tenant} if f.store.tenant else {},
        kc_scope={"tenant_id": f.store.tenant} if f.store.tenant else {"domain_id": f.domain},
        model_catalog=f.service._model_catalog, model_scope={"tenant_id": str(f.store.tenant)} if f.store.tenant else {},
        technical_name_field="model_name" if f.store.tenant else "served_model_name")
    f.processor = CaseOutputProcessor(knowledge_service=f.service, options_factory=lambda *_: f.options)
    f.finalizer = OutputJobFinalizer()
    f.authority = SimpleNamespace(request=SimpleNamespace(source_agent_ids=(f.run.agent_id,)))
    f.ref, f.phase = {}, "PROCESSING"
    async def checkpoint(*, stage, processing_ref):
        f.ref = deepcopy(processing_ref)
    async def receipt(*, processing_ref):
        f.ref = deepcopy(processing_ref)
    f.checkpoint, f.receipt = AsyncMock(side_effect=checkpoint), AsyncMock(side_effect=receipt)
    await freeze(f)
    return f


async def freeze(f):
    f.prepared = await prepare_report_case(uow=f.uow, domain_id=f.domain,
        report_id=f.report.report_id, actor_id=f.actor)
    f.frozen = await freeze_report_case_input(prepared=f.prepared, uow=f.uow)
    f.execution = OutputExecution(uuid7(), "CASE_EXTRACTION", f.scope, {}, uuid7(), "ACCEPTED", f.frozen)


async def advance(f):
    execution = OutputExecution(f.execution.job_id, "CASE_EXTRACTION", f.scope,
        deepcopy(f.ref), uuid7(), "INDEXING" if f.ref else "ACCEPTED", f.frozen)
    return await f.processor.advance(execution, authority=f.authority,
        checkpoint=f.checkpoint, record_receipt=f.receipt)


def job_for(f, *, retryable=False):
    snapshot = f.frozen.model_dump(mode="json")
    return SimpleNamespace(job_id=f.execution.job_id, job_type="CASE_EXTRACTION",
        domain_id=f.domain, actor_id=f.actor, retryable=retryable, processing_ref_json=deepcopy(f.ref),
        input_snapshot_json=snapshot, source_json=snapshot["source"],
        input_hash=canonical_hash({key: value for key, value in snapshot.items() if key != "frozen_at"}))


def test_report_case_http_is_quick_persisted_idempotent_202_without_model_or_kc():
    async def run():
        f = await fixture()
        app = api(f.store, scopes=("aiops.run", "aiops.manage"))
        app.include_router(knowledge_api.router)
        app.state.operations_knowledge_service = f.service
        path = f"/internal/v1/aiops/operations-knowledge/reports/{f.report.report_id}:extract-case"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://aiops.test") as client:
            assert (await client.post(path)).status_code == 428
            assert not f.store.jobs
            first = await client.post(path, headers={"Idempotency-Key": "report-case-click"})
            assert first.status_code == 202, first.text
            receipt = first.json()
            assert receipt["status"] == "QUEUED" and receipt["result"] is None
            assert receipt["source"]["kind"] == "REPORT"
            assert UUID(receipt["job_id"]) in f.store.jobs
            assert (await client.post(path, headers={"Idempotency-Key": "report-case-click"})).json()["job_id"] == receipt["job_id"]
            assert len(f.store.jobs) == len(f.store.events) == len(f.store.notifications) == 1
            f.artifact.content_hash = "f" * 64
            assert (await client.post(path, headers={"Idempotency-Key": "report-case-click"})).status_code == 422
        f.service._structured_model.generate_structured.assert_not_awaited()
        f.documents.stage.assert_not_awaited()
    asyncio.run(run())


def test_report_case_public_and_client_require_and_forward_idempotency():
    async def run():
        f = await fixture()
        if f.store.tenant:
            from main_api.api import ops as public
        else:
            from main_api.api import aiops_app as public
        from fastapi import HTTPException
        from platform_clients.aiops import AIOpsManagementClient
        client = SimpleNamespace(extract_operations_diagnosis_case=AsyncMock(return_value={"job_id": "queued"}))
        request = SimpleNamespace(headers={}, state=SimpleNamespace(auth_context=human_context(f.store)),
            app=SimpleNamespace(state=SimpleNamespace(aiops_client=client)))
        guard = patch.object(public, "_require", new=AsyncMock()) if not f.store.tenant else patch.object(public, "_client", return_value=client)
        with guard:
            with pytest.raises(HTTPException) as caught:
                await public.extract_operations_diagnosis_case(f.report.report_id, request)
            assert caught.value.status_code == 428
            client.extract_operations_diagnosis_case.assert_not_awaited()
            request.headers["Idempotency-Key"] = "case-key"
            assert await public.extract_operations_diagnosis_case(f.report.report_id, request) == {"job_id": "queued"}
        assert client.extract_operations_diagnosis_case.await_args.kwargs["idempotency_key"] == "case-key"
        narrow = object.__new__(AIOpsManagementClient)
        narrow._json = AsyncMock(return_value={})
        await narrow.extract_operations_diagnosis_case(f.report.report_id,
            auth_context=request.state.auth_context, idempotency_key="case-key")
        assert narrow._json.await_args.kwargs["headers"] == {"Idempotency-Key": "case-key"}
        route = next(r for r in public.router.routes if r.path.endswith("/reports/{report_id}:extract-case"))
        assert route.status_code == 202 and route.response_model.__name__ == "OutputJobReceipt"
    asyncio.run(run())


def test_case_success_waits_for_exact_index_and_notifies_without_publication():
    async def run():
        f = await fixture()
        first = await advance(f)
        assert first.stage == "INDEXING" and first.finalization is None
        assert f.ref["intake"]["intake_id"] == str(f.report.report_id)
        f.phase = "READY"
        second = await advance(f)
        assert second.stage == "FINALIZING"
        job = job_for(f)
        result = await f.finalizer.succeeded(uow=f.uow, job=job, execution=f.execution,
            authority=f.authority, material=second.finalization, now=f.now)
        f.uow.platform_notifications.emit_output_job_event.assert_not_awaited()
        job.status = "SUCCEEDED"
        await f.finalizer.notify_succeeded(uow=f.uow, job=job, result=result, now=f.now)
        version = f.state.versions[result.asset_version_id]
        assert version.status == "DRAFT" and f.state.assets[version.asset_id].current_version_id is None
        assert version.source_hash == f.report.content_hash
        assert f.service._structured_model.generate_structured.await_count == 1
        assert f.documents.stage.await_count == 1
        assert f.uow.platform_notifications.emit_output_job_event.await_args.kwargs["event_type"] == "aiops.case.draft_ready"
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["report", "target", "agent", "binding", "artifact", "source_list", "plan", "owner", "security"])
def test_worker_and_retry_reject_real_source_change_without_new_model_or_upload(mutation):
    async def run():
        f = await fixture()
        if mutation == "report": f.report.report_version += 1
        if mutation == "target": f.uow.targets.get_scoped.return_value.row_version += 1
        if mutation == "agent": f.uow.agents.get.return_value.status = "DISABLED"
        if mutation == "binding": f.uow.agents.version_has_target.return_value = False
        if mutation == "artifact": f.artifact.content_hash = "f" * 64
        if mutation == "source_list": f.state.report_sources[f.report.report_id] = []
        if mutation == "plan": f.run.plan_snapshot_json["new"] = True
        if mutation == "security": f.artifact.security_level = 5
        job = job_for(f)
        if mutation == "owner": job.actor_id = str(uuid7()); f.scope = OutputJobScope(domain_id=f.domain,
            actor_id=job.actor_id, tenant_id=f.store.tenant, trace_id="changed")
        with pytest.raises(AIOpsApplicationError): await advance(f)
        with pytest.raises(AIOpsApplicationError): await OutputJobSourceValidator().retry(f.uow, job)
        f.service._structured_model.generate_structured.assert_not_awaited()
        f.documents.stage.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("retryable", [False, True])
def test_failure_restores_accepted_revision_before_revoked_source_cleanup(retryable):
    async def run():
        f = await fixture()
        saved = {}
        async def crash(recovery):
            saved.update(recovery.model_dump(mode="json"))
            raise RuntimeError("受理回执已发生，模拟登记前崩溃")
        with pytest.raises(RuntimeError):
            await execute_prepared_report_case(f.service, domain_id=f.domain, prepared=f.prepared,
                context=f.options.context, actor_id=f.actor, entity_scope=f.options.entity_scope,
                kc_scope=f.options.kc_scope, model_catalog=f.options.model_catalog,
                model_scope=f.options.model_scope, technical_name_field=f.options.technical_name_field,
                record_receipt=crash)
        assert len(f.state.versions) == 1
        f.ref = {"intake": saved}
        f.uow.agents.version_has_target.return_value = False
        await f.finalizer.failed(uow=f.uow, job=job_for(f, retryable=retryable), execution=f.execution, now=f.now)
        version = next(iter(f.state.versions.values()))
        index = f.state.relations[version.asset_version_id]["indexes"][0]
        assert str(index.document_version_id) == saved["document_version_id"]
        assert version.status == ("PROCESSING" if retryable else "REVIEW_REQUIRED")
        assert any(call.args[0].event_type == "OPS_CASE_WITHDRAWAL_REQUESTED"
            for call in f.uow.outbox.add.await_args_list) is not retryable
        f.documents.publish.assert_not_awaited()
    asyncio.run(run())


def test_presentation_revision_reuses_original_index_without_rewriting_old_source():
    async def run():
        f = await fixture()
        await advance(f)
        f.phase = "READY"
        first = await advance(f)
        version_id = UUID(first.finalization["case_result"]["asset_version_id"])
        original = deepcopy(f.state.versions[version_id].profile_json)
        await edit(f)
        f.report, f.ref = f.state.current, {}
        await freeze(f)
        assert f.prepared.request_hash != original["source"]["request_hash"]
        replay = await advance(f)
        assert replay.stage == "FINALIZING"
        result = await f.finalizer.succeeded(uow=f.uow, job=job_for(f), execution=f.execution,
            authority=f.authority, material=replay.finalization, now=f.now)
        assert result.asset_version_id == version_id
        assert f.state.versions[version_id].profile_json == original
        assert len(f.state.versions) == 1 and f.documents.stage.await_count == 1
    asyncio.run(run())


def test_frozen_envelope_rejects_mismatched_report_or_artifact_without_live_fallback():
    async def run():
        f = await fixture()
        assert report_case_from_frozen_input(f.frozen) == f.prepared
        changed = f.frozen.model_copy(update={"source_fingerprint": "f" * 64})
        with pytest.raises(AIOpsApplicationError): report_case_from_frozen_input(changed)
        f.report.report_version += 1
        with pytest.raises(AIOpsApplicationError):
            await validate_prepared_report_case(uow=f.uow, domain_id=f.domain, prepared=f.prepared)
    asyncio.run(run())


def test_corrupt_receipt_cannot_create_or_withdraw_an_unrelated_document():
    async def run():
        f = await fixture()
        await advance(f)
        f.phase = "READY"
        await advance(f)
        original = next(iter(f.state.versions.values()))
        receipt = deepcopy(f.ref["intake"])
        receipt["document_version_id"] = str(uuid7())
        f.ref = {"intake": receipt}
        with pytest.raises(AIOpsApplicationError):
            await f.finalizer.failed(uow=f.uow, job=job_for(f), execution=f.execution, now=f.now)
        assert len(f.state.versions) == 1 and original.status == "DRAFT"
        f.uow.outbox.add.assert_not_awaited()
        f.documents.stage.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["source", "result", "index", "asset", "version", "withdrawal", "security"])
def test_report_case_finalizer_rejects_changed_or_unrelated_result(mutation):
    async def run():
        f = await fixture()
        await advance(f)
        f.phase = "READY"
        material = (await advance(f)).finalization
        version = next(iter(f.state.versions.values()))
        asset = f.state.assets[version.asset_id]
        if mutation == "source": f.uow.targets.get_scoped.return_value.row_version += 1
        if mutation == "result": material["case_result"]["asset_version_id"] = str(uuid7())
        if mutation == "index": f.state.relations[version.asset_version_id]["indexes"][0].document_version_id = uuid7()
        if mutation == "asset": asset.row_version += 1
        if mutation == "version": version.row_version += 1
        if mutation == "withdrawal": f.uow.outbox.get_by_idempotency.return_value = SimpleNamespace(status="PENDING")
        if mutation == "security": asset.security_level = 0
        with pytest.raises(AIOpsApplicationError):
            await f.finalizer.succeeded(uow=f.uow, job=job_for(f), execution=f.execution,
                authority=f.authority, material=material, now=f.now)
        f.uow.platform_notifications.emit_output_job_event.assert_not_awaited()
    asyncio.run(run())


def test_actual_worker_waits_for_kc_and_atomically_finishes_with_notification():
    from aiops_agent.workers.output_jobs import OutputJobWorker
    from tests.unit.aiops_agent.test_aiops_output_job_worker import Store as WorkerStore, MemoryUow as WorkerUow
    async def run():
        f = await fixture()
        store = WorkerStore()
        store.job.job_id, store.job.job_type = f.execution.job_id, "CASE_EXTRACTION"
        store.job.domain_id, store.job.actor_id, store.job.tenant_id = f.domain, f.actor, f.scope.tenant_id
        snapshot = job_for(f)
        store.job.input_snapshot_json, store.job.input_hash = snapshot.input_snapshot_json, snapshot.input_hash
        store.job.source_json = snapshot.source_json
        store.job.processing_ref_json = {}
        store.authority.resolve.side_effect = lambda _: f.authority
        notification_broken = False
        class TransactionUow(WorkerUow):
            def __init__(self):
                super().__init__(store)
                for name in ("operations_knowledge", "inspections", "agents", "targets", "turns", "conversations", "outbox"):
                    setattr(self, name, getattr(f.uow, name))
                self.runs = SimpleNamespace(**vars(f.uow.runs))
                self.runs.database_now = AsyncMock(side_effect=lambda: store.now)
                self.platform_notifications = SimpleNamespace(emit_output_job_event=self.notify)
            async def notify(self, *, event_type, job, **_):
                if notification_broken:
                    raise RuntimeError("模拟通知事务暂不可写")
                self.notifications.append((event_type, job.job_id))
        worker = OutputJobWorker(uow_factory=TransactionUow, authority=store.authority,
            processor=f.processor, finalizer=f.finalizer, worker_id="report-case-worker")
        assert await worker.run_once()
        assert store.job.status == "RUNNING" and store.job.stage == "INDEXING"
        assert store.job.result_json is None and not store.notifications
        f.phase, notification_broken = "READY", True
        store.now = store.job.available_at
        assert await worker.run_once()
        assert store.job.status == "RUNNING" and store.job.result_json is None
        assert not store.notifications
        notification_broken = False
        store.now = store.job.available_at
        assert await worker.run_once()
        assert store.job.status == "SUCCEEDED" and store.job.result_json["asset_version_id"]
        assert store.notifications == [("aiops.case.draft_ready", store.job.job_id)]
        assert f.documents.stage.await_count == 1 and len(f.state.versions) == 1
    asyncio.run(run())


def test_report_reuses_actual_conversation_case_without_rewriting_its_request_identity():
    from tests.unit.aiops_agent.test_aiops_conversation_case_output import fixture as conversation_fixture, advance as conversation_advance
    from tests.support.aiops_report_output import render_report_material
    from aiops_agent.application.runtime.service import AIOpsRuntimeService
    from aiops_agent.application.reporting import resolve_system_template
    async def run():
        f = await conversation_fixture()
        await conversation_advance(f)
        f.phase = "READY"
        first = await conversation_advance(f)
        result_id = UUID(first.finalization["case_result"]["asset_version_id"])
        version = f.state.versions[result_id]
        original_profile = deepcopy(version.profile_json)
        f.state.closure = next(row for row in f.state.artifacts.values() if row.schema_version == "DIAGNOSIS_CLOSURE.v1")
        f.uow.operations_knowledge.find_case_by_artifact_id.side_effect = lambda **kw: (
            (f.state.assets[version.asset_id], version) if kw["artifact_id"] == f.state.closure.artifact_id else None)
        f.state.turns[:] = f.state.turns[:1]
        f.state.current = None
        runtime = AIOpsRuntimeService(uow_factory=lambda: f.manager,
            blueprint_registry=AsyncMock(), handler_registry=AsyncMock())
        await render_report_material(runtime, domain_id=f.domain, actor_id=f.actor,
            conversation_id=f.conversation_id, template=resolve_system_template("system:diagnosis.standard"), trace_id="same-closure")
        f.report = f.state.current
        assert f.state.artifacts[f.report.content_artifact_id].payload_json["scope"]["closure_artifact_id"] == str(f.state.closure.artifact_id)
        f.report.is_current = True
        assert await f.uow.inspections.get_report_scoped(report_id=f.report.report_id, domain_id=f.domain) is f.report
        assert f.report.status in {"READY", "PARTIAL"}
        prepared = await prepare_report_case(uow=f.uow, domain_id=f.domain, report_id=f.report.report_id, actor_id=f.actor)
        f.frozen = await freeze_report_case_input(prepared=prepared, uow=f.uow)
        f.ref = {}
        f.execution = OutputExecution(uuid7(), "CASE_EXTRACTION", f.scope, {}, uuid7(), "ACCEPTED", f.frozen)
        output = await advance(f)
        assert output.stage == "FINALIZING"
        f.uow.platform_notifications.emit_output_job_event = AsyncMock()
        result = await OutputJobFinalizer().succeeded(uow=f.uow, job=job_for(f), execution=f.execution,
            authority=f.authority, material=output.finalization, now=f.now)
        assert result.asset_version_id == result_id and version.profile_json == original_profile
        assert len(f.state.versions) == 1 and f.documents.stage.await_count == 1
    asyncio.run(run())


def test_registry_queries_scope_exact_intake_and_keep_retired_duplicates_out_of_canonical_lookup():
    from aiops_agent.repositories.knowledge import OperationsKnowledgeRepository
    async def run():
        session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(first=lambda: None)), commit=AsyncMock())
        repo, domain, revision, report_id = OperationsKnowledgeRepository(session), 1, uuid7(), uuid7()
        await repo.find_case_by_intake_revision(domain_id=domain, revision_id=revision)
        params = session.execute.await_args.args[0].compile().params
        assert domain in params.values() and revision in params.values() and "DIAGNOSIS_CASE" in params.values()
        await repo.find_by_source_hash(domain_id=domain, asset_kind="DIAGNOSIS_CASE", source_hash="a" * 64)
        params = session.execute.await_args.args[0].compile().params
        assert domain in params.values() and "RETIRED" in params.values()
        await repo.find_case_by_report_id(domain_id=domain, report_id=report_id)
        params = session.execute.await_args.args[0].compile().params
        assert domain in params.values() and report_id in params.values() and "REPORT" in params.values()
        session.commit.assert_not_awaited()
    asyncio.run(run())
