"""真实案例应用的异步处理回归；模拟本 App 文档状态，不冒充实库或模型验收。"""

import asyncio
from copy import deepcopy
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from aiops_agent.application.case_closure import execute_prepared_conversation_case
from aiops_agent.application.conversation_case_output import (
    CaseExecutionOptions, CaseOutputProcessor, submit_conversation_case,
)
from aiops_agent.application.output_jobs import OutputJobScope
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.workers.output_jobs import OutputExecution
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_case_frozen_input import prepare
from tests.unit.aiops_agent.test_aiops_conversation_case_closure import setup


async def fixture():
    f = await setup()
    f.uow.agents.get.return_value.status = "ACTIVE"
    f.prepared, f.frozen = await prepare(f, included=(f.turn_id,))
    tenant = uuid7() if "tenant_id" in inspect.signature(f.service.close_diagnosis).parameters else None
    f.scope = OutputJobScope(domain_id=f.domain, actor_id=f.actor, trace_id="trace", tenant_id=tenant)
    # 服务身份不携带登录用户；案例业务署名显式来自核验后的发起人。
    f.service_context = SimpleNamespace(principal_id=uuid7(), asserted_user_id=None,
        client_id="aiops-worker", principal_kind="SERVICE")
    f.options = CaseExecutionOptions(context=f.service_context,
        entity_scope={"tenant_id": tenant} if tenant else {},
        kc_scope={"tenant_id": tenant} if tenant else {"domain_id": f.domain},
        model_catalog=f.service._model_catalog if tenant else f.service._llm_model_catalog,
        model_scope={"tenant_id": str(tenant)} if tenant else {},
        technical_name_field="model_name" if tenant else "served_model_name")
    f.execution = OutputExecution(uuid7(), "CASE_EXTRACTION", f.scope, {}, uuid7(), "ACCEPTED", f.frozen)
    f.authority = SimpleNamespace(request=SimpleNamespace(source_agent_ids=(f.run.agent_id,)))
    f.ref = {}
    async def checkpoint(*, stage, processing_ref):
        f.ref = deepcopy(processing_ref)
    async def receipt(*, processing_ref):
        f.ref = deepcopy(processing_ref)
    f.checkpoint, f.receipt = AsyncMock(side_effect=checkpoint), AsyncMock(side_effect=receipt)
    f.processor = CaseOutputProcessor(knowledge_service=f.service,
        options_factory=lambda *_: f.options)
    original_store = f.uow.operations_knowledge.add_asset_version.side_effect
    async def store(**kw):
        await original_store(**kw)
        relations = f.state.relations[kw["version"].asset_version_id]
        relations.update(scopes=[], reviews=[])
    f.uow.operations_knowledge.add_asset_version.side_effect = store
    f.phase = "PROCESSING"
    return f


async def advance(f):
    execution = OutputExecution(f.execution.job_id, "CASE_EXTRACTION", f.scope,
        deepcopy(f.ref), uuid7(), "INDEXING" if f.ref else "ACCEPTED", f.frozen)
    return await f.processor.advance(execution, authority=f.authority,
        checkpoint=f.checkpoint, record_receipt=f.receipt)


async def execute(f, *, recovery=None, checkpoint=None, record_receipt=None):
    return await execute_prepared_conversation_case(f.service, domain_id=f.domain,
        prepared=f.prepared, context=f.service_context, actor_id=f.actor,
        entity_scope=f.options.entity_scope, kc_scope=f.options.kc_scope,
        model_catalog=f.options.model_catalog, model_scope=f.options.model_scope,
        technical_name_field=f.options.technical_name_field,
        recovery=recovery, checkpoint=checkpoint, record_receipt=record_receipt)


def test_kc_acceptance_waits_for_exact_indexed_draft_and_never_publishes():
    async def run():
        f = await fixture()
        first = await advance(f)
        assert first.stage == "INDEXING" and first.finalization is None
        assert f.ref["intake"]["document_version_id"] == str(next(iter(f.documents.records)))
        assert f.service._structured_model.generate_structured.await_count == 1
        f.phase = "READY"
        f.later.turn.status = "RUNNING"
        second = await advance(f)
        assert second.stage == "FINALIZING"
        version = next(iter(f.state.versions.values()))
        asset = f.state.assets[version.asset_id]
        assert version.status == "DRAFT" and asset.current_version_id is None
        assert str(version.created_by) == str(f.actor)
        assert f.documents.stage.await_count == 1
        assert f.service._structured_model.generate_structured.await_count == 1
        assert second.finalization["case_result"]["asset_version_id"] == str(version.asset_version_id)
    asyncio.run(run())


@pytest.mark.parametrize("stage", ["before_intake", "after_intake"])
def test_crash_reuses_frozen_redaction_and_accepted_revision(stage):
    async def run():
        f = await fixture()
        saved = {}
        async def freeze(recovery):
            saved.update(recovery.model_dump(mode="json"))
        async def crash(recovery):
            saved.update(recovery.model_dump(mode="json"))
            raise RuntimeError("模拟进程在检查点中断")
        if stage == "before_intake":
            with pytest.raises(RuntimeError):
                await execute(f, checkpoint=crash)
            f.documents.stage.assert_not_awaited()
        else:
            with pytest.raises(RuntimeError):
                await execute(f, checkpoint=freeze, record_receipt=crash)
            assert saved["document_version_id"] == str(next(iter(f.documents.records)))
        f.service._structured_model.generate_structured.side_effect = AssertionError("不允许重新生成脱敏正文")
        result = await execute(f, recovery=saved)
        assert result["version"]["status"] == "PROCESSING"
        assert f.documents.stage.await_count == 1
        f.documents.register.assert_awaited_once()
        assert len(f.state.versions) == 1
        index = next(iter(f.state.relations.values()))["indexes"][0]
        assert str(index.document_version_id) == str(next(iter(f.documents.records)))
    asyncio.run(run())


@pytest.mark.parametrize("field", ["title", "body_sha256", "request_hash", "source_fingerprint", "configuration_ref"])
def test_corrupt_recovery_never_uploads_changed_content(field):
    async def run():
        f = await fixture()
        saved = {}
        async def crash(recovery):
            saved.update(recovery.model_dump(mode="json"))
            raise RuntimeError()
        with pytest.raises(RuntimeError):
            await execute(f, checkpoint=crash)
        saved[field] = str(uuid7()) if field == "configuration_ref" else "f" * 64
        with pytest.raises(AIOpsApplicationError):
            await execute(f, recovery=saved)
        f.documents.stage.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["inactive", "changed"])
def test_source_agent_currentness_is_not_inferred_from_authority_echo(failure):
    async def run():
        f = await fixture()
        if failure == "inactive":
            f.uow.agents.get.return_value.status = "DISABLED"
        else:
            f.authority.request.source_agent_ids = (uuid7(),)
        with pytest.raises(AIOpsApplicationError):
            await advance(f)
        f.documents.stage.assert_not_awaited()
        f.service._structured_model.generate_structured.assert_not_awaited()
    asyncio.run(run())


def test_submission_callback_is_database_only_and_checks_owner():
    async def run():
        f = await fixture()
        jobs = SimpleNamespace(submit=AsyncMock())
        await submit_conversation_case(knowledge_service=f.service, jobs=jobs, scope=f.scope,
            conversation_id=f.conversation_id, command=SimpleNamespace(), idempotency_key="request")
        args = jobs.submit.await_args.kwargs
        assert args["job_type"] == "CASE_EXTRACTION" and args["idempotency_key"] == "request"
        with pytest.raises(AIOpsApplicationError):
            await args["freeze"](f.uow, SimpleNamespace(created_by="another-user"))
        f.service._structured_model.generate_structured.assert_not_awaited()
        f.documents.stage.assert_not_awaited()
    asyncio.run(run())

def test_lost_intake_response_replays_identical_body_and_idempotency_key():
    async def run():
        f = await fixture()
        saved = {}
        async def save(recovery):
            saved.update(recovery.model_dump(mode="json"))
        async def lost(_recovery):
            raise RuntimeError("外部受理后本地回执未保存")
        with pytest.raises(RuntimeError):
            await execute(f, checkpoint=save, record_receipt=lost)
        assert saved["ingestion_registered"] is False
        first = deepcopy(f.documents.stage.await_args.kwargs)
        await execute(f, recovery=saved)
        second = f.documents.stage.await_args.kwargs
        assert first["body"] == second["body"]
        assert first["version_id"] == second["version_id"]
        assert f.service._structured_model.generate_structured.await_count == 1
        assert f.documents.stage.await_count == 1
        assert len(f.state.versions) == 1
    asyncio.run(run())


def test_accepted_recovery_records_invalid_source_and_queues_exact_withdrawal():
    async def run():
        f = await fixture()
        saved = {}
        async def crash(recovery):
            saved.update(recovery.model_dump(mode="json"))
            raise RuntimeError()
        with pytest.raises(RuntimeError):
            await execute(f, record_receipt=crash)
        f.artifact.payload_json["changed"] = True
        f.documents.publish.reset_mock()
        with pytest.raises(AIOpsApplicationError):
            await execute(f, recovery=saved)
        assert len(f.state.versions) == 1
        version = next(iter(f.state.versions.values()))
        assert version.status == "REVIEW_REQUIRED"
        f.documents.stage.assert_awaited_once()
        f.documents.register.assert_awaited_once()
        f.documents.publish.assert_not_awaited()
        events = [call.args[0] for call in f.uow.outbox.add.await_args_list]
        assert any(row.event_type == "OPS_CASE_WITHDRAWAL_REQUESTED"
            and row.payload_json["document_version_id"] == saved["document_version_id"] for row in events)
    asyncio.run(run())


@pytest.mark.parametrize("retryable", [False, True])
def test_authority_loss_after_acceptance_never_approves_and_outage_does_not_permanently_withdraw(retryable):
    async def run():
        f = await fixture()
        accepted = False
        async def receipt(_recovery):
            nonlocal accepted
            accepted = True
        async def authorize():
            if accepted:
                raise AIOpsApplicationError(code="AUTHORITY_CHANGED", message="授权变化",
                    status_code=503 if retryable else 403, retryable=retryable)
        with pytest.raises(AIOpsApplicationError):
            await execute_prepared_conversation_case(f.service, domain_id=f.domain,
                prepared=f.prepared, context=f.service_context, actor_id=f.actor,
                entity_scope=f.options.entity_scope, kc_scope=f.options.kc_scope,
                model_catalog=f.options.model_catalog, model_scope=f.options.model_scope,
                technical_name_field=f.options.technical_name_field,
                record_receipt=receipt, source_preflight_hook=authorize)
        version = next(iter(f.state.versions.values()))
        assert version.status == ("PROCESSING" if retryable else "REVIEW_REQUIRED")
        f.documents.publish.assert_not_awaited()
        events = [call.args[0] for call in f.uow.outbox.add.await_args_list]
        assert any(row.event_type == "OPS_CASE_WITHDRAWAL_REQUESTED" for row in events) is not retryable
    asyncio.run(run())

def test_real_case_processor_runs_under_persistent_worker_and_resumes_after_index_wait():
    from aiops_agent.playbooks import canonical_hash
    from platform_core.contracts.aiops.output_jobs import OutputJobResult
    from tests.unit.aiops_agent.test_aiops_output_job_worker import Store

    async def run():
        f = await fixture()
        store = Store()
        job = store.job
        job.job_type, job.domain_id, job.actor_id, job.tenant_id = (
            "CASE_EXTRACTION", f.scope.domain_id, f.scope.actor_id, f.scope.tenant_id)
        job.input_snapshot_json = f.frozen.model_dump(mode="json")
        job.source_json = f.frozen.source.model_dump(mode="json")
        job.input_hash = canonical_hash({key: value for key, value in job.input_snapshot_json.items() if key != "frozen_at"})
        store.authority.resolve.side_effect = lambda _: f.authority
        store.processor = f.processor
        async def succeeded(*, uow, material, job, **_):
            result = OutputJobResult.model_validate(material["case_result"])
            return result
        async def notify_succeeded(*, uow, job, **_):
            assert job.status == "SUCCEEDED"
            uow.notifications.append(("case-draft-ready", job.job_id))
        store.finalizer.succeeded.side_effect = succeeded
        store.finalizer.notify_succeeded.side_effect = notify_succeeded
        worker = store.worker()
        await worker.run_once()
        assert store.job.status == "RUNNING" and store.job.stage == "INDEXING"
        assert not store.notifications
        assert store.job.processing_ref_json["intake"]["document_version_id"] == str(next(iter(f.documents.records)))
        store.now = store.job.available_at
        f.phase = "READY"
        await worker.run_once()
        assert store.job.status == "SUCCEEDED"
        assert store.job.result_json["asset_version_id"] in {
            str(value) for value in f.state.versions}
        assert store.notifications == [("case-draft-ready", job.job_id)]
        f.documents.stage.assert_awaited_once()
        assert not await worker.run_once()
        assert len(f.state.versions) == 1
    asyncio.run(run())

def test_recovery_finds_its_exact_artifact_even_after_a_later_case_version_exists():
    async def run():
        f = await fixture()
        first = await execute(f)
        version = f.state.versions[UUID(first["version"]["asset_version_id"])]
        asset = f.state.assets[version.asset_id]
        f.uow.operations_knowledge.find_case_by_artifact_id.return_value = (asset, version)
        f.uow.operations_knowledge.find_case_by_external_id.side_effect = None
        f.uow.operations_knowledge.find_case_by_external_id.return_value = (
            asset, SimpleNamespace(profile_json={"source": {"request_hash": "f" * 64}}))
        again = await execute(f)
        assert again["version"]["asset_version_id"] == first["version"]["asset_version_id"]
        assert again["replayed"] is True
        assert len(f.state.versions) == 1
        f.documents.stage.assert_awaited_once()
        f.service._structured_model.generate_structured.assert_awaited_once()
    asyncio.run(run())
