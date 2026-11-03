"""真实报告受理、冻结处理和事务终态回归；不冒充实库或浏览器验收。"""

import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from aiops_agent.api.runtime import routes
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.output_finalization import OutputJobFinalizer
from aiops_agent.application.output_jobs import OutputJobScope, OutputJobService
from aiops_agent.application.report_output import (
    ReportOutputProcessor, report_from_frozen_input, submit_report_output, validate_report_input,
)
from aiops_agent.application.reporting import resolve_system_template
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.playbooks import canonical_hash
from aiops_agent.workers.output_jobs import OutputExecution, OutputJobWorker
from platform_core.contracts.aiops.output_jobs import FrozenOutputInput
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_conversation_case_closure import setup, refute_first_claim
from tests.unit.aiops_agent.test_aiops_output_job_api import api
from tests.unit.aiops_agent.test_aiops_output_jobs import Store, MemoryUow


async def fixture(*, second=False):
    f = await setup(second=second)
    f.knowledge = f.service
    f.store = Store()
    if f.store.tenant is None:
        f.domain, f.actor = f.store.domain, str(f.actor)
    f.store.domain, f.store.actor = f.domain, f.actor
    f.scope = OutputJobScope(domain_id=f.domain, actor_id=f.actor,
        tenant_id=f.store.tenant, trace_id="report-job")
    f.store.scope = f.scope
    f.store.source = f.store.source.model_copy(update={"source_id": f.conversation_id})
    f.store.root.created_by = f.actor
    f.conversation.created_by, f.conversation.agent_id = f.actor, f.run.agent_id
    for run in f.state.runs.values():
        run.domain_id, run.actor_id = f.domain, f.actor
        if f.store.tenant:
            run.tenant_id = f.store.tenant
    f.state.current = None
    f.uow.inspections.publish_report.reset_mock()
    f.uow.platform_notifications.emit_report_ready.reset_mock()
    f.uow.inspections.get_report_scoped = AsyncMock(side_effect=lambda **_: f.state.current)
    f.uow.runs.get_run = AsyncMock(side_effect=lambda **kw: f.state.runs.get(kw["ops_run_id"]))
    f.uow.targets.get_scoped.return_value.status = "ENABLED"
    f.uow.targets.get_scoped.return_value.row_version = 1
    f.template = resolve_system_template("system:diagnosis.standard")
    def unit():
        value = MemoryUow(f.store)
        for name in ("runs", "turns", "targets", "agents", "conversations", "inspections"):
            setattr(value, name, getattr(f.uow, name))
        return value
    f.factory = unit
    f.jobs = f.store.service = OutputJobService(uow_factory=unit)
    f.runtime = AIOpsRuntimeService(uow_factory=lambda: f.manager,
        blueprint_registry=AsyncMock(), handler_registry=AsyncMock())
    f.uow.output_jobs = SimpleNamespace(lock_source=AsyncMock(return_value=f.store.root))
    f.processor = ReportOutputProcessor(uow_factory=lambda: f.manager)
    f.finalizer = OutputJobFinalizer(runtime_service=f.runtime)
    return f


async def submit(f, *, conversation=True, key="report-click"):
    return await submit_report_output(runtime_service=f.runtime, jobs=f.jobs, scope=f.scope,
        conversation_id=f.conversation_id if conversation else None,
        ops_run_id=None if conversation else f.run_id, template=f.template,
        template_version_id=None, period_kind="AD_HOC", idempotency_key=key)


async def freeze(f):
    receipt = await submit(f)
    f.job = f.store.jobs[receipt.job_id]
    f.job.status = "RUNNING"
    f.frozen = FrozenOutputInput.model_validate(f.job.input_snapshot_json)
    f.execution = OutputExecution(f.job.job_id, "REPORT_GENERATION", f.scope,
        {}, uuid7(), "ASSEMBLING", f.frozen)
    return f.frozen


async def finish(f):
    advance = await f.processor.advance(f.execution, authority=object(),
        checkpoint=AsyncMock(), record_receipt=AsyncMock())
    result = await f.finalizer.succeeded(uow=f.uow, job=f.job, execution=f.execution,
        authority=object(), material=advance.finalization, now=f.now)
    f.job.status = "SUCCEEDED"
    await f.finalizer.notify_succeeded(uow=f.uow, job=f.job, result=result, now=f.now)
    return result


@pytest.mark.parametrize("conversation", [False, True])
def test_internal_report_submission_is_database_only_persistent_202_and_idempotent(conversation):
    async def run():
        f = await fixture()
        if not conversation:
            f.store.source = f.store.source.model_copy(update={"kind": "RUN", "source_id": f.run_id})
            f.runtime.get_run = AsyncMock(return_value=f.run)
        app = api(f.store)
        app.include_router(routes.router)
        app.state.aiops_runtime_service = f.runtime
        app.state.session_report_template_service = SimpleNamespace(resolve=AsyncMock(return_value=f.template))
        payload = {"conversation_id": str(f.conversation_id)} if conversation else {"ops_run_id": str(f.run_id)}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://aiops.test") as client:
            path = "/internal/v1/aiops/reports:generate"
            assert (await client.post(path, json=payload)).status_code == 428
            assert not f.store.jobs
            first = await client.post(path, json=payload, headers={"Idempotency-Key": "report-click"})
            assert first.status_code == 202, first.text
            assert first.json()["status"] == "QUEUED" and first.json()["result"] is None
            repeated = await client.post(path, json=payload, headers={"Idempotency-Key": "report-click"})
            assert repeated.status_code == 202 and repeated.json()["job_id"] == first.json()["job_id"]
            assert len(f.store.jobs) == len(f.store.events) == len(f.store.notifications) == 1
            f.artifact.content_hash = "f" * 64
            assert (await client.post(path, json=payload, headers={"Idempotency-Key": "report-click"})).status_code == 409
        f.uow.inspections.publish_report.assert_not_awaited()
        f.knowledge._structured_model.generate_structured.assert_not_awaited()
        f.knowledge._kc.ingest_user_file.assert_not_awaited()
    asyncio.run(run())


def test_public_and_narrow_client_forward_queued_contract_and_require_idempotency():
    async def run():
        from main_api.api import ops as public
        from platform_clients.aiops import AIOpsManagementClient
        f = await fixture()
        client = SimpleNamespace(generate_user_report=AsyncMock(return_value={"job_id": "queued"}))
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(aiops_client=client)),
            state=SimpleNamespace(auth_context=object()))
        command = public.ReportGenerationPayload(conversation_id=f.conversation_id)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as caught:
            await public.generate_report(command, request)
        assert caught.value.status_code == 428
        client.generate_user_report.assert_not_awaited()
        assert await public.generate_report(command, request, "report-key") == {"job_id": "queued"}
        assert client.generate_user_report.await_args.kwargs["idempotency_key"] == "report-key"
        route = next(route for route in public.router.routes if route.path.endswith("/reports:generate"))
        assert route.status_code == 202 and route.response_model.__name__ == "OutputJobReceipt"
        narrow = object.__new__(AIOpsManagementClient)
        narrow._json = AsyncMock(return_value={})
        await narrow.generate_user_report(conversation_id=f.conversation_id, ops_run_id=None,
            template_ref=f.template.template_ref, template_version_id=None, period_kind="AD_HOC",
            idempotency_key="report-key", auth_context=object())
        assert narrow._json.await_args.kwargs["idempotency_key"] == "report-key"
    asyncio.run(run())


@pytest.mark.parametrize("refuted", [False, True])
def test_later_turn_does_not_expand_input_but_real_counterevidence_blocks(refuted):
    async def run():
        f = await fixture(second=True)
        if refuted:
            refute_first_claim(f)
        turns = list(f.state.turns)
        f.state.turns[:] = turns[:1]
        await freeze(f)
        selected = f.frozen.parameters["report"]["source"]["conversation"]["included_turn_ids"]
        assert selected == [str(f.turn_id)]
        f.state.turns[:] = turns
        if refuted:
            with pytest.raises(AIOpsApplicationError):
                await finish(f)
        else:
            f.later.turn.status = "RUNNING"
            result = await finish(f)
            assert result.report_id and f.state.current.status == "READY"
            content = f.state.artifacts[f.state.current.content_artifact_id].payload_json
            assert content["scope"]["conversation"]["included_turn_ids"] == selected
        f.knowledge._structured_model.generate_structured.assert_not_awaited()
        f.knowledge._kc.ingest_user_file.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["artifact", "target", "agent", "binding", "turn", "anchor", "owner"])
def test_frozen_report_rechecks_source_authorization_and_membership(mutation):
    async def run():
        f = await fixture()
        await freeze(f)
        if mutation == "artifact": f.artifact.payload_json["changed"] = True
        if mutation == "target": f.uow.targets.get_scoped.return_value.row_version += 1
        if mutation == "agent": f.uow.agents.get.return_value.status = "DISABLED"
        if mutation == "binding": f.uow.agents.version_has_target.return_value = False
        if mutation == "turn": f.turn.conversation_id = uuid7()
        if mutation == "anchor": f.run.status = "FAILED"
        if mutation == "owner": f.conversation.created_by = "another"
        with pytest.raises(AIOpsApplicationError):
            await finish(f)
        f.uow.inspections.publish_report.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["retired", "definition"])
def test_exact_template_version_survives_current_pointer_change_but_not_invalid_source(mutation):
    async def run():
        from aiops_agent.application.report_templates import SessionReportTemplateService
        f = await fixture()
        template_id, version_id = uuid7(), uuid7()
        definition, _ = SessionReportTemplateService._normalize_definition(
            display_name="业务空间诊断报告",
            definition={"sections": [{"kind": "FINDINGS"}]},
        )
        root = SimpleNamespace(template_id=template_id, current_version_id=version_id, status="ACTIVE")
        root.display_name = "业务空间诊断报告"
        version = SimpleNamespace(template_id=template_id, template_version_id=version_id,
            domain_id=f.domain, version_no=1, content_hash=canonical_hash(definition), definition_json=definition)
        f.uow.inspections.get_session_report_template = AsyncMock(return_value=root)
        f.uow.inspections.get_session_report_template_version = AsyncMock(return_value=version)
        templates = SessionReportTemplateService(uow_factory=lambda: f.manager)
        f.template = await templates.resolve(domain_id=f.domain, template_ref="domain:" + str(template_id),
            template_version_id=version_id)
        assert f.template.content_hash == version.content_hash
        await freeze(f)
        root.current_version_id = uuid7()
        result = await finish(f)
        assert result.report_id and f.state.current.template_id == f.template.template_ref
        assert f.state.current.template_version == "1"
        assert f.uow.inspections.get_session_report_template_version.await_args.kwargs["template_version_id"] == version_id
        if mutation == "retired":
            root.status = "INACTIVE"
        else:
            version.definition_json["sections"][0]["title"] = "已被改写"
        with pytest.raises(AIOpsApplicationError):
            await validate_report_input(uow=f.uow, scope=f.scope, frozen=f.frozen)
    asyncio.run(run())


@pytest.mark.parametrize("changed", [False, True])
def test_report_retry_revalidates_originals_without_new_model_or_evidence(changed):
    async def run():
        from aiops_agent.application.output_job_sources import OutputJobSourceValidator
        f = await fixture()
        await freeze(f)
        if changed:
            f.artifact.content_hash = "f" * 64
            with pytest.raises(AIOpsApplicationError):
                await OutputJobSourceValidator().retry(f.uow, f.job)
        else:
            await OutputJobSourceValidator().retry(f.uow, f.job)
        f.knowledge._structured_model.generate_structured.assert_not_awaited()
        f.knowledge._kc.ingest_user_file.assert_not_awaited()
        f.uow.inspections.publish_report.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("invalidated", [False, True])
def test_monthly_comparison_reference_is_frozen_and_rechecked(invalidated):
    async def run():
        from aiops_agent.application.report_output import freeze_report_input
        from aiops_agent.application.report_preparation import prepare_conversation_report
        f = await fixture()
        await freeze(f)
        await finish(f)
        dependency = f.state.current
        dependency.is_current = True
        f.uow.inspections.get_report_by_content_scoped = AsyncMock(return_value=dependency)
        f.uow.inspections.get_report_scoped = AsyncMock(return_value=dependency)
        f.uow.inspections.list_report_successors_scoped = AsyncMock(return_value=[])
        prepared = await prepare_conversation_report(f.runtime, uow=f.uow, domain_id=f.domain,
            actor_id=f.actor, conversation_id=f.conversation_id, template=f.template, trace_id="monthly-reference")
        prepared = replace(prepared, period_kind="MONTHLY",
            extra_artifacts=((dependency.ops_run_id, f.state.artifacts[dependency.content_artifact_id]),))
        frozen = await freeze_report_input(prepared=prepared, uow=f.uow, scope=f.scope,
            source=f.frozen.source)
        material, _ = report_from_frozen_input(frozen)
        assert material.source_reports[str(dependency.content_artifact_id)] == dependency.report_id
        if invalidated:
            dependency.is_current = False
            with pytest.raises(AIOpsApplicationError):
                await validate_report_input(uow=f.uow, scope=f.scope, frozen=frozen)
        else:
            await validate_report_input(uow=f.uow, scope=f.scope, frozen=frozen)
    asyncio.run(run())


def test_malformed_source_reference_is_controlled_and_never_enqueues():
    async def run():
        f = await fixture()
        original = f.runtime._aggregate_conversation_sources
        def malformed(**kwargs):
            source = original(**kwargs)
            source["evidence_refs"] = ({"artifact_id": "not-a-uuid"},)
            return source
        f.runtime._aggregate_conversation_sources = malformed
        with pytest.raises(AIOpsApplicationError) as caught:
            await submit(f)
        assert caught.value.code == "AIOPS_REPORT_OUTPUT_SOURCE_INVALID"
        assert not f.store.jobs and not f.store.events
    asyncio.run(run())


@pytest.mark.parametrize("complete", [False, True])
def test_report_content_partial_and_security_are_independent_of_job_success_and_reuse(complete):
    async def run():
        f = await fixture()
        f.artifact.security_level = 4
        if not complete:
            f.artifact.payload_json["status"] = "PARTIAL"
            f.artifact.payload_json["investigation_record"]["goal_assessments"][0]["status"] = "PARTIAL"
            f.artifact.payload_json["investigation_record"]["stop_reason"] = "PARTIAL_ANSWER"
            f.artifact.content_hash = canonical_hash(f.artifact.payload_json)
        await freeze(f)
        first = await finish(f)
        repeated = await finish(f)
        assert first == repeated and first.report_version == "1"
        assert f.state.current.security_level == 4
        assert f.state.artifacts[f.state.current.content_artifact_id].security_level == 4
        assert f.state.current.status == ("READY" if complete else "PARTIAL")
        assert f.uow.inspections.publish_report.await_count == 1
        assert f.uow.platform_notifications.emit_report_ready.await_count == 2
        assert f.uow.platform_notifications.emit_report_ready.await_args.kwargs["output_job"] is f.job
    asyncio.run(run())


@pytest.mark.parametrize("lowered", ["report", "content"])
def test_report_reuse_cannot_downgrade_frozen_source_security(lowered):
    async def run():
        f = await fixture()
        f.artifact.security_level = 4
        await freeze(f)
        await finish(f)
        existing = f.state.current
        target = existing if lowered == "report" else f.state.artifacts[existing.content_artifact_id]
        target.security_level = 1
        await finish(f)
        assert f.uow.inspections.publish_report.await_count == 2
        assert f.state.current.security_level >= 4
        assert f.state.artifacts[f.state.current.content_artifact_id].security_level >= 4
    asyncio.run(run())


@pytest.mark.parametrize("field", ["source_fingerprint", "template", "period", "reference"])
def test_damaged_frozen_report_is_controlled_error(field):
    async def run():
        f = await fixture()
        await freeze(f)
        data = f.frozen.model_dump(mode="json")
        if field == "source_fingerprint": data[field] = "f" * 64
        if field == "template": data[field]["content_hash"] = "f" * 64
        if field == "period": data["parameters"]["report"]["period_kind"] = "INVALID"
        if field == "reference": data["artifacts"].append(deepcopy(data["artifacts"][0]))
        with pytest.raises(AIOpsApplicationError) as caught:
            report_from_frozen_input(FrozenOutputInput.model_validate(data))
        assert caught.value.code == "AIOPS_OUTPUT_JOB_INPUT_INVALID"
    asyncio.run(run())

@pytest.mark.parametrize("fail_notification", [False, True])
def test_actual_report_worker_commits_report_job_and_notice_atomically_and_recovers(fail_notification):
    from tests.unit.aiops_agent.test_aiops_output_job_worker import Store as WorkerStore, MemoryUow as WorkerUow

    async def run():
        f = await fixture()
        receipt = await submit(f)
        store = WorkerStore()
        submitted = f.store.jobs[receipt.job_id]
        store.job = SimpleNamespace(**{
            key: getattr(submitted, key, None) for key in (
                "job_id", "job_type", "domain_id", "actor_id", "tenant_id", "source_kind", "source_id",
                "source_json", "input_snapshot_json", "input_hash", "trace_id", "status", "stage",
                "available_at", "submitted_at", "attempt_count", "started_at", "completed_at",
                "updated_at", "heartbeat_at", "lease_token", "lease_until", "lease_owner",
                "error_code", "error_message", "retryable", "failure_stage", "row_version",
            )})
        store.job.processing_ref_json = None
        store.job.result_json = None
        store.now = submitted.submitted_at
        f.fail_notice = fail_notification

        class ReportUow(WorkerUow):
            async def __aenter__(self):
                await super().__aenter__()
                self.report_state = deepcopy(f.state)
                for name in ("targets", "agents", "turns", "conversations"):
                    setattr(self, name, getattr(f.uow, name))
                self.runs = SimpleNamespace(**vars(f.uow.runs))
                self.runs.database_now = AsyncMock(side_effect=lambda: store.now)
                self.runs.get_artifact = AsyncMock(side_effect=lambda *, artifact_id:
                    self.report_state.artifacts.get(artifact_id))
                async def add_artifact(entity):
                    entity.artifact_id = entity.artifact_id or uuid7()
                    self.report_state.artifacts[entity.artifact_id] = entity
                    return entity
                self.runs.add_artifact = AsyncMock(side_effect=add_artifact)
                self.inspections = SimpleNamespace(**vars(f.uow.inspections))
                async def publish(entity):
                    self.report_state.current = entity
                    self.outputs.append(str(entity.report_id))
                    return entity
                self.inspections.publish_report = AsyncMock(side_effect=publish)
                self.inspections.get_report_scoped = AsyncMock(side_effect=lambda **_: self.report_state.current)
                for method in ("get_current_report_for_run_template", "get_current_report_for_conversation_template"):
                    setattr(self.inspections, method, AsyncMock(side_effect=lambda **_: self.report_state.current))
                async def notify(**kw):
                    if f.fail_notice:
                        raise RuntimeError("模拟报告通知入队失败")
                    self.notifications.append(("report-ready", str(kw["report"].report_id), str(kw["output_job"].job_id)))
                self.platform_notifications = SimpleNamespace(emit_report_ready=AsyncMock(side_effect=notify))
                self.outbox = SimpleNamespace(add=AsyncMock())
                return self

            async def lock_source(self, **_):
                return f.store.root

            async def commit(self):
                await super().commit()
                f.state.current, f.state.artifacts = self.report_state.current, self.report_state.artifacts

        store.authority.resolve.side_effect = lambda _: object()
        worker = OutputJobWorker(uow_factory=lambda: ReportUow(store), authority=store.authority,
            processor=ReportOutputProcessor(uow_factory=lambda: ReportUow(store)),
            finalizer=f.finalizer, worker_id="actual-report-worker")
        await worker.run_once()
        if fail_notification:
            assert store.job.status == "RUNNING"
            assert not store.outputs and not store.notifications and f.state.current is None
            f.fail_notice = False
            store.now = store.job.available_at
            await worker.run_once()
        assert store.job.status == "SUCCEEDED"
        assert store.job.result_json["report_id"] == str(f.state.current.report_id)
        assert len(store.outputs) == len(store.notifications) == 1
        assert store.notifications[0][2] == str(store.job.job_id)
        assert not await worker.run_once()
        assert len(store.outputs) == len(store.notifications) == 1
    asyncio.run(run())
