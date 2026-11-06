"""真实输出 Worker 装配与平台 SERVICE 上下文回归；不连接外部依赖。"""

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.bootstrap.output_jobs import create_output_job_worker
from aiops_agent.application.conversation_case_output import CaseExecutionOptions
from aiops_agent.application.output_authorization import OutputJobAuthority
from aiops_agent.application.output_finalization import OutputJobFinalizer
from aiops_agent.application.output_processing import OutputJobProcessor
from aiops_agent.application.report_output import ReportOutputProcessor
from aiops_agent.workers.output_jobs import OutputJobWorker
from platform_core.contracts import AuthContext
from tests.unit.aiops_agent.test_aiops_conversation_case_output import fixture


def settings():
    return SimpleNamespace(worker=SimpleNamespace(
        service_name="actual-aiops-worker", worker_id="worker-instance",
        lease_seconds=120, heartbeat_seconds=30, claim_interval_seconds=2),
        clients=SimpleNamespace(main_api=SimpleNamespace(
            base_url="http://main-api.test", audience="main-api",
            timeout_seconds=7)))


def test_factory_uses_actual_components_endpoint_and_configured_leases():
    runtime = object()
    worker = create_output_job_worker(settings(),
        uow_factory=object(), knowledge_service=object(), runtime_service=runtime, session=object())
    assert isinstance(worker, OutputJobWorker)
    assert isinstance(worker._authority, OutputJobAuthority)
    assert isinstance(worker._finalizer, OutputJobFinalizer)
    assert isinstance(worker._processor, OutputJobProcessor)
    assert isinstance(worker._processor.report_processor, ReportOutputProcessor)
    assert worker._finalizer._runtime_service is runtime
    assert worker._authority._client._base_url == "http://main-api.test"
    assert worker._authority._client._caller == "actual-aiops-worker"
    assert worker._owner == "worker-instance-outputs"
    assert worker._heartbeat_seconds == 30 and worker._lease.total_seconds() == 120


def test_real_platform_service_context_is_scoped_but_never_impersonates_actor():
    async def run():
        f = await fixture()
        worker = create_output_job_worker(settings(), uow_factory=f.service._uow_factory,
            knowledge_service=f.service, runtime_service=object(), session=object())
        options = worker._processor.case_processor._options_factory(f.execution, f.authority)
        assert isinstance(options, CaseExecutionOptions)
        assert isinstance(options.context, AuthContext)
        assert str(options.context.principal_kind).upper() == "SERVICE"
        assert options.context.asserted_user_id is None
        assert str(options.context.domain_id) == str(f.domain)
        assert options.context.calling_service == "actual-aiops-worker"
        assert not options.context.api_key_id
        if f.scope.tenant_id is not None:
            assert options.context.principal_id != f.scope.actor_id
            assert options.context.tenant_ids == (f.scope.tenant_id,)
            assert options.context.app_id == "aiops"
            assert options.technical_name_field == "model_name"
        else:
            assert options.context.tenant_id is None
            assert options.technical_name_field == "served_model_name"
        output = await worker._processor.advance(f.execution,
            authority=f.authority, checkpoint=f.checkpoint, record_receipt=f.receipt)
        assert output.stage == "INDEXING" and output.finalization is None
        f.service._documents.stage.assert_awaited_once()
        access = f.service._documents.register.await_args.args[1]
        assert str(access.subject_ref) == str(f.scope.actor_id)
        assert str(access.domain_id) == str(f.domain)
    asyncio.run(run())


def test_main_worker_lifespan_installs_output_consumer_in_existing_component_lifecycle():
    from aiops_agent.bootstrap import worker
    source = inspect.getsource(worker.create_aiops_worker_probe)
    assert "output_worker = create_output_job_worker(" in source
    assert "output_job_worker=output_worker" in source
    assert '"output-jobs": output_worker' in source
    assert "named_components.items()" in source
    assert "named_components.values()" in source
