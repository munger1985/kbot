"""装配当前授权、冻结案例处理与事务终态，不创建另一套后台进程。"""

from aiops_agent.application.conversation_case_output import (
    CaseExecutionOptions, CaseOutputProcessor,
)
from aiops_agent.application.output_authorization import OutputJobAuthority
from aiops_agent.application.output_finalization import OutputJobFinalizer
from aiops_agent.application.output_processing import OutputJobProcessor
from aiops_agent.application.report_output import ReportOutputProcessor
from aiops_agent.workers.output_jobs import OutputJobWorker
from platform_clients.output_authority import OutputAuthorityClient
from platform_core.security import create_service_auth_context


def create_output_job_worker(settings, *, uow_factory, knowledge_service, runtime_service, session,
    plan_capture_reader=None):
    config = settings.worker
    dependency = settings.clients.main_api
    authority = OutputJobAuthority(client=OutputAuthorityClient(
        base_url=dependency.base_url, audience=dependency.audience,
        caller_service=config.service_name, session=session,
        timeout_seconds=dependency.timeout_seconds))

    def options_factory(execution, snapshot):
        # 当前授权已由 Worker 查验；下游只传播 SERVICE，不伪造发起人的登录身份。
        context = create_service_auth_context(
            caller_service=config.service_name, trace_id=execution.scope.trace_id)

        if execution.scope.tenant_id is not None:
            raise ValueError("KBot 后台输出不接受跨平台 Tenant")
        context = context.model_validate({**context.model_dump(),
            "domain_id": str(execution.scope.domain_id)})
        return CaseExecutionOptions(context=context, entity_scope={},
            kc_scope={"domain_id": execution.scope.domain_id},
            model_catalog=knowledge_service._llm_model_catalog,
            model_scope={}, technical_name_field="served_model_name")

    return OutputJobWorker(
        uow_factory=uow_factory, authority=authority,
        processor=OutputJobProcessor(
            case_processor=CaseOutputProcessor(
                knowledge_service=knowledge_service, options_factory=options_factory),
            report_processor=ReportOutputProcessor(uow_factory=uow_factory,
                plan_capture_reader=plan_capture_reader)),
        finalizer=OutputJobFinalizer(plan_capture_reader=plan_capture_reader,
            runtime_service=runtime_service),
        worker_id=f"{config.worker_id}-outputs", lease_seconds=config.lease_seconds,
        heartbeat_seconds=config.heartbeat_seconds, interval_seconds=config.claim_interval_seconds,
    )
