"""输出任务的发起人作用域查询与冻结来源重试接口。"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request

from aiops_agent.api.dependencies import get_aiops_auth_context, require_service_scope
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.contracts.aiops.output_jobs import (
    OutputJobList, OutputJobReceipt, OutputJobRetry, OutputJobStatus, OutputJobType,
)
from aiops_agent.application.output_jobs import OutputJobScope
from fastapi import HTTPException
from platform_core.contracts.aiops.output_projection import OutputProjectionRequest, OutputProjectionSnapshot


router = APIRouter(prefix="/internal/v1/aiops/output-jobs", tags=["AIOps Output Jobs"])


@router.post(":projection", response_model=OutputProjectionSnapshot)
async def get_output_projection(body: OutputProjectionRequest, request: Request,
    context: AuthContext = Depends(get_aiops_auth_context)):
    require_service_scope(request, "aiops.output.reconcile")
    if context.principal_kind != PrincipalKind.SERVICE:
        raise HTTPException(403, {"code": "AIOPS_OUTPUT_PROJECTION_SERVICE_REQUIRED"})
    return await request.app.state.output_projection_service.read(body)


def output_job_scope(request, context):
    require_service_scope(request, "aiops.run")
    if (context.principal_kind != PrincipalKind.PORTAL or not context.asserted_user_id
        or context.domain_id is None):
        raise HTTPException(403, {"code": "AIOPS_OUTPUT_HUMAN_CONTEXT_REQUIRED"})
    try:
        domain_id = int(context.domain_id)
        if domain_id < 1:
            raise ValueError()
    except ValueError as exc:
        raise HTTPException(403, {"code": "AIOPS_DOMAIN_CONTEXT_REQUIRED"}) from exc
    return OutputJobScope(domain_id=domain_id, actor_id=context.asserted_user_id,
        trace_id=context.trace_id)


@router.get("", response_model=OutputJobList)
async def list_output_jobs(request: Request,
    job_type: OutputJobType | None = None, status: OutputJobStatus | None = None,
    conversation_id: UUID | None = None, before_id: UUID | None = None,
    limit: int = Query(50, ge=1, le=100),
    context: AuthContext = Depends(get_aiops_auth_context)):
    return await request.app.state.output_job_service.list(
        scope=output_job_scope(request, context),
        authorize=request.app.state.output_job_source_validator.read,
        job_type=job_type, status=status, conversation_id=conversation_id,
        before_id=before_id, limit=limit)


@router.get("/{job_id}", response_model=OutputJobReceipt)
async def get_output_job(job_id: UUID, request: Request,
    context: AuthContext = Depends(get_aiops_auth_context)):
    return await request.app.state.output_job_service.get(
        scope=output_job_scope(request, context), job_id=job_id,
        authorize=request.app.state.output_job_source_validator.read)


@router.post("/{job_id}:retry", status_code=202, response_model=OutputJobReceipt)
async def retry_output_job(job_id: UUID, body: OutputJobRetry, request: Request,
    context: AuthContext = Depends(get_aiops_auth_context)):
    async def authorize(uow, job):
        if job.job_type == "CASE_EXTRACTION" and job.source_kind == "REPORT":
            require_service_scope(request, "aiops.manage")
        await request.app.state.output_job_source_validator.retry(uow, job)
    return await request.app.state.output_job_service.retry(
        scope=output_job_scope(request, context), job_id=job_id,
        expected_row_version=body.expected_row_version, authorize=authorize)
