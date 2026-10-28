"""DB Executor 反向 Claim 内部接口。"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from aiops_agent.api.changes.routes import get_service
from aiops_agent.api.dependencies import require_service_scope
from aiops_agent.application.changes import AIOpsChangeService
from platform_core.contracts.aiops.executor import (
    CredentialIssueRequest,
    CredentialIssueResponse,
    ExecutionStatusEvent,
    MutationClaimReceipt,
    MutationClaimRequest,
)
from aiops_agent.application.credential_issuance import CredentialIssuanceService, CredentialIssueError
from aiops_agent.application.execution_fence import ReadonlyExecutionFenceService
from aiops_agent.application.execution_fence import ExecutionFenceError
from aiops_agent.diagnostics.grants import DiagnosticGrantError
from platform_core.contracts.aiops.sql_plan_execution import (
    SqlPlanCaptureExecute, SqlPlanCaptureExchange, SqlPlanCaptureReport, SqlPlanCaptureResult,
)
from aiops_agent.application.readonly_lifecycle import ReadonlyLifecycleService, ReadonlyLifecycleError
from platform_core.contracts.aiops.executor import ReadonlyTerminationRequest
from platform_core.contracts.aiops.readonly_lifecycle import ReadonlyTerminationReceipt
from platform_core.contracts.aiops.executor import ReadonlyExecutionFenceRequest, ReadonlyExecutionFenceReceipt
from aiops_agent.application.resource_governance import ResourceGovernanceService, ResourceAdmissionError
from platform_core.contracts.aiops.executor import ResourceLeaseReceipt, ResourceReleaseReceipt
from platform_core.contracts.aiops.internal import EventReceipt


router = APIRouter(
    prefix="/internal/v1/aiops/executions",
    tags=["AIOps Executions"],
)
event_router = APIRouter(
    prefix="/internal/v1/aiops",
    tags=["AIOps Executions"],
)
Service = Annotated[AIOpsChangeService, Depends(get_service)]


@router.post(
    "/{execution_id}/claim",
    response_model=MutationClaimReceipt,
)
async def claim_execution(
    execution_id: UUID,
    body: MutationClaimRequest,
    request: Request,
    service: Service,
) -> MutationClaimReceipt:
    require_service_scope(request, "aiops.execution.claim")
    context = request.state.auth_context
    return await service.claim_execution(
        execution_id=execution_id,
        command=body,
        trace_id=context.trace_id,
    )


@event_router.post(
    "/executor-events",
    response_model=EventReceipt,
)
async def apply_execution_event(
    body: ExecutionStatusEvent,
    request: Request,
    service: Service,
) -> EventReceipt:
    require_service_scope(request, "aiops.execution.callback")
    context = request.state.auth_context
    return await service.apply_execution_event(
        event=body,
        trace_id=context.trace_id,
    )


@event_router.post("/credentials:issue", response_model=CredentialIssueResponse)
async def issue_credential(body: CredentialIssueRequest, request: Request) -> CredentialIssueResponse:
    """仅向已验证的 Executor 发放且消费一次连接材料。"""
    require_service_scope(request, "aiops.credentials.issue")
    service = CredentialIssuanceService(
        uow_factory=request.app.state.configuration_service._uow_factory,
        diagnostic_codec=request.app.state.diagnostic_grant_codec,
        mutation_codec=request.app.state.mutation_grant_codec,
        credential_service=request.app.state.managed_credential_service,
        audience="kbot-aiops-db-executor",
    )
    try:
        return await service.issue(body.grant)
    except CredentialIssueError as exc:
        raise HTTPException(status_code=403, detail={"code": exc.code}) from exc
    except ResourceAdmissionError as exc:
        raise HTTPException(status_code=429, detail={"code": exc.code}) from exc


@event_router.post("/resources:release", response_model=ResourceReleaseReceipt)
async def release_resource(body: ResourceLeaseReceipt, request: Request) -> ResourceReleaseReceipt:
    """连接与响应流清理完成后释放，不重放已经过期的执行授权。"""
    require_service_scope(request, "aiops.resources.release")
    service = ResourceGovernanceService(uow_factory=request.app.state.configuration_service._uow_factory)
    try:
        await service.release(body)
    except ResourceAdmissionError as exc:
        raise HTTPException(status_code=403, detail={"code": exc.code}) from exc
    return ResourceReleaseReceipt()


@event_router.post("/sql-plan-captures:exchange", response_model=SqlPlanCaptureExchange)
async def exchange_sql_plan_capture(body: SqlPlanCaptureExecute, request: Request):
    """仅隔离Executor持专用Grant一次兑换；普通输出序列化仍屏蔽保护字段。"""
    require_service_scope(request, "aiops.plan-capture.exchange")
    try:
        result = await request.app.state.sql_plan_capture_execution_service.exchange(body.grant)
    except (ExecutionFenceError, DiagnosticGrantError) as exc:
        raise HTTPException(status_code=403, detail={"code": exc.code}, headers={"Cache-Control": "no-store"}) from None
    except ResourceAdmissionError as exc:
        raise HTTPException(status_code=429, detail={"code": exc.code}, headers={"Cache-Control": "no-store"}) from None
    return JSONResponse(content=result.protected_transport(), headers={"Cache-Control": "no-store"})


@event_router.post("/sql-plan-captures:report", response_model=SqlPlanCaptureResult)
async def report_sql_plan_capture(body: SqlPlanCaptureReport, request: Request, response: Response):
    """独立Executor回传安全估算结果；回传权限不能消费材料或执行SQL。"""
    require_service_scope(request, "aiops.plan-capture.report")
    response.headers["Cache-Control"] = "no-store"
    try:
        return await request.app.state.sql_plan_capture_execution_service.report(body)
    except ExecutionFenceError as exc:
        raise HTTPException(status_code=403, detail={"code": exc.code}, headers={"Cache-Control": "no-store"}) from None


@event_router.post("/readonly-executions:check", response_model=ReadonlyExecutionFenceReceipt)
async def check_readonly_execution(body: ReadonlyExecutionFenceRequest, request: Request) -> ReadonlyExecutionFenceReceipt:
    """只向 Executor 返回当前执行围栏，不签发新的授权或连接材料。"""
    require_service_scope(request, "aiops.readonly.check")
    service = ReadonlyExecutionFenceService(
        uow_factory=request.app.state.configuration_service._uow_factory,
        diagnostic_codec=request.app.state.diagnostic_grant_codec,
        audience="kbot-aiops-db-executor",
        plan_captures=getattr(request.app.state, "sql_plan_capture_execution_service", None),
    )
    return await service.check(body)


@event_router.post("/readonly-executions:report-termination", response_model=ReadonlyTerminationReceipt)
async def report_readonly_termination(body: ReadonlyTerminationRequest, request: Request) -> ReadonlyTerminationReceipt:
    """已取消运行也接收只追加回执；独立 scope 不授予执行权。"""
    require_service_scope(request, "aiops.readonly.report")
    service = ReadonlyLifecycleService(uow_factory=request.app.state.configuration_service._uow_factory)
    try:
        return await service.record(body)
    except ReadonlyLifecycleError as exc:
        raise HTTPException(status_code=409 if exc.code == "READONLY_TERMINATION_CONFLICT" else 403,
            detail={"code": exc.code}) from exc
