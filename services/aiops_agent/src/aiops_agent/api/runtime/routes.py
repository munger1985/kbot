"""AIOps Run 与 Worker Internal API。"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request

from aiops_agent.api.dependencies import (
    get_aiops_auth_context,
    require_service_scope,
)
from aiops_agent.application.runtime import AIOpsRuntimeService
from aiops_agent.application.configuration.common import ConfigurationScope
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops import (
    ClaimOpsTaskCommand,
    CompleteOpsTaskCommand,
    CreateOpsRunCommand,
    DelegationEventPage,
    DiagnosticQueryApprovalDecision,
    FailOpsTaskCommand,
    HeartbeatOpsTaskCommand,
    HitlResponse,
    HitlResult,
    HitlSkipCommand,
    OpsCommand,
    OpsRunEventPage,
    PendingInputView,
    RootDelegationRequest,
    RootDelegationResult,
    TaskLease,
    TaskMutationReceipt,
)
from platform_core.contracts.aiops.internal import (
    OpsRunReceipt,
    RootDelegationReceipt,
)
from platform_core.contracts.aiops.public import (
    InspectionFirePage,
    InspectionFireView,
    OpsRunResult,
    OpsRunPage,
    OpsRunSummary,
    ReportPage,
    ReportVersionPage,
    ReportView,
    SituationPage,
    SituationView,
)


router = APIRouter(prefix="/internal/v1/aiops", tags=["AIOps Runtime"])


def get_service(request: Request) -> AIOpsRuntimeService:
    return request.app.state.aiops_runtime_service


Service = Annotated[AIOpsRuntimeService, Depends(get_service)]
Auth = Annotated[AuthContext, Depends(get_aiops_auth_context)]


def _scope(request: Request, context: AuthContext) -> int:
    if context.domain_id is None:
        raise RuntimeError("AIOps 请求缺少 Domain")
    try:
        domain_id = int(context.domain_id)
    except ValueError as exc:
        raise RuntimeError("AIOps Domain 必须是数字标识") from exc
    return domain_id


def _query_scope(
    request: Request, context: AuthContext
) -> ConfigurationScope:
    return ConfigurationScope.from_auth(
        auth_context=context,
    )


def _ensure_agent_authorized(
    context: AuthContext, agent_id: UUID
) -> None:
    if (
        context.authorized_agent_ids
        and agent_id not in context.authorized_agent_ids
    ):
        raise HTTPException(
            status_code=404,
            detail={
                "code": "OPS_RESOURCE_NOT_FOUND",
                "message": "Ops Run 不存在",
            },
        )


@router.post("/runs", response_model=OpsRunReceipt, status_code=201)
async def create_run(
    body: CreateOpsRunCommand,
    request: Request,
    service: Service,
    context: Auth,
) -> OpsRunReceipt:
    require_service_scope(request, "aiops.run")
    domain_id = _scope(request, context)
    _ensure_agent_authorized(context, body.agent_id)
    command = body.model_copy(
        update={
            "domain_id": domain_id,
            "actor_id": context.asserted_user_id or context.client_id,
        }
    )
    return await service.create_run(command)


@router.post(
    "/delegations",
    response_model=RootDelegationReceipt,
    status_code=201,
)
async def create_delegation(
    body: RootDelegationRequest,
    request: Request,
    service: Service,
    context: Auth,
) -> RootDelegationReceipt:
    require_service_scope(request, "aiops.delegate")
    domain_id = _scope(request, context)
    return await service.create_delegated_run(
        request=body,
        domain_id=domain_id,
        actor_id=context.asserted_user_id or context.client_id,
        trace_id=context.trace_id,
    )


@router.get(
    "/delegations/{delegation_id}/events",
    response_model=DelegationEventPage,
)
async def list_delegation_events(
    delegation_id: UUID,
    request: Request,
    service: Service,
    context: Auth,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=200),
) -> DelegationEventPage:
    require_service_scope(request, "aiops.delegate")
    domain_id = _scope(request, context)
    return await service.list_delegation_events(
        delegation_id=delegation_id,
        domain_id=domain_id,
        after_sequence=after,
        limit=limit,
    )


@router.get(
    "/delegations/{delegation_id}/result",
    response_model=RootDelegationResult,
)
async def get_delegation_result(
    delegation_id: UUID,
    request: Request,
    service: Service,
    context: Auth,
) -> RootDelegationResult:
    require_service_scope(request, "aiops.delegate")
    domain_id = _scope(request, context)
    return await service.get_delegation_result(
        delegation_id=delegation_id,
        domain_id=domain_id,
    )


@router.post(
    "/delegations/{delegation_id}/cancel",
    response_model=OpsRunReceipt,
)
async def cancel_delegation(
    delegation_id: UUID,
    request: Request,
    service: Service,
    context: Auth,
    idempotency_key: str = Header(alias="Idempotency-Key"),
) -> OpsRunReceipt:
    require_service_scope(request, "aiops.delegate")
    domain_id = _scope(request, context)
    return await service.cancel_delegation(
        delegation_id=delegation_id,
        domain_id=domain_id,
        actor_id=context.asserted_user_id or context.client_id,
        idempotency_key=idempotency_key,
        trace_id=context.trace_id,
    )


@router.get("/runs", response_model=OpsRunPage)
async def list_runs(
    request: Request, service: Service, context: Auth,
    target_id: UUID | None = None, status: str | None = None,
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> OpsRunPage:
    require_service_scope(request, "aiops.run")
    return await service.list_runs(
        scope=_query_scope(request, context), target_id=target_id,
        status=status, agent_ids=tuple(context.authorized_agent_ids),
        cursor=cursor, limit=limit,
    )


@router.get("/situations", response_model=SituationPage)
async def list_situations(
    request: Request, service: Service, context: Auth,
    target_id: UUID | None = None, status: str | None = None,
    severity: str | None = None,
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> SituationPage:
    require_service_scope(request, "aiops.run")
    return await service.list_situations(
        scope=_query_scope(request, context), target_id=target_id,
        status=status, severity=severity, cursor=cursor, limit=limit,
    )


@router.get("/situations/{situation_id}", response_model=SituationView)
async def get_situation(
    situation_id: UUID, request: Request, service: Service, context: Auth,
) -> SituationView:
    require_service_scope(request, "aiops.run")
    return await service.get_situation(
        situation_id=situation_id, domain_id=_scope(request, context)
    )


@router.get("/runs/{run_id}", response_model=OpsRunSummary)
async def get_run(
    run_id: UUID,
    request: Request,
    service: Service,
    context: Auth,
) -> OpsRunSummary:
    require_service_scope(request, "aiops.run")
    domain_id = _scope(request, context)
    result = await service.get_run(
        ops_run_id=run_id, domain_id=domain_id
    )
    _ensure_agent_authorized(context, result.agent_id)
    return result


@router.get("/runs/{run_id}/result", response_model=OpsRunResult)
async def get_run_result(
    run_id: UUID,
    request: Request,
    service: Service,
    context: Auth,
) -> OpsRunResult:
    require_service_scope(request, "aiops.run")
    domain_id = _scope(request, context)
    summary = await service.get_run(
        ops_run_id=run_id,
        domain_id=domain_id,
    )
    _ensure_agent_authorized(context, summary.agent_id)
    return await service.get_run_result(
        ops_run_id=run_id,
        domain_id=domain_id,
    )


@router.get("/reports/{report_id}", response_model=ReportView)
async def get_report(
    report_id: UUID,
    request: Request,
    service: Service,
    context: Auth,
) -> ReportView:
    require_service_scope(request, "aiops.run")
    domain_id = _scope(request, context)
    return await service.get_report(
        report_id=report_id,
        domain_id=domain_id,
    )


@router.get("/reports", response_model=ReportPage)
async def list_reports(
    request: Request,
    service: Service,
    context: Auth,
    target_id: UUID | None = None,
    report_type: str | None = None,
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> ReportPage:
    require_service_scope(request, "aiops.run")
    return await service.list_reports(
        scope=_query_scope(request, context),
        target_id=target_id,
        report_type=report_type,
        cursor=cursor,
        limit=limit,
    )


@router.get(
    "/reports/{report_id}/versions",
    response_model=ReportVersionPage,
)
async def list_report_versions(
    report_id: UUID,
    request: Request,
    service: Service,
    context: Auth,
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> ReportVersionPage:
    require_service_scope(request, "aiops.run")
    return await service.list_report_versions(
        scope=_query_scope(request, context),
        report_id=report_id,
        cursor=cursor,
        limit=limit,
    )


@router.get("/inspection-fires", response_model=InspectionFirePage)
async def list_inspection_fires(
    request: Request,
    service: Service,
    context: Auth,
    plan_id: UUID | None = None,
    status: str | None = None,
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> InspectionFirePage:
    require_service_scope(request, "aiops.run")
    return await service.list_inspection_fires(
        scope=_query_scope(request, context),
        plan_id=plan_id,
        status=status,
        cursor=cursor,
        limit=limit,
    )


@router.get(
    "/inspection-fires/{fire_id}",
    response_model=InspectionFireView,
)
async def get_inspection_fire(
    fire_id: UUID,
    request: Request,
    service: Service,
    context: Auth,
) -> InspectionFireView:
    require_service_scope(request, "aiops.run")
    domain_id = _scope(request, context)
    return await service.get_inspection_fire(
        inspection_fire_id=fire_id,
        domain_id=domain_id,
    )


@router.get(
    "/runs/{run_id}/events",
    response_model=OpsRunEventPage,
)
async def list_events(
    run_id: UUID,
    request: Request,
    service: Service,
    context: Auth,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=200),
) -> OpsRunEventPage:
    identity = require_service_scope(request, "aiops.run")
    domain_id = _scope(request, context)
    summary = await service.get_run(
        ops_run_id=run_id, domain_id=domain_id
    )
    _ensure_agent_authorized(context, summary.agent_id)
    return await service.list_events(
        ops_run_id=run_id,
        domain_id=domain_id,
        after_sequence=after,
        user_only=identity.subject == "kbot-main-api",
        limit=limit,
    )


@router.get(
    "/runs/{run_id}/pending-input",
    response_model=PendingInputView,
)
async def get_pending_input(
    run_id: UUID,
    request: Request,
    service: Service,
    context: Auth,
) -> PendingInputView:
    require_service_scope(request, "aiops.hitl")
    domain_id = _scope(request, context)
    return await service.get_pending_input(
        ops_run_id=run_id,
        domain_id=domain_id,
        actor_id=context.asserted_user_id or context.client_id,
    )


@router.get("/hitl/{hitl_id}", response_model=PendingInputView)
async def get_hitl_input(
    hitl_id: UUID,
    request: Request,
    service: Service,
    context: Auth,
) -> PendingInputView:
    require_service_scope(request, "aiops.hitl")
    domain_id = _scope(request, context)
    return await service.get_hitl_input(
        hitl_id=hitl_id,
        domain_id=domain_id,
        actor_id=context.asserted_user_id or context.client_id,
    )


@router.post("/hitl/{hitl_id}/response", response_model=HitlResult)
async def respond_hitl(
    hitl_id: UUID,
    body: HitlResponse,
    request: Request,
    service: Service,
    context: Auth,
) -> HitlResult:
    require_service_scope(request, "aiops.hitl")
    domain_id = _scope(request, context)
    return await service.respond_hitl(
        hitl_id=hitl_id,
        domain_id=domain_id,
        actor_id=context.asserted_user_id or context.client_id,
        response=body,
        idempotency_key=request.headers.get(
            "Idempotency-Key", str(body.expected_row_version)
        ),
        trace_id=context.trace_id,
    )


@router.post("/hitl/{hitl_id}/decision", response_model=HitlResult)
async def decide_diagnostic_query(
    hitl_id: UUID,
    body: DiagnosticQueryApprovalDecision,
    request: Request,
    service: Service,
    context: Auth,
) -> HitlResult:
    require_service_scope(request, "aiops.hitl")
    domain_id = _scope(request, context)
    return await service.decide_diagnostic_query(
        hitl_id=hitl_id,
        domain_id=domain_id,
        actor_id=context.asserted_user_id or context.client_id,
        decision=body,
        idempotency_key=request.headers.get(
            "Idempotency-Key", str(body.expected_row_version)
        ),
        trace_id=context.trace_id,
    )


@router.post("/hitl/{hitl_id}/skip", response_model=HitlResult)
async def skip_hitl(
    hitl_id: UUID,
    body: HitlSkipCommand,
    request: Request,
    service: Service,
    context: Auth,
) -> HitlResult:
    require_service_scope(request, "aiops.hitl")
    domain_id = _scope(request, context)
    return await service.skip_hitl(
        hitl_id=hitl_id,
        domain_id=domain_id,
        actor_id=context.asserted_user_id or context.client_id,
        expected_row_version=body.expected_row_version,
        idempotency_key=request.headers.get(
            "Idempotency-Key", str(body.expected_row_version)
        ),
        trace_id=context.trace_id,
    )


@router.post("/runs/{run_id}/commands", response_model=OpsRunReceipt)
async def command_run(
    run_id: UUID,
    body: OpsCommand,
    request: Request,
    service: Service,
    context: Auth,
) -> OpsRunReceipt:
    require_service_scope(request, "aiops.run")
    if body.ops_run_id != run_id:
        raise ValueError("Path Run ID 与命令不一致")
    if body.command.command_type != "CANCEL_RUN":
        raise ValueError("步骤 4 仅支持 CANCEL_RUN")
    domain_id = _scope(request, context)
    summary = await service.get_run(
        ops_run_id=run_id, domain_id=domain_id
    )
    _ensure_agent_authorized(context, summary.agent_id)
    return await service.request_cancel(
        ops_run_id=run_id,
        domain_id=domain_id,
        actor_id=context.asserted_user_id or context.client_id,
        expected_row_version=body.command.expected_row_version,
        idempotency_key=body.idempotency_key,
        trace_id=context.trace_id,
    )


@router.post("/tasks/claim", response_model=TaskLease | None)
async def claim_task(
    body: ClaimOpsTaskCommand,
    request: Request,
    service: Service,
) -> TaskLease | None:
    require_service_scope(request, "aiops.task")
    return await service.claim_task(body)


@router.post("/tasks/heartbeat", response_model=TaskLease)
async def heartbeat_task(
    body: HeartbeatOpsTaskCommand,
    request: Request,
    service: Service,
) -> TaskLease:
    require_service_scope(request, "aiops.task")
    return await service.heartbeat_task(body)


@router.post("/tasks/complete", response_model=TaskMutationReceipt)
async def complete_task(
    body: CompleteOpsTaskCommand,
    request: Request,
    service: Service,
) -> TaskMutationReceipt:
    require_service_scope(request, "aiops.task")
    return await service.complete_task(body)


@router.post("/tasks/fail", response_model=TaskMutationReceipt)
async def fail_task(
    body: FailOpsTaskCommand,
    request: Request,
    service: Service,
) -> TaskMutationReceipt:
    require_service_scope(request, "aiops.task")
    return await service.fail_task(body)
