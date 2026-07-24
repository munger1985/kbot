"""AIOps Run 与 Worker Internal API。"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from aiops_agent.api.dependencies import (
    get_aiops_auth_context,
    require_service_scope,
)
from aiops_agent.application.runtime import AIOpsRuntimeService
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops import (
    ClaimOpsTaskCommand,
    CompleteOpsTaskCommand,
    CreateOpsRunCommand,
    FailOpsTaskCommand,
    HeartbeatOpsTaskCommand,
    HitlResponse,
    HitlResult,
    HitlSkipCommand,
    OpsCommand,
    OpsRunEventPage,
    PendingInputView,
    TaskLease,
    TaskMutationReceipt,
)
from platform_core.contracts.aiops.internal import OpsRunReceipt
from platform_core.contracts.aiops.public import OpsRunSummary


router = APIRouter(prefix="/internal/v1/aiops", tags=["AIOps Runtime"])


def get_service(request: Request) -> AIOpsRuntimeService:
    return request.app.state.aiops_runtime_service


Service = Annotated[AIOpsRuntimeService, Depends(get_service)]
Auth = Annotated[AuthContext, Depends(get_aiops_auth_context)]


def _scope(request: Request, context: AuthContext) -> tuple[int, int]:
    if context.domain_id is None:
        raise RuntimeError("AIOps 请求缺少 Domain")
    try:
        domain_id = int(context.domain_id)
    except ValueError as exc:
        raise RuntimeError("AIOps Domain 必须是数字标识") from exc
    return request.app.state.runtime.settings.platform.app_id, domain_id


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
    app_id, domain_id = _scope(request, context)
    _ensure_agent_authorized(context, body.agent_id)
    command = body.model_copy(
        update={
            "app_id": app_id,
            "domain_id": domain_id,
            "actor_id": context.asserted_user_id or context.client_id,
        }
    )
    return await service.create_run(command)


@router.get("/runs/{run_id}", response_model=OpsRunSummary)
async def get_run(
    run_id: UUID,
    request: Request,
    service: Service,
    context: Auth,
) -> OpsRunSummary:
    require_service_scope(request, "aiops.run")
    app_id, domain_id = _scope(request, context)
    result = await service.get_run(
        ops_run_id=run_id, app_id=app_id, domain_id=domain_id
    )
    _ensure_agent_authorized(context, result.agent_id)
    return result


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
    app_id, domain_id = _scope(request, context)
    summary = await service.get_run(
        ops_run_id=run_id, app_id=app_id, domain_id=domain_id
    )
    _ensure_agent_authorized(context, summary.agent_id)
    return await service.list_events(
        ops_run_id=run_id,
        app_id=app_id,
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
    app_id, domain_id = _scope(request, context)
    return await service.get_pending_input(
        ops_run_id=run_id,
        app_id=app_id,
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
    app_id, domain_id = _scope(request, context)
    return await service.get_hitl_input(
        hitl_id=hitl_id,
        app_id=app_id,
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
    app_id, domain_id = _scope(request, context)
    return await service.respond_hitl(
        hitl_id=hitl_id,
        app_id=app_id,
        domain_id=domain_id,
        actor_id=context.asserted_user_id or context.client_id,
        response=body,
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
    app_id, domain_id = _scope(request, context)
    return await service.skip_hitl(
        hitl_id=hitl_id,
        app_id=app_id,
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
    app_id, domain_id = _scope(request, context)
    summary = await service.get_run(
        ops_run_id=run_id, app_id=app_id, domain_id=domain_id
    )
    _ensure_agent_authorized(context, summary.agent_id)
    return await service.request_cancel(
        ops_run_id=run_id,
        app_id=app_id,
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
