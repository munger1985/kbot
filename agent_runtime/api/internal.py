"""Agent Runtime 内部命令与查询接口。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.application import (
    AgentRuntimeConflict,
    AgentRuntimeNotFound,
    ArtifactInput,
    CancelRunCommand,
    ClaimTaskCommand,
    CompleteTaskCommand,
    CreateRunCommand,
    FailTaskCommand,
    HeartbeatTaskCommand,
    InstallPlanCommand,
    StaleTaskLease,
)
from agent_runtime.domain.planning import PlanDraft
from agent_runtime.domain.planning import PlanValidationError
from agent_runtime.domain.state_machine import InvalidStateTransition
from platform_core.contracts import (
    AgentRunEvent,
    AgentRunReceipt,
    AgentRunSummary,
    CreateAgentRunRequest,
    INTERNAL_API_V1,
)


router = APIRouter(prefix=f"{INTERNAL_API_V1}/runs", tags=["Agent Runtime"])


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InstallPlanRequest(_RequestModel):
    expected_row_version: int = Field(ge=1)
    plan: PlanDraft


class ClaimTaskRequest(_RequestModel):
    worker_id: str = Field(min_length=1, max_length=256)
    lease_seconds: int = Field(default=120, ge=15, le=3600)


class HeartbeatTaskRequest(_RequestModel):
    expected_row_version: int = Field(ge=1)
    worker_id: str = Field(min_length=1, max_length=256)
    lease_token: UUID
    lease_seconds: int = Field(default=120, ge=15, le=3600)


class CompleteTaskRequest(_RequestModel):
    expected_row_version: int = Field(ge=1)
    worker_id: str = Field(min_length=1, max_length=256)
    lease_token: UUID
    artifact: ArtifactInput


class FailTaskRequest(_RequestModel):
    expected_row_version: int = Field(ge=1)
    worker_id: str = Field(min_length=1, max_length=256)
    lease_token: UUID
    error_code: str = Field(min_length=1, max_length=128)
    error_message: str = Field(min_length=1, max_length=1000)
    retryable: bool = False
    retry_at: datetime | None = None


class CancelRunRequest(_RequestModel):
    expected_row_version: int = Field(ge=1)


def _service(request: Request):
    service = getattr(request.app.state, "agent_runtime_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "RUNTIME_NOT_READY",
                "message": "Agent Runtime 尚未初始化",
            },
        )
    return service


def _identity(request: Request) -> tuple[int, str, str, str]:
    context = getattr(request.state, "auth_context", None)
    if context is None or not context.domain_id:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "DOMAIN_CONTEXT_REQUIRED",
                "message": "当前命令缺少受信 Domain 上下文",
            },
        )
    try:
        domain_id = int(context.domain_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_DOMAIN",
                "message": "Domain ID 必须是数字标识",
            },
        ) from exc
    actor_id = context.asserted_user_id or context.client_id
    return domain_id, actor_id, context.request_id, context.trace_id


def _service_identity(request: Request) -> tuple[str, str]:
    context = getattr(request.state, "auth_context", None)
    if context is None:
        raise HTTPException(status_code=401, detail="缺少内部身份上下文")
    return context.client_id, context.trace_id


def _raise_runtime_error(exc: Exception) -> None:
    if isinstance(exc, AgentRuntimeNotFound):
        status = 404
    elif isinstance(exc, StaleTaskLease):
        status = 409
    elif isinstance(exc, AgentRuntimeConflict):
        status = (
            422
            if exc.code.startswith("ARTIFACT_")
            else 409
        )
    elif isinstance(exc, InvalidStateTransition):
        status = 409
        exc = AgentRuntimeConflict("STATE_CONFLICT", str(exc))
    elif isinstance(exc, PlanValidationError):
        status = 429 if exc.code == "BUDGET_EXCEEDED" else 422
    else:
        raise exc
    raise HTTPException(
        status_code=status,
        detail={"code": exc.code, "message": str(exc)},
    )


@router.post("", status_code=202, response_model=AgentRunReceipt)
async def create_run(
    payload: CreateAgentRunRequest,
    request: Request,
    idempotency_key: str = Header(alias="Idempotency-Key"),
) -> AgentRunReceipt:
    domain_id, actor_id, request_id, trace_id = _identity(request)
    try:
        return await _service(request).create_run(
            CreateRunCommand(
                app_id=request.app.state.platform_app_id,
                domain_id=domain_id,
                agent_id=payload.agent_id,
                actor_id=actor_id,
                request_id=request_id,
                trace_id=trace_id,
                idempotency_key=idempotency_key,
                original_input=payload.input,
                collection_ids=payload.collection_ids,
                security_level=payload.security_level,
                client_metadata=payload.client_metadata,
                policy_snapshot={},
                budget=request.app.state.agent_runtime_budget,
            )
        )
    except (AgentRuntimeConflict, AgentRuntimeNotFound) as exc:
        _raise_runtime_error(exc)


@router.post(
    "/{run_id}/plan",
    status_code=202,
    response_model=AgentRunReceipt,
)
async def install_plan(
    run_id: UUID,
    payload: InstallPlanRequest,
    request: Request,
    idempotency_key: str = Header(alias="Idempotency-Key"),
) -> AgentRunReceipt:
    domain_id, actor_id, _, trace_id = _identity(request)
    try:
        return await _service(request).install_plan(
            InstallPlanCommand(
                app_id=request.app.state.platform_app_id,
                domain_id=domain_id,
                run_id=run_id,
                expected_row_version=payload.expected_row_version,
                plan=payload.plan,
                actor_id=actor_id,
                trace_id=trace_id,
                idempotency_key=idempotency_key,
            )
        )
    except (
        AgentRuntimeConflict,
        AgentRuntimeNotFound,
        InvalidStateTransition,
        PlanValidationError,
    ) as exc:
        _raise_runtime_error(exc)


@router.get("/{run_id}", response_model=AgentRunSummary)
async def get_run(run_id: UUID, request: Request) -> AgentRunSummary:
    domain_id, _, _, _ = _identity(request)
    try:
        return await _service(request).get_run(
            run_id=run_id,
            app_id=request.app.state.platform_app_id,
            domain_id=domain_id,
        )
    except AgentRuntimeNotFound as exc:
        _raise_runtime_error(exc)


@router.get("/{run_id}/events", response_model=list[AgentRunEvent])
async def list_events(
    run_id: UUID,
    request: Request,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=500),
) -> list[AgentRunEvent]:
    domain_id, _, _, _ = _identity(request)
    try:
        return await _service(request).list_events(
            run_id=run_id,
            app_id=request.app.state.platform_app_id,
            domain_id=domain_id,
            after_sequence=after,
            limit=limit,
        )
    except AgentRuntimeNotFound as exc:
        _raise_runtime_error(exc)


@router.post(
    "/{run_id}/cancel",
    status_code=202,
    response_model=AgentRunReceipt,
)
async def cancel_run(
    run_id: UUID,
    payload: CancelRunRequest,
    request: Request,
    idempotency_key: str = Header(alias="Idempotency-Key"),
) -> AgentRunReceipt:
    domain_id, actor_id, _, trace_id = _identity(request)
    try:
        return await _service(request).cancel_run(
            CancelRunCommand(
                app_id=request.app.state.platform_app_id,
                domain_id=domain_id,
                run_id=run_id,
                expected_row_version=payload.expected_row_version,
                actor_id=actor_id,
                trace_id=trace_id,
                idempotency_key=idempotency_key,
            )
        )
    except (
        AgentRuntimeConflict,
        AgentRuntimeNotFound,
        InvalidStateTransition,
    ) as exc:
        _raise_runtime_error(exc)


task_router = APIRouter(
    prefix=f"{INTERNAL_API_V1}/tasks", tags=["Agent Runtime Worker"]
)


@task_router.post("/claim")
async def claim_task(payload: ClaimTaskRequest, request: Request):
    _, trace_id = _service_identity(request)
    try:
        lease = await _service(request).claim_task(
            ClaimTaskCommand(
                worker_id=payload.worker_id,
                lease_seconds=payload.lease_seconds,
                trace_id=trace_id,
            )
        )
    except (
        AgentRuntimeConflict,
        InvalidStateTransition,
    ) as exc:
        _raise_runtime_error(exc)
    if lease is None:
        return Response(status_code=204)
    return lease


@task_router.post("/{task_id}/heartbeat")
async def heartbeat_task(
    task_id: UUID, payload: HeartbeatTaskRequest, request: Request
):
    try:
        return await _service(request).heartbeat_task(
            HeartbeatTaskCommand(task_id=task_id, **payload.model_dump())
        )
    except (AgentRuntimeConflict, StaleTaskLease) as exc:
        _raise_runtime_error(exc)


@task_router.post("/{task_id}/complete")
async def complete_task(
    task_id: UUID,
    payload: CompleteTaskRequest,
    request: Request,
    idempotency_key: str = Header(alias="Idempotency-Key"),
):
    actor_id, trace_id = _service_identity(request)
    try:
        return await _service(request).complete_task(
            CompleteTaskCommand(
                task_id=task_id,
                actor_id=actor_id,
                trace_id=trace_id,
                idempotency_key=idempotency_key,
                **payload.model_dump(),
            )
        )
    except (
        AgentRuntimeConflict,
        AgentRuntimeNotFound,
        StaleTaskLease,
        InvalidStateTransition,
    ) as exc:
        _raise_runtime_error(exc)


@task_router.post("/{task_id}/fail")
async def fail_task(
    task_id: UUID,
    payload: FailTaskRequest,
    request: Request,
    idempotency_key: str = Header(alias="Idempotency-Key"),
):
    actor_id, trace_id = _service_identity(request)
    try:
        return await _service(request).fail_task(
            FailTaskCommand(
                task_id=task_id,
                actor_id=actor_id,
                trace_id=trace_id,
                idempotency_key=idempotency_key,
                **payload.model_dump(),
            )
        )
    except (
        AgentRuntimeConflict,
        AgentRuntimeNotFound,
        StaleTaskLease,
        InvalidStateTransition,
    ) as exc:
        _raise_runtime_error(exc)
