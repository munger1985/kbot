"""隔离 DB Executor 的内部只读诊断入口。"""

from fastapi import APIRouter, HTTPException, Request

from aiops_agent.api.dependencies import require_service_scope
from aiops_agent.actions import MutationGrantError
from aiops_agent.diagnostics.grants import DiagnosticGrantError
from aiops_agent.executor import MutationExecutionError
from platform_core.contracts.aiops.executor import (
    DynamicReadDiagnosticRequest,
    ExecutionResultRef,
    MutationExecutionRequest,
    ReadDiagnosticRequest,
    ReadDiagnosticResult,
)


router = APIRouter(prefix="/internal/v1/db-executor", tags=["DB Executor"])


@router.post("/diagnostics", response_model=ReadDiagnosticResult)
async def execute_diagnostic(
    payload: ReadDiagnosticRequest,
    request: Request,
) -> ReadDiagnosticResult:
    require_service_scope(request, "db-executor.diagnostic")
    try:
        return await request.app.state.diagnostic_executor.execute(payload)
    except DiagnosticGrantError as exc:
        raise HTTPException(
            status_code=403,
            detail={
                "code": exc.code,
                "message": "诊断执行授权无效或已过期",
            },
        ) from exc


@router.post("/dynamic-diagnostics", response_model=ReadDiagnosticResult)
async def execute_dynamic_diagnostic(
    payload: DynamicReadDiagnosticRequest,
    request: Request,
) -> ReadDiagnosticResult:
    """执行经过双端策略验证的 Oracle 动态只读查询。"""
    require_service_scope(request, "db-executor.diagnostic")
    try:
        return await request.app.state.dynamic_diagnostic_executor.execute(
            payload
        )
    except DiagnosticGrantError as exc:
        raise HTTPException(
            status_code=403,
            detail={
                "code": exc.code,
                "message": "动态诊断执行授权无效或已过期",
            },
        ) from exc


@router.post("/executions", response_model=ExecutionResultRef)
async def execute_mutation(
    payload: MutationExecutionRequest,
    request: Request,
) -> ExecutionResultRef:
    require_service_scope(request, "db-executor.mutation")
    context = request.state.auth_context
    try:
        return await request.app.state.mutation_executor.execute(
            payload,
            trace_id=context.trace_id,
        )
    except MutationGrantError as exc:
        raise HTTPException(
            status_code=403,
            detail={
                "code": exc.code,
                "message": "变更执行授权无效或已过期",
            },
        ) from exc
    except MutationExecutionError as exc:
        status = 503 if exc.code == "MUTATION_DISABLED" else 409
        raise HTTPException(
            status_code=status,
            detail={
                "code": exc.code,
                "message": "变更执行请求未通过安全围栏",
            },
        ) from exc
