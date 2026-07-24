"""隔离 DB Executor 的内部只读诊断入口。"""

from fastapi import APIRouter, HTTPException, Request

from aiops_agent.api.dependencies import require_service_scope
from aiops_agent.diagnostics.grants import DiagnosticGrantError
from platform_core.contracts.aiops.executor import (
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
