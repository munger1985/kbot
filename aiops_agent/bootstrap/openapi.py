"""生成三套彼此隔离的 AIOps 契约 OpenAPI。"""

from uuid import UUID

from fastapi import FastAPI, HTTPException

from aiops_agent.api.management import router as internal_config_router
from aiops_agent.api.runtime import router as internal_runtime_router
from aiops_agent.api.intake import router as internal_intake_router
from aiops_agent.api.changes import router as internal_changes_router
from aiops_agent.api.executions import (
    event_router as internal_execution_events_router,
    router as internal_executions_router,
)
from main_api.api.ops import router as public_config_router
from main_api.api.integrations import router as public_integration_router
from platform_core.contracts.aiops.executor import (
    ExecutionResultRef,
    MutationExecutionRequest,
    ReadDiagnosticRequest,
    ReadDiagnosticResult,
)
from platform_core.contracts.aiops.internal import (
    CreateOpsRunCommand,
    OpsCommand,
    OpsRunReceipt as InternalOpsRunReceipt,
)


def _not_implemented() -> None:
    raise HTTPException(
        status_code=501,
        detail={"code": "CONTRACT_ONLY", "message": "仅用于冻结 OpenAPI 契约"},
    )


def create_public_contract_app() -> FastAPI:
    """创建 Main API 后续映射使用的公开契约快照 App。"""
    app = FastAPI(title="KBot AIOps Public Contract", version="1.0.0")
    app.include_router(public_config_router)
    app.include_router(public_integration_router)

    return app


def create_internal_contract_app() -> FastAPI:
    """创建 AIOps API 内部调用契约快照 App。"""
    app = FastAPI(title="KBot AIOps Internal Contract", version="1.0.0")
    app.include_router(internal_config_router)
    app.include_router(internal_runtime_router)
    app.include_router(internal_intake_router)
    app.include_router(internal_changes_router)
    app.include_router(internal_executions_router)
    app.include_router(internal_execution_events_router)

    return app


def create_executor_contract_app() -> FastAPI:
    """创建 AIOps DB Executor 隔离契约快照 App。"""
    app = FastAPI(title="KBot AIOps DB Executor Contract", version="1.0.0")

    @app.post(
        "/internal/v1/db-executor/diagnostics",
        response_model=ReadDiagnosticResult,
    )
    async def execute_diagnostic(payload: ReadDiagnosticRequest):
        _not_implemented()

    @app.post(
        "/internal/v1/db-executor/executions",
        response_model=ExecutionResultRef,
    )
    async def execute_mutation(payload: MutationExecutionRequest):
        _not_implemented()

    @app.get(
        "/internal/v1/db-executor/executions/{executor_request_id}",
        response_model=ExecutionResultRef,
    )
    async def execution_status(executor_request_id: UUID):
        _not_implemented()

    return app
