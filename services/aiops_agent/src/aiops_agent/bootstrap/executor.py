"""AIOps DB Executor Bootstrap。"""

from contextlib import asynccontextmanager

import aiohttp
from loguru import logger
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from aiops_agent.adapters.aiops_execution_client import AIOpsExecutionClient
from aiops_agent.actions import ActionRegistry, create_mutation_grant_codec
from aiops_agent.api.executor import router as executor_router
from aiops_agent.bootstrap.common import (
    AIOpsProcessRuntime,
    configure_process_logging,
    create_process_app,
)
from aiops_agent.config import AIOpsSettings, get_aiops_settings
from aiops_agent.diagnostics import (
    create_diagnostic_grant_codec,
    create_diagnostic_registry,
)
from aiops_agent.executor import (
    DiagnosticExecutorService,
    DynamicDiagnosticExecutorService,
    MutationExecutorService,
)
from aiops_agent.executor.drivers import (
    MySQLDiagnosticDriver,
    MySQLMutationDriver,
    OracleDiagnosticDriver,
    PostgreSQLDiagnosticDriver,
    OracleMutationDriver,
)
from platform_core.contracts.aiops.executor import DiagnosticLimits
from platform_core.security import (
    create_auth_context_codec,
    create_scoped_internal_auth_middleware,
    create_service_identity_codec,
)


def create_aiops_executor(
    settings: AIOpsSettings | None = None,
):
    """创建不持有 KBot Schema 凭据的隔离 DB Executor。"""
    resolved = settings or get_aiops_settings()
    config = resolved.executor

    @asynccontextmanager
    async def lifespan(app):
        configure_process_logging(
            resolved,
            process="db_executor",
        )
        registry = create_diagnostic_registry(resolved)
        action_registry = ActionRegistry.load()
        client_session = aiohttp.ClientSession()
        oracle_driver = OracleDiagnosticDriver()
        diagnostic_executor = DiagnosticExecutorService(
            registry=registry,
            grant_codec=create_diagnostic_grant_codec(resolved),
            control_plane=AIOpsExecutionClient(
                base_url=resolved.clients.aiops_api.base_url,
                audience=resolved.clients.aiops_api.audience,
                caller_service=config.service_name,
                timeout_seconds=resolved.clients.aiops_api.timeout_seconds,
                session=client_session,
            ),
            drivers=(
                oracle_driver,
                MySQLDiagnosticDriver(),
                PostgreSQLDiagnosticDriver(),
            ),
            hard_limits=DiagnosticLimits(
                statement_timeout_seconds=config.statement_timeout_seconds,
                max_result_rows=config.max_result_rows,
                max_result_bytes=config.max_result_bytes,
                max_columns=config.max_result_columns,
                max_cell_chars=config.max_cell_chars,
            ),
            concurrency=config.readonly_concurrency,
        )
        dynamic_diagnostic_executor = DynamicDiagnosticExecutorService(
            grant_codec=create_diagnostic_grant_codec(resolved),
            control_plane=AIOpsExecutionClient(
                base_url=resolved.clients.aiops_api.base_url,
                audience=resolved.clients.aiops_api.audience,
                caller_service=config.service_name,
                timeout_seconds=resolved.clients.aiops_api.timeout_seconds,
                session=client_session,
            ),
            oracle_driver=oracle_driver,
            hard_limits=DiagnosticLimits(
                statement_timeout_seconds=config.statement_timeout_seconds,
                max_result_rows=config.max_result_rows,
                max_result_bytes=config.max_result_bytes,
                max_columns=config.max_result_columns,
                max_cell_chars=config.max_cell_chars,
            ),
            concurrency=config.readonly_concurrency,
        )
        mutation_executor = MutationExecutorService(
            enabled=config.mutation_enabled,
            executor_instance_id=config.executor_instance_id,
            registry=action_registry,
            grant_codec=create_mutation_grant_codec(resolved),
            control_plane=AIOpsExecutionClient(
                base_url=resolved.clients.aiops_api.base_url,
                audience=resolved.clients.aiops_api.audience,
                caller_service=config.service_name,
                timeout_seconds=(
                    resolved.clients.aiops_api.timeout_seconds
                ),
                session=client_session,
            ),
            drivers=(
                OracleMutationDriver(),
                MySQLMutationDriver(),
            ),
            concurrency=config.mutation_concurrency,
        )
        runtime = AIOpsProcessRuntime(
            settings=resolved,
            service_name=config.service_name,
            components={
                "template_registry": True,
                "secret_provider": True,
                "identity_verifier": True,
            },
        )
        app.state.runtime = runtime
        app.state.ready_check = runtime.check_executor_components
        app.state.auth_context_codec = create_auth_context_codec()
        app.state.service_identity_codec = create_service_identity_codec()
        app.state.diagnostic_executor = diagnostic_executor
        app.state.dynamic_diagnostic_executor = dynamic_diagnostic_executor
        app.state.mutation_executor = mutation_executor
        await runtime.start()
        logger.info(
            "正在启动 AIOps DB Executor：catalog_hash={} Mutation={}",
            registry.catalog_hash,
            config.mutation_enabled,
        )
        try:
            yield
        finally:
            await client_session.close()
            await runtime.close()
            logger.info("AIOps DB Executor 资源已释放")

    app = create_process_app(
        title="KBot AIOps DB Executor Internal API",
        description="只接受类型化诊断和变更模板的高权限执行边界。",
        service_name=config.service_name,
        service_version=config.service_version,
        debug=resolved.platform.debug,
        lifespan=lifespan,
    )
    app.middleware("http")(
        create_scoped_internal_auth_middleware(
            audience=config.service_name,
            allowed_callers={
                "kbot-aiops-worker": frozenset(
                    {
                        "db-executor.diagnostic",
                        "db-executor.mutation",
                        "db-executor.status.read",
                    }
                )
            },
        )
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request, exc):
        del request, exc
        return JSONResponse(
            status_code=400,
            content={
                "code": "EXECUTOR_REQUEST_INVALID",
                "message": "DB Executor 请求契约无效",
            },
        )

    app.include_router(executor_router)
    return app
