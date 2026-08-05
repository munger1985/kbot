"""Data Query API Bootstrap。"""

from data_query.bootstrap.common import (
    DataQueryProcessRuntime,
    configure_process_logging,
    create_process_app,
)
from data_query.api import management_router, model_reference_router, runtime_router
from data_query.application import (
    DataQueryManagementError,
    DataSourceConnectionError,
    DataQueryManagementService,
    DataQueryRunError,
    DataQueryRuntimeService,
    SemanticModelPublicationError,
    SchemaMetadataError,
    SemanticModelValidationError,
)
from data_query.config import DataQuerySettings, get_data_query_settings
from data_query.adapters import CredentialCipher, DatabaseCredentialService
from data_query.connectors.connection_tester import test_data_source_connection
from platform_core.database import create_database_runtime
from data_query.persistence import create_data_query_uow_factory
from platform_core.security import (
    create_auth_context_codec,
    create_scoped_internal_auth_middleware,
    create_service_identity_codec,
)
from platform_clients import AIModelClient, AIModelConfigClient
from fastapi import Request
from fastapi.responses import JSONResponse


def create_data_query_api(settings: DataQuerySettings | None = None):
    """创建受内部服务身份保护的 Data Query API。"""
    resolved = settings or get_data_query_settings()
    configure_process_logging(resolved, process="api")
    database_runtime = create_database_runtime(resolved)
    uow_factory = create_data_query_uow_factory(
        database_runtime.session_factory
    )
    runtime = DataQueryProcessRuntime(
        settings=resolved,
        service_name=resolved.api.service_name,
        database_runtime=database_runtime,
        uow_factory=uow_factory,
    )
    app = create_process_app(
        title="KBot Data Query API",
        description="Data Query 管理与运行服务。",
        runtime=runtime,
        debug=resolved.platform.debug,
    )
    app.state.auth_context_codec = create_auth_context_codec()
    app.state.service_identity_codec = create_service_identity_codec()
    app.state.uow_factory = uow_factory
    app.state.management_service = DataQueryManagementService(
        uow_factory=uow_factory,
        credential_service=DatabaseCredentialService(
            uow_factory=uow_factory,
            cipher=CredentialCipher.from_environment(),
        ),
        connection_tester=test_data_source_connection,
        model_config_client=AIModelConfigClient(
            base_url=resolved.llm.base_url,
            timeout=resolved.llm.timeout_seconds,
            caller_service=resolved.api.service_name,
            audience=resolved.llm.audience,
        ),
        model_client=AIModelClient(
            caller_service=resolved.api.service_name,
            llm_config=resolved.llm,
        ),
    )
    app.state.runtime_service = DataQueryRuntimeService(
        uow_factory=uow_factory
    )
    app.middleware("http")(
        create_scoped_internal_auth_middleware(
            audience=resolved.api.service_name,
            allowed_callers={
                "kbot-main-api": frozenset({"data_query.manage", "data_query.delegate"}),
                "kbot-agent-runtime-api": frozenset({"data_query.delegate"}),
                "kbot-agent-runtime-worker": frozenset({"data_query.delegate"}),
                "kbot-data-query-worker": frozenset({"data_query.worker"}),
                "kbot-model-embedding": frozenset({"model.references"}),
                "kbot-model-llm": frozenset({"model.references"}),
                "kbot-model-visual": frozenset({"model.references"}),
                "kbot-model-vlm": frozenset({"model.references"}),
            },
        )
    )
    app.include_router(management_router)
    app.include_router(runtime_router)
    app.include_router(model_reference_router)

    @app.exception_handler(DataQueryManagementError)
    @app.exception_handler(SemanticModelPublicationError)
    @app.exception_handler(SchemaMetadataError)
    @app.exception_handler(SemanticModelValidationError)
    @app.exception_handler(DataQueryRunError)
    @app.exception_handler(DataSourceConnectionError)
    async def management_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        if isinstance(exc, DataSourceConnectionError):
            return JSONResponse(
                status_code=422,
                content={"code": exc.code, "detail": exc.public_message},
            )
        if str(exc) == "SEMANTIC_MODEL_AI_METADATA_NOT_APPROVED":
            return JSONResponse(
                status_code=422,
                content={
                    "code": str(exc),
                    "detail": "使用 AI 增强前必须确认允许所选模型处理数据库结构元数据。",
                },
            )
        return JSONResponse(status_code=409, content={"code": str(exc)})

    return app
