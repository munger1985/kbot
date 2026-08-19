"""AIOps Internal API Bootstrap。"""

import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger

from aiops_agent.adapters.agent_catalog import AIOpsAgentValidator
from aiops_agent.adapters.image_evidence import ImageEvidenceModelClient
from aiops_agent.actions import (
    ActionRegistry,
    create_mutation_grant_codec,
)
from aiops_agent.adapters.secret_store import ConfiguredSecretStore
from aiops_agent.application.managed_credentials import (
    AIOpsManagedCredentialService,
)
from aiops_agent.adapters.monitoring import MonitorProviderRegistry
from aiops_agent.adapters.db_executor_client import DatabaseExecutorClient
from aiops_agent.adapters.model_serving import AIOpsStructuredModelClient
from aiops_agent.adapters.monitoring.payload_store import (
    LocalMonitorPayloadStore,
)
from aiops_agent.api.management import router as management_router
from aiops_agent.api.agents import router as agent_router
from aiops_agent.api.conversations import router as conversation_router
from aiops_agent.api.report_templates import router as report_template_router
from aiops_agent.application.conversations import AIOpsConversationService
from aiops_agent.application.report_templates import InspectionReportTemplateService
from aiops_agent.api.runtime import router as runtime_router
from aiops_agent.api.intake import router as intake_router
from aiops_agent.api.changes import router as changes_router
from aiops_agent.api.executions import (
    event_router as execution_events_router,
    router as executions_router,
)
from aiops_agent.application.changes import AIOpsChangeService
from aiops_agent.application.agents import AIOpsAgentService
from aiops_agent.application.monitoring import MonitorWebhookIntakeService
from aiops_agent.application.configuration import AIOpsConfigurationService
from aiops_agent.application.runtime import AIOpsRuntimeService
from aiops_agent.application.configuration.common import SignedCursorCodec
from aiops_agent.application.configuration.schedule import (
    InspectionTemplateRegistry,
)
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.bootstrap.common import (
    AIOpsProcessRuntime,
    configure_process_logging,
    create_process_app,
)
from aiops_agent.config import AIOpsSettings, get_aiops_settings
from aiops_agent.persistence import create_aiops_uow_factory
from aiops_agent.orchestration import create_kernel_blueprint_registry
from aiops_agent.orchestration.diagnosis import DiagnosisPromptRegistry
from aiops_agent.workers import create_runtime_handler_registry
from aiops_agent.adapters.monitoring.catalog import load_metric_catalog
from aiops_agent.diagnostics import (
    create_diagnostic_grant_codec,
    create_diagnostic_registry,
)
from platform_core.database.oracle import create_database_runtime
from platform_core.managed_credentials import ManagedCredentialCipher
from platform_clients.model import AIModelConfigClient
from platform_clients.knowledge_core import KnowledgeCoreClient
from platform_core.security import (
    create_auth_context_codec,
    create_scoped_internal_auth_middleware,
    create_service_identity_codec,
)


def create_aiops_api(
    settings: AIOpsSettings | None = None,
):
    """创建配置、运行内核与监控接入 Internal API。"""
    resolved = settings or get_aiops_settings()
    config = resolved.api

    @asynccontextmanager
    async def lifespan(app):
        configure_process_logging(
            resolved,
            process="api",
        )
        database_runtime = create_database_runtime(resolved)
        runtime = AIOpsProcessRuntime(
            settings=resolved,
            service_name=config.service_name,
            database_runtime=database_runtime,
            uow_factory=create_aiops_uow_factory(
                database_runtime.session_factory
            ),
        )
        app.state.runtime = runtime
        app.state.agent_service = AIOpsAgentService(
            uow_factory=runtime.uow_factory
        )
        app.state.conversation_service = AIOpsConversationService(
            uow_factory=runtime.uow_factory,
            image_model_client=ImageEvidenceModelClient(
                caller_service=config.service_name,
                ocr_config=resolved.clients.model_ocr,
                vlm_config=resolved.clients.model_vlm,
            ),
        )
        app.state.report_template_service = InspectionReportTemplateService(
            uow_factory=runtime.uow_factory
        )
        app.state.ready_check = runtime.check_aiops_schema
        app.state.auth_context_codec = create_auth_context_codec()
        app.state.service_identity_codec = create_service_identity_codec()
        # 先完成不可恢复的配置校验，避免失败时遗留 HTTP 会话。
        credential_cipher = ManagedCredentialCipher.from_environment()
        managed_credential_service = AIOpsManagedCredentialService(
            uow_factory=runtime.uow_factory,
            cipher=credential_cipher,
        )
        agent_catalog = AIOpsAgentValidator(
            app.state.agent_service,
            model_client=AIModelConfigClient(
                base_url=resolved.clients.model_serving.base_url,
                timeout=resolved.clients.model_serving.timeout_seconds,
                caller_service=config.service_name,
                audience=resolved.clients.model_serving.audience,
            ),
        )
        cursor_secret = os.getenv(resolved.management.cursor_secret_env)
        if not cursor_secret:
            if resolved.is_production():
                raise RuntimeError(
                    "生产环境缺少 AIOps Cursor 签名密钥"
                )
            cursor_secret = hashlib.sha256(
                f"{config.service_name}:development:cursor".encode("utf-8")
            ).hexdigest()
            logger.warning("开发环境使用临时派生的 AIOps Cursor 签名密钥")
        secret_store = ConfiguredSecretStore(
            managed_credentials=managed_credential_service,
        )
        cursor_codec = SignedCursorCodec(
            secret=cursor_secret,
            ttl_seconds=resolved.management.cursor_ttl_seconds,
        )
        app.state.configuration_service = AIOpsConfigurationService(
            uow_factory=runtime.uow_factory,
            cursor_codec=cursor_codec,
            secret_store=secret_store,
            agent_catalog=agent_catalog,
            template_registry=InspectionTemplateRegistry(
                resolved.management.inspection_templates
            ),
            management=resolved.management,
            max_inspection_targets=(
                resolved.limits.max_targets_per_inspection_fire
            ),
            credential_cipher=credential_cipher,
            managed_credential_service=managed_credential_service,
        )
        metric_catalog = load_metric_catalog(
            Path(resolved.monitoring.catalog_path)
            if resolved.monitoring.catalog_path
            else None
        )
        diagnostic_registry = create_diagnostic_registry(resolved)
        action_registry = ActionRegistry.load()
        app.state.change_service = AIOpsChangeService(
            uow_factory=runtime.uow_factory,
            action_registry=action_registry,
            approval_enabled=(
                resolved.management.agent_execution_enabled
            ),
            mutation_enabled=resolved.executor.mutation_enabled,
            mutation_grant_codec=(
                create_mutation_grant_codec(resolved)
                if resolved.executor.mutation_enabled
                else None
            ),
            mutation_grant_issuer=(
                resolved.executor.mutation_grant_issuer
            ),
            mutation_grant_audience=resolved.executor.service_name,
            mutation_grant_ttl_seconds=(
                resolved.executor.mutation_grant_ttl_seconds
            ),
            mutation_statement_timeout_seconds=(
                resolved.executor.statement_timeout_seconds
            ),
        )
        diagnosis_prompts = DiagnosisPromptRegistry.load(
            Path(resolved.diagnosis.prompt_catalog_path)
            if resolved.diagnosis.prompt_catalog_path
            else None
        )
        # 资源目录和签名配置完成校验后再创建网络会话。
        client_session = aiohttp.ClientSession()
        provider_registry = MonitorProviderRegistry(
            session=client_session,
            request_timeout_seconds=(
                resolved.monitoring.provider_timeout_seconds
            ),
            webhook_replay_seconds=(
                resolved.monitoring.webhook_replay_seconds
            ),
        )
        diagnosis_model_client = AIOpsStructuredModelClient(
            base_url=resolved.clients.model_serving.base_url,
            audience=resolved.clients.model_serving.audience,
            caller_service=config.service_name,
            timeout_seconds=resolved.clients.model_serving.timeout_seconds,
            session=client_session,
        )
        knowledge_core_client = KnowledgeCoreClient(
            base_url=resolved.clients.knowledge_core.base_url,
            caller_service=config.service_name,
            audience=resolved.clients.knowledge_core.audience,
            timeout_seconds=resolved.clients.knowledge_core.timeout_seconds,
            session=client_session,
        )
        diagnostic_grant_codec = create_diagnostic_grant_codec(resolved)
        app.state.managed_credential_service = managed_credential_service
        app.state.diagnostic_grant_codec = diagnostic_grant_codec
        app.state.mutation_grant_codec = create_mutation_grant_codec(resolved)
        db_executor_client = DatabaseExecutorClient(
            base_url=resolved.clients.db_executor.base_url,
            audience=resolved.clients.db_executor.audience,
            caller_service=resolved.worker.service_name,
            timeout_seconds=resolved.clients.db_executor.timeout_seconds,
            session=client_session,
        )
        handler_registry = create_runtime_handler_registry(
            monitor_provider_registry=provider_registry,
            secret_store=secret_store,
            db_executor_client=db_executor_client,
            diagnostic_grant_codec=diagnostic_grant_codec,
            diagnostic_grant_issuer=resolved.executor.grant_issuer,
            diagnostic_grant_audience=resolved.executor.service_name,
            diagnostic_grant_ttl_seconds=(
                resolved.executor.grant_ttl_seconds
            ),
            diagnosis_model_client=diagnosis_model_client,
            diagnosis_prompt_registry=diagnosis_prompts,
            diagnostic_registry=diagnostic_registry,
            knowledge_core_client=knowledge_core_client,
            diagnosis_caller_service=config.service_name,
            action_registry=action_registry,
            action_execution_enabled=(
                resolved.management.agent_execution_enabled
            ),
        )
        app.state.monitor_provider_registry = provider_registry
        app.state.monitor_secret_store = secret_store
        app.state.metric_catalog = metric_catalog
        app.state.aiops_runtime_service = AIOpsRuntimeService(
            uow_factory=runtime.uow_factory,
            blueprint_registry=create_kernel_blueprint_registry(),
            handler_registry=handler_registry,
            max_tasks_per_run=resolved.limits.max_tasks_per_run,
            default_run_timeout_seconds=(
                resolved.limits.run_timeout_seconds
            ),
            metric_catalog=metric_catalog,
            default_observation_window_seconds=(
                resolved.monitoring.default_window_seconds
            ),
            max_monitor_response_bytes=(
                resolved.monitoring.max_response_bytes
            ),
            diagnostic_registry=diagnostic_registry,
            diagnosis_config=resolved.diagnosis,
            diagnosis_prompt_registry=diagnosis_prompts,
            agent_catalog=agent_catalog,
            cursor_codec=cursor_codec,
        )
        app.state.monitor_intake_service = MonitorWebhookIntakeService(
            uow_factory=runtime.uow_factory,
            provider_registry=provider_registry,
            secret_store=secret_store,
            system_agent_id=resolved.runtime.system_aiops_agent_id,
            max_webhook_bytes=resolved.monitoring.max_webhook_bytes,
            payload_store=LocalMonitorPayloadStore(
                Path(resolved.monitoring.payload_store_root)
            ),
        )
        await runtime.start()
        logger.info("正在启动 AIOps 配置管理 API")
        try:
            yield
        finally:
            await client_session.close()
            await runtime.close()
            logger.info("AIOps API 资源已释放")

    app = create_process_app(
        title="KBot AIOps Internal API",
        description="AIOps 管理、委派、监控接入和 Executor 回调边界。",
        service_name=config.service_name,
        service_version=config.service_version,
        debug=resolved.platform.debug,
        lifespan=lifespan,
    )
    app.middleware("http")(
        create_scoped_internal_auth_middleware(
            audience=config.service_name,
            allowed_callers={
                "kbot-main-api": frozenset(
                    {
                        "aiops.manage",
                        "aiops.run",
                        "aiops.hitl",
                        "aiops.approve",
                        "aiops.monitor.intake",
                    }
                ),
                "kbot-agent-runtime-api": frozenset({"aiops.delegate"}),
                "kbot-aiops-worker": frozenset(
                    {
                        "aiops.task",
                        "aiops.artifact",
                        "aiops.outbox",
                        "aiops.execution.request",
                    }
                ),
                "kbot-aiops-scheduler": frozenset({"aiops.schedule"}),
                "kbot-aiops-db-executor": frozenset(
                    {
                        "aiops.execution.claim",
                        "aiops.execution.callback",
                        "aiops.credentials.issue",
                    }
                ),
            },
        )
    )
    app.include_router(management_router)
    app.include_router(agent_router)
    app.include_router(conversation_router)
    app.include_router(report_template_router)
    app.include_router(runtime_router)
    app.include_router(intake_router)
    app.include_router(changes_router)
    app.include_router(executions_router)
    app.include_router(execution_events_router)

    @app.exception_handler(AIOpsApplicationError)
    async def application_error_handler(
        request: Request,
        exc: AIOpsApplicationError,
    ) -> JSONResponse:
        context = getattr(request.state, "auth_context", None)
        request_id = (
            getattr(context, "request_id", None)
            or request.headers.get("X-Request-ID")
            or "unknown"
        )
        return JSONResponse(
            status_code=exc.status_code,
            media_type="application/problem+json",
            content={
                "type": f"urn:kbot:error:{exc.code.lower()}",
                "title": "AIOps 请求失败",
                "status": exc.status_code,
                "code": exc.code,
                "detail": exc.message,
                "request_id": request_id,
                "retryable": exc.retryable,
            },
        )
    return app
