"""AIOps Task Worker、Reconciler 与 Outbox Bootstrap。"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
from loguru import logger

from aiops_agent.adapters.diagnostic_sources import DiagnosticSourceAdapterRegistry
from aiops_agent.actions import ActionRegistry
from aiops_agent.adapters.db_executor_client import DatabaseExecutorClient
from aiops_agent.adapters.model_serving import AIOpsStructuredModelClient
from aiops_agent.adapters.agent_catalog import AIOpsAgentValidator
from aiops_agent.adapters.secret_store import ConfiguredSecretStore
from aiops_agent.bootstrap.common import (
    AIOpsProcessRuntime,
    configure_process_logging,
    create_process_app,
)
from aiops_agent.config import AIOpsSettings, get_aiops_settings
from aiops_agent.persistence import create_aiops_uow_factory
from aiops_agent.application.runtime import AIOpsRuntimeService
from aiops_agent.application.turn_queue import TurnQueueService
from aiops_agent.application.turn_planner import TurnPlannerService
from aiops_agent.application.turn_planning import TurnPlanningService
from aiops_agent.application.agents import AIOpsAgentService
from aiops_agent.application.managed_credentials import (
    AIOpsManagedCredentialService,
)
from aiops_agent.application.diagnostic_sources import (
    DiagnosticSourceConnectivityCheckService,
)
from aiops_agent.application.targets import TargetConnectivityCheckService
from aiops_agent.adapters.diagnostic_sources.catalog import load_metric_catalog
from aiops_agent.diagnostics import (
    create_diagnostic_grant_codec,
    create_diagnostic_registry,
)
from aiops_agent.skills import (
    DbaIntentRouter,
    DbaSkillPlanner,
    DbaSkillRegistry,
    SkillExecutionSnapshotBuilder,
    SkillPlanCompiler,
)
from aiops_agent.orchestration import create_kernel_blueprint_registry
from aiops_agent.orchestration.diagnosis import DiagnosisPromptRegistry
from aiops_agent.workers import (
    AIOpsOutboxDispatcher,
    AIOpsDomainOutboxSink,
    AIOpsReconciler,
    AIOpsTaskWorker,
    LoggingOutboxSink,
    create_runtime_handler_registry,
)
from platform_core.database.oracle import create_database_runtime
from platform_core.managed_credentials import ManagedCredentialCipher
from platform_clients.knowledge_core import KnowledgeCoreClient
from platform_clients.model import AIModelConfigClient


def create_aiops_worker_probe(
    settings: AIOpsSettings | None = None,
):
    """创建可多副本运行的 Task、恢复、监控与 Outbox Worker。"""
    resolved = settings or get_aiops_settings()
    config = resolved.worker

    @asynccontextmanager
    async def lifespan(app):
        configure_process_logging(
            resolved,
            process="worker",
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
        app.state.ready_check = runtime.check_aiops_schema
        action_registry = ActionRegistry.load()
        agent_catalog = AIOpsAgentValidator(
            AIOpsAgentService(
                uow_factory=runtime.uow_factory,
                action_registry=action_registry,
            ),
            model_client=AIModelConfigClient(
                base_url=resolved.clients.model_serving.base_url,
                timeout=resolved.clients.model_serving.timeout_seconds,
                caller_service=config.service_name,
                audience=resolved.clients.model_serving.audience,
            ),
        )
        managed_credential_service = AIOpsManagedCredentialService(
            uow_factory=runtime.uow_factory,
            cipher=ManagedCredentialCipher.from_environment(),
        )
        secret_store = ConfiguredSecretStore(
            managed_credentials=managed_credential_service,
        )
        metric_catalog = load_metric_catalog(
            Path(resolved.monitoring.catalog_path)
            if resolved.monitoring.catalog_path
            else None
        )
        diagnostic_registry = create_diagnostic_registry(resolved)
        skill_registry = DbaSkillRegistry.load(
            allowed_tools=frozenset(
                (tool.definition.tool_id, tool.definition.version)
                for tool in diagnostic_registry.tools
            )
        )
        execution_snapshot_builder = SkillExecutionSnapshotBuilder(
            skill_registry=skill_registry,
            diagnostic_registry=diagnostic_registry,
        )
        execution_snapshot_builder.validate_catalog()
        diagnosis_prompts = DiagnosisPromptRegistry.load(
            Path(resolved.diagnosis.prompt_catalog_path)
            if resolved.diagnosis.prompt_catalog_path
            else None
        )
        # 本地资源校验失败时不应提前创建网络会话。
        client_session = aiohttp.ClientSession()
        diagnostic_source_registry = DiagnosticSourceAdapterRegistry(
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
        db_executor_client = DatabaseExecutorClient(
            base_url=resolved.clients.db_executor.base_url,
            audience=resolved.clients.db_executor.audience,
            caller_service=config.service_name,
            timeout_seconds=resolved.clients.db_executor.timeout_seconds,
            session=client_session,
        )
        handler_registry = create_runtime_handler_registry(
            diagnostic_source_registry=diagnostic_source_registry,
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
        runtime_service = AIOpsRuntimeService(
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
        )
        workers = [
            AIOpsTaskWorker(
                runtime_service=runtime_service,
                handler_registry=handler_registry,
                worker_id=f"{config.worker_id}-{index + 1}",
                lease_seconds=config.lease_seconds,
                heartbeat_seconds=config.heartbeat_seconds,
                poll_interval_seconds=config.claim_interval_seconds,
            )
            for index in range(config.concurrency)
        ]
        reconciler = AIOpsReconciler(
            runtime_service=runtime_service,
            interval_seconds=config.claim_interval_seconds,
        )
        dispatcher = AIOpsOutboxDispatcher(
            uow_factory=runtime.uow_factory,
            sink=AIOpsDomainOutboxSink(
                runtime_service=runtime_service,
                fallback=LoggingOutboxSink(),
                diagnostic_source_connectivity_service=DiagnosticSourceConnectivityCheckService(
                    uow_factory=runtime.uow_factory,
                    diagnostic_source_registry=diagnostic_source_registry,
                    secret_store=secret_store,
                ),
                target_connectivity_service=TargetConnectivityCheckService(
                    uow_factory=runtime.uow_factory,
                    managed_credentials=managed_credential_service,
                ),
                db_executor_client=db_executor_client,
                turn_queue_service=TurnQueueService(
                    uow_factory=runtime.uow_factory,
                ),
                turn_planner_service=TurnPlannerService(
                    uow_factory=runtime.uow_factory,
                ),
                turn_planning_service=TurnPlanningService(
                    uow_factory=runtime.uow_factory,
                    intent_router=DbaIntentRouter(diagnosis_model_client),
                    skill_planner=DbaSkillPlanner(skill_registry),
                    skill_compiler=SkillPlanCompiler(skill_registry),
                    execution_snapshot_builder=(
                        execution_snapshot_builder
                    ),
                    agent_catalog=agent_catalog,
                ),
            ),
            dispatcher_id=f"{config.worker_id}-outbox",
            lease_seconds=config.lease_seconds,
            interval_seconds=config.claim_interval_seconds,
        )
        components = [*workers, reconciler, dispatcher]
        background_tasks = [
            asyncio.create_task(component.run_forever())
            for component in components
        ]
        await runtime.start()
        logger.info(
            "正在启动 AIOps Worker：concurrency={}",
            config.concurrency,
        )
        try:
            yield
        finally:
            for component in components:
                component.stop()
            await asyncio.gather(
                *background_tasks, return_exceptions=True
            )
            await client_session.close()
            await runtime.close()
            logger.info("AIOps Worker 资源已释放")

    return create_process_app(
        title="KBot AIOps Worker",
        description="AIOps Task、恢复和 Outbox 后台进程。",
        service_name=config.service_name,
        service_version=config.service_version,
        debug=False,
        lifespan=lifespan,
    )
