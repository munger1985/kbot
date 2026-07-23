"""AIOps Task Worker、Reconciler 与 Outbox Bootstrap。"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
from loguru import logger

from aiops_agent.adapters.monitoring import MonitorProviderRegistry
from aiops_agent.adapters.secret_store import ConfiguredSecretStore
from aiops_agent.bootstrap.common import (
    AIOpsProcessRuntime,
    configure_process_logging,
    create_process_app,
)
from aiops_agent.config import AIOpsSettings, get_aiops_settings
from aiops_agent.persistence import create_aiops_uow_factory
from aiops_agent.application.runtime import AIOpsRuntimeService
from aiops_agent.application.monitoring import MonitorHealthCheckService
from aiops_agent.adapters.monitoring.catalog import load_metric_catalog
from aiops_agent.orchestration import create_kernel_blueprint_registry
from aiops_agent.workers import (
    AIOpsOutboxDispatcher,
    AIOpsDomainOutboxSink,
    AIOpsReconciler,
    AIOpsTaskWorker,
    LoggingOutboxSink,
    create_runtime_handler_registry,
)
from platform_core.database.oracle import create_database_runtime


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
            service_name=config.service_name,
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
        client_session = aiohttp.ClientSession()
        secret_store = ConfiguredSecretStore(
            provider=resolved.secret_store.provider,
            allowed_schemes=resolved.secret_store.allowed_schemes,
        )
        metric_catalog = load_metric_catalog(
            Path(resolved.monitoring.catalog_path)
            if resolved.monitoring.catalog_path
            else None
        )
        provider_registry = MonitorProviderRegistry(
            session=client_session,
            request_timeout_seconds=(
                resolved.monitoring.provider_timeout_seconds
            ),
            webhook_replay_seconds=(
                resolved.monitoring.webhook_replay_seconds
            ),
        )
        handler_registry = create_runtime_handler_registry(
            monitor_provider_registry=provider_registry,
            secret_store=secret_store,
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
                monitor_health_service=MonitorHealthCheckService(
                    uow_factory=runtime.uow_factory,
                    provider_registry=provider_registry,
                    secret_store=secret_store,
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
