"""AIOps Scheduler 探针 Bootstrap。"""

import asyncio
from contextlib import asynccontextmanager

from loguru import logger

from aiops_agent.bootstrap.common import (
    AIOpsProcessRuntime,
    configure_process_logging,
    create_process_app,
)
from aiops_agent.config import AIOpsSettings, get_aiops_settings
from aiops_agent.persistence import create_aiops_uow_factory
from aiops_agent.scheduling import (
    AIOpsConnectivityScheduler,
    AIOpsInspectionScheduler,
)
from platform_core.database.oracle import create_database_runtime


def create_aiops_scheduler_probe(
    settings: AIOpsSettings | None = None,
):
    """创建可多副本领取 Plan、生成 Fire 并收敛结果的 Scheduler。"""
    resolved = settings or get_aiops_settings()
    config = resolved.scheduler

    @asynccontextmanager
    async def lifespan(app):
        configure_process_logging(
            resolved,
            process="scheduler",
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
        await runtime.start()
        scheduler = AIOpsInspectionScheduler(
            uow_factory=runtime.uow_factory,
            scheduler_id=config.scheduler_id,
            lease_seconds=config.lease_seconds,
            interval_seconds=config.scan_interval_seconds,
            misfire_grace_seconds=config.misfire_grace_seconds,
        )
        connectivity_scheduler = AIOpsConnectivityScheduler(
            uow_factory=runtime.uow_factory,
            scheduler_id=config.scheduler_id,
            interval_seconds=config.connectivity_check_interval_seconds,
            jitter_seconds=config.connectivity_check_jitter_seconds,
            scan_interval_seconds=config.scan_interval_seconds,
        )
        background_tasks = [
            asyncio.create_task(scheduler.run_forever()),
            asyncio.create_task(connectivity_scheduler.run_forever()),
        ]
        logger.info("正在启动 AIOps Inspection Scheduler")
        try:
            yield
        finally:
            scheduler.stop()
            connectivity_scheduler.stop()
            await asyncio.gather(
                *background_tasks, return_exceptions=True
            )
            await runtime.close()
            logger.info("AIOps Inspection Scheduler 资源已释放")

    return create_process_app(
        title="KBot AIOps Scheduler",
        description="AIOps 巡检 Plan、Fire 与结果收敛进程。",
        service_name=config.service_name,
        service_version=config.service_version,
        debug=False,
        lifespan=lifespan,
    )
