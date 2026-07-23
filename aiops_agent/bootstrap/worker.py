"""AIOps Worker 探针 Bootstrap。"""

from contextlib import asynccontextmanager

from loguru import logger

from aiops_agent.bootstrap.common import (
    AIOpsProcessRuntime,
    configure_process_logging,
    create_process_app,
)
from aiops_agent.config import AIOpsSettings, get_aiops_settings
from platform_core.database.oracle import create_database_runtime


def create_aiops_worker_probe(
    settings: AIOpsSettings | None = None,
):
    """创建尚不领取 Task 的 Worker 步骤 0 探针。"""
    resolved = settings or get_aiops_settings()
    config = resolved.worker

    @asynccontextmanager
    async def lifespan(app):
        configure_process_logging(
            resolved,
            service_name=config.service_name,
        )
        runtime = AIOpsProcessRuntime(
            settings=resolved,
            service_name=config.service_name,
            database_runtime=create_database_runtime(resolved),
        )
        app.state.runtime = runtime
        app.state.ready_check = runtime.check_aiops_schema
        await runtime.start()
        logger.info("正在启动 AIOps Worker 步骤 0 探针")
        try:
            yield
        finally:
            await runtime.close()
            logger.info("AIOps Worker 探针资源已释放")

    return create_process_app(
        title="KBot AIOps Worker Probe",
        description="AIOps Worker 的内部存活与就绪探针。",
        service_name=config.service_name,
        service_version=config.service_version,
        debug=False,
        lifespan=lifespan,
    )
