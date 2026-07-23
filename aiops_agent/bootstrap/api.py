"""AIOps Internal API Bootstrap。"""

from contextlib import asynccontextmanager

from loguru import logger

from aiops_agent.bootstrap.common import (
    AIOpsProcessRuntime,
    configure_process_logging,
    create_process_app,
)
from aiops_agent.config import AIOpsSettings, get_aiops_settings
from platform_core.database.oracle import create_database_runtime
from platform_core.security import (
    create_auth_context_codec,
    create_scoped_internal_auth_middleware,
    create_service_identity_codec,
)


def create_aiops_api(
    settings: AIOpsSettings | None = None,
):
    """创建只含系统探针的步骤 0 Internal API。"""
    resolved = settings or get_aiops_settings()
    config = resolved.api

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
        app.state.auth_context_codec = create_auth_context_codec()
        app.state.service_identity_codec = create_service_identity_codec()
        await runtime.start()
        logger.info("正在启动 AIOps API 步骤 0 骨架")
        try:
            yield
        finally:
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
                    }
                ),
            },
        )
    )
    return app
