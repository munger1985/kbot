"""AIOps DB Executor Bootstrap。"""

from contextlib import asynccontextmanager

from loguru import logger

from aiops_agent.bootstrap.common import (
    AIOpsProcessRuntime,
    configure_process_logging,
    create_process_app,
)
from aiops_agent.config import AIOpsSettings, get_aiops_settings
from platform_core.security import (
    create_auth_context_codec,
    create_scoped_internal_auth_middleware,
    create_service_identity_codec,
)


def create_aiops_executor(
    settings: AIOpsSettings | None = None,
):
    """创建不持有 KBot Schema 连接的步骤 0 Executor。"""
    resolved = settings or get_aiops_settings()
    config = resolved.executor

    @asynccontextmanager
    async def lifespan(app):
        configure_process_logging(
            resolved,
            service_name=config.service_name,
        )
        runtime = AIOpsProcessRuntime(
            settings=resolved,
            service_name=config.service_name,
            components={
                "template_registry": False,
                "secret_provider": False,
                "identity_verifier": False,
            },
        )
        app.state.runtime = runtime
        app.state.ready_check = runtime.check_executor_components
        app.state.auth_context_codec = create_auth_context_codec()
        app.state.service_identity_codec = create_service_identity_codec()
        await runtime.start()
        logger.info(
            "正在启动 AIOps DB Executor 步骤 0 骨架，Mutation={}",
            config.mutation_enabled,
        )
        try:
            yield
        finally:
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
    return app
