"""知识检索应用内部 API 进程。"""

import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import HTTPException
from fastapi_offline import FastAPIOffline
from sqlalchemy import text

from knowledge_retrieval_app.api import agent_router
from knowledge_retrieval_app.application import KnowledgeRetrievalAgentService
from knowledge_retrieval_app.config import get_knowledge_retrieval_app_settings
from knowledge_retrieval_app.persistence import create_knowledge_retrieval_app_uow
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager
from platform_core.middleware.log_middleware import log_requests
from platform_core.platform.port_check import check_port_available
from platform_core.security import create_scoped_internal_auth_middleware


settings = get_knowledge_retrieval_app_settings()
config = settings.api


@asynccontextmanager
async def lifespan(app: FastAPIOffline):
    LogManager(
        LogConfig(
            service="knowledge_retrieval_app",
            process="api",
            log_dir=settings.log.dir,
            level=settings.log.level,
            rotation=settings.log.rotation,
            retention=settings.log.retention,
        )
    ).setup()
    database_runtime = create_database_runtime(settings)
    app.state.db_runtime = database_runtime
    app.state.agent_service = KnowledgeRetrievalAgentService(
        uow_factory=create_knowledge_retrieval_app_uow(
            database_runtime.session_factory
        )
    )
    try:
        yield
    finally:
        await database_runtime.close()


app = FastAPIOffline(
    title="KBot Knowledge Retrieval App Internal API",
    version=config.service_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.platform.debug else None,
)
app.middleware("http")(
    create_scoped_internal_auth_middleware(
        audience=config.service_name,
        allowed_callers={
            "kbot-main-api": frozenset({"knowledge_retrieval.manage"}),
            "kbot-agent-runtime-api": frozenset(
                {"knowledge_retrieval.manage"}
            ),
        },
    )
)
app.middleware("http")(log_requests)
app.include_router(agent_router)


@app.get("/healthz")
async def live():
    return {"status": "live", "service": config.service_name}


@app.get("/readyz")
async def ready():
    try:
        async with app.state.db_runtime.session_factory() as session:
            await session.execute(text("SELECT 1 FROM DUAL"))
    except Exception as exc:
        raise HTTPException(503, {"code": "DATABASE_NOT_READY"}) from exc
    return {"status": "ready", "service": config.service_name}


if __name__ == "__main__":
    if not check_port_available(
        config.service_host, config.service_port, config.service_name
    ):
        sys.exit(1)
    uvicorn.run(
        app,
        host=config.service_host,
        port=config.service_port,
        log_config=None,
    )
