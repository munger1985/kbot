"""知识检索应用内部 API 进程。"""

import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import HTTPException
from fastapi_offline import FastAPIOffline
from sqlalchemy import text

from knowledge_retrieval_app.composition.knowledge_business import install_business_knowledge
from knowledge_retrieval_app.api.knowledge_library import create_library_router
from knowledge_retrieval_app.api import agent_router, research_router
from knowledge_retrieval_app.application import (
    KnowledgeRetrievalAgentService, ResearchRunService,
)
from knowledge_retrieval_app.config import get_knowledge_retrieval_app_settings
from knowledge_retrieval_app.persistence import create_knowledge_retrieval_app_uow
from platform_clients import AIModelConfigClient
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager
from platform_core.middleware.log_middleware import log_requests
from platform_core.platform.port_check import check_port_available
from platform_core.security import (
    create_auth_context_codec,
    create_scoped_internal_auth_middleware,
    create_service_identity_codec,
)


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
    uow_factory = create_knowledge_retrieval_app_uow(database_runtime.session_factory)
    catalog_client = AIModelConfigClient(
        base_url=settings.llm.base_url, timeout=settings.llm.timeout_seconds,
        caller_service=config.service_name, audience=settings.llm.audience,
    )
    install_business_knowledge(app, settings=settings, uow_factory=uow_factory, register_routes=False)
    app.state.agent_service = KnowledgeRetrievalAgentService(
        uow_factory=uow_factory, catalog_client=catalog_client,
    )
    app.state.research_service = ResearchRunService(
        uow_factory=uow_factory, catalog_client=catalog_client,
    )
    app.state.auth_context_codec = create_auth_context_codec()
    app.state.service_identity_codec = create_service_identity_codec()
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
            "kbot-main-api": frozenset({"knowledge_retrieval.manage", "knowledge.scope.create",
                "knowledge.retrieve", "knowledge.content.read", "knowledge.publish", "knowledge.withdraw"}),
            "kbot-agent-runtime-worker": frozenset({"knowledge.scope.create", "knowledge.retrieve", "knowledge.content.read"}),
            "kbot-knowledge-core-api": frozenset({"knowledge.source.read"}),
            "kbot-agent-runtime-api": frozenset({"knowledge_retrieval.manage", "knowledge.scope.create",
                "knowledge.retrieve", "knowledge.content.read"}),
            "kbot-model-embedding": frozenset({"knowledge_retrieval.manage"}),
            "kbot-model-llm": frozenset({"knowledge_retrieval.manage"}),
            "kbot-model-visual": frozenset({"knowledge_retrieval.manage"}),
            "kbot-model-vlm": frozenset({"knowledge_retrieval.manage"}),
            "kbot-model-ocr": frozenset({"knowledge_retrieval.manage"}),
        },
    )
)
app.middleware("http")(log_requests)
app.include_router(agent_router)
app.include_router(create_library_router())
# 路由在启动前登记，离线 OpenAPI 与实际运行暴露同一组端点。
from knowledge_retrieval_app.api.knowledge_retrieval import create_knowledge_retrieval_router, install_knowledge_errors
from knowledge_retrieval_app.api.knowledge_retrieval_auth import knowledge_access_dependency
from knowledge_retrieval_app.api.knowledge_source import create_knowledge_source_router
from knowledge_retrieval_app.api.knowledge_source_auth import source_scope
from platform_core.security.knowledge_delegation import create_knowledge_delegation_codec

app.include_router(create_knowledge_retrieval_router(access_dependency=knowledge_access_dependency(
    audience=config.service_name, delegation_codec=create_knowledge_delegation_codec(),
    allowed_callers=frozenset({"kbot-main-api", "kbot-agent-runtime-api", "kbot-agent-runtime-worker"}))))
app.include_router(create_knowledge_source_router(scope_dependency=source_scope))
install_knowledge_errors(app)
app.include_router(research_router)


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
