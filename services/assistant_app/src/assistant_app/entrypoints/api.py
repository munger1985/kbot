"""智能工作台内部 API 进程。"""

import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import HTTPException
from fastapi_offline import FastAPIOffline
from sqlalchemy import text

from assistant_app.adapters import AssistantLocalObjectStore
from assistant_app.api import (
    agent_router,
    binding_router,
    image_generation_router,
    media_router,
    research_router,
    run_router,
)
from assistant_app.application import (
    AssistantAgentService,
    AssistantRunQueryService,
    ImageGenerationService,
    MediaAssetService,
    ModelBindingService,
    ResearchRunService,
)
from assistant_app.config import get_assistant_app_settings
from assistant_app.persistence import create_assistant_app_uow
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


settings = get_assistant_app_settings()
config = settings.api


@asynccontextmanager
async def lifespan(app: FastAPIOffline):
    LogManager(LogConfig(
        service="assistant_app", process="api", log_dir=settings.log.dir,
        level=settings.log.level, rotation=settings.log.rotation,
        retention=settings.log.retention,
    )).setup()
    database_runtime = create_database_runtime(settings)
    uow_factory = create_assistant_app_uow(database_runtime.session_factory)
    catalog_client = AIModelConfigClient(
        base_url=settings.llm.base_url,
        timeout=settings.llm.timeout_seconds,
        caller_service=config.service_name,
        audience=settings.llm.audience,
    )
    object_store = AssistantLocalObjectStore(settings.storage.local_object_storage_path)
    app.state.db_runtime = database_runtime
    app.state.agent_service = AssistantAgentService(uow_factory=uow_factory)
    app.state.binding_service = ModelBindingService(
        uow_factory=uow_factory, catalog_client=catalog_client,
    )
    app.state.research_service = ResearchRunService(
        uow_factory=uow_factory, catalog_client=catalog_client,
    )
    app.state.image_generation_service = ImageGenerationService(
        uow_factory=uow_factory, catalog_client=catalog_client,
    )
    app.state.media_service = MediaAssetService(
        uow_factory=uow_factory, object_store=object_store,
    )
    app.state.run_query_service = AssistantRunQueryService(uow_factory=uow_factory)
    app.state.auth_context_codec = create_auth_context_codec()
    app.state.service_identity_codec = create_service_identity_codec()
    try:
        yield
    finally:
        await database_runtime.close()


app = FastAPIOffline(
    title="KBot Assistant App Internal API",
    version=config.service_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.platform.debug else None,
)
app.middleware("http")(
    create_scoped_internal_auth_middleware(
        audience=config.service_name,
        allowed_callers={"kbot-main-api": frozenset({"assistant.manage"})},
    )
)
app.middleware("http")(log_requests)
app.include_router(agent_router)
app.include_router(binding_router)
app.include_router(research_router)
app.include_router(image_generation_router)
app.include_router(media_router)
app.include_router(run_router)


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
    if not check_port_available(config.service_host, config.service_port, config.service_name):
        sys.exit(1)
    uvicorn.run(app, host=config.service_host, port=config.service_port, log_config=None)
