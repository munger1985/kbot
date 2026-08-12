"""KM Asset App 内部 API 进程。"""

import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi_offline import FastAPIOffline
from sqlalchemy import text

from km_asset_app.api import agent_router, asset_router
from km_asset_app.application import KmAgentService, KmAssetService, KmCredentialService
from km_asset_app.config import get_km_asset_settings
from km_asset_app.persistence import create_km_asset_uow
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager
from platform_core.middleware.log_middleware import log_requests
from platform_core.platform.port_check import check_port_available
from platform_core.security import create_auth_context_codec, create_scoped_internal_auth_middleware, create_service_identity_codec
from platform_core.managed_credentials import ManagedCredentialCipher
from platform_clients import DataQueryClient, DataQueryClientError


settings = get_km_asset_settings()
config = settings.api


@asynccontextmanager
async def lifespan(app: FastAPIOffline):
    LogManager(LogConfig(service="km_asset_app", process="api", log_dir=settings.log.dir, level=settings.log.level, rotation=settings.log.rotation, retention=settings.log.retention)).setup()
    runtime = create_database_runtime(settings)
    app.state.db_runtime = runtime
    data_query_client = DataQueryClient(base_url=settings.data_query.base_url, caller_service=config.service_name, audience=settings.data_query.audience, timeout_seconds=settings.data_query.timeout_seconds)
    app.state.data_query_client = data_query_client
    app.state.km_asset_service = KmAssetService(uow_factory=create_km_asset_uow(runtime.session_factory), credential_service=KmCredentialService(cipher=ManagedCredentialCipher.from_environment()), data_query_client=data_query_client)
    app.state.km_agent_service = KmAgentService(
        uow_factory=create_km_asset_uow(runtime.session_factory),
        data_query_client=data_query_client,
    )
    app.state.auth_context_codec = create_auth_context_codec()
    app.state.service_identity_codec = create_service_identity_codec()
    try:
        yield
    finally:
        await data_query_client.close()
        await runtime.close()


app = FastAPIOffline(title="KBot KM Asset App Internal API", version=config.service_version, lifespan=lifespan, docs_url="/docs" if settings.platform.debug else None)
app.middleware("http")(create_scoped_internal_auth_middleware(audience=config.service_name, allowed_callers={"kbot-main-api": frozenset({"km_asset.manage"}), "kbot-km-asset-app-worker": frozenset({"km_asset.worker"}), "kbot-data-query-api": frozenset({"km_asset.reconcile"}), "kbot-agent-runtime-api": frozenset({"km_asset.manage"})}))
app.middleware("http")(log_requests)
app.include_router(asset_router)
app.include_router(agent_router)


@app.exception_handler(DataQueryClientError)
async def data_query_error_handler(_request: Request, exc: DataQueryClientError):
    status_code = 503 if exc.status_code >= 500 else exc.status_code
    return JSONResponse(status_code=status_code, content={"detail": {"code": exc.code, "message": str(exc)}})


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
