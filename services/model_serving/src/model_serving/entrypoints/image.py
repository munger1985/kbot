"""Model Serving 图像生成与编辑独立进程。"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi_offline import FastAPIOffline
from loguru import logger

from model_serving.common.bootstrap import create_model_registry
from model_serving.common.management_router import create_model_management_router
from model_serving.config import get_model_serving_settings
from model_serving.image import ImageModelAdapter, ImageService, create_image_router
from platform_core.database.oracle import create_database_runtime
from platform_core.dictionary import ModelCategory
from platform_core.logger import LogConfig, LogManager
from platform_core.platform.port_check import check_port_available
from platform_core.security import create_internal_auth_middleware


settings = get_model_serving_settings()
config = settings.image


@asynccontextmanager
async def lifespan(app: FastAPI):
    LogManager(LogConfig(
        service="model_serving",
        process="image",
        log_dir=settings.log.dir,
        level=settings.log.level,
        rotation=settings.log.rotation,
        retention=settings.log.retention,
    )).setup()
    runtime = create_database_runtime()
    registry = create_model_registry(
        session_factory=runtime.session_factory,
        runtime_service=None,
        service_name=config.service_name,
        settings=settings,
    )
    app.state.service_name = config.service_name
    app.state.model_registry = registry
    app.state.image_service = ImageService(
        registry=registry,
        adapter=ImageModelAdapter(timeout_seconds=config.timeout),
    )
    try:
        yield
    finally:
        await runtime.close()


app: FastAPI = FastAPIOffline(
    title="KBot Image Generation API",
    version=config.service_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.platform.debug else None,
    redoc_url="/redoc" if settings.platform.debug else None,
)
app.middleware("http")(
    create_internal_auth_middleware(audience=config.service_name)
)
app.include_router(
    create_model_management_router(category=ModelCategory.IMAGE_GENERATION.value)
)
app.include_router(create_image_router())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": config.service_name}


def main() -> None:
    """启动独立图像生成进程。"""
    if not check_port_available(
        config.service_host, config.service_port, config.service_name,
    ):
        raise SystemExit(1)
    logger.info(
        "正在启动图像生成服务 | 监听地址：{}:{}",
        config.service_host,
        config.service_port,
    )
    uvicorn.run(
        app,
        host=config.service_host,
        port=config.service_port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
