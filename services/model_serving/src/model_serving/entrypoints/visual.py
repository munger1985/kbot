"""视觉嵌入微服务应用程序。

提供 ColQwen2 模型的图片→embedding HTTP API。
"""

import os
import sys
import time
from datetime import datetime
from typing import Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from loguru import logger

from model_serving.config import get_model_serving_settings
from platform_core.contracts import INTERNAL_API_V1
from platform_core.dictionary import ModelCategory
from platform_core.logger import LogConfig, LogManager
from platform_core.middleware.log_middleware import log_requests
from platform_core.database.oracle import create_database_runtime
from model_serving.visual.visual_service import VisualService
from model_serving.visual.schema import VisualEmbeddingRequest, VisualEmbeddingResponse
from platform_core.security import create_internal_auth_middleware
from platform_core.platform.port_check import check_port_available
from model_serving.common.management_router import create_model_management_router
from model_serving.common.bootstrap import create_model_registry

# 从 VisualConfig 读取视觉嵌入参数
settings = get_model_serving_settings()
config = settings.visual
SERVICE_NAME = config.service_name
SERVICE_VERSION = config.service_version
SERVICE_HOST = config.service_host
SERVICE_PORT = config.service_port

# 日志参数
LOG_DIR = settings.log.dir
LOG_LEVEL = settings.log.level
LOG_ROTATION = settings.log.rotation
LOG_RETENTION = settings.log.retention
DEBUG_MODE = settings.platform.debug

# 全局服务实例
visual_service = VisualService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理服务生命周期"""
    app.state.service_name = SERVICE_NAME
    db_runtime = create_database_runtime()
    app.state.db_runtime = db_runtime
    visual_service.bind_session_factory(db_runtime.session_factory)
    app.state.model_registry = create_model_registry(
        session_factory=db_runtime.session_factory,
        runtime_service=visual_service,
        service_name=SERVICE_NAME,
        settings=settings,
    )

    log_conf = LogConfig(
        service="model_serving",
        process="visual",
        log_dir=LOG_DIR,
        level=LOG_LEVEL,
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION,
    )
    LogManager(log_conf).setup()

    start_time = time.time()
    logger.info(f"正在启动 [{SERVICE_NAME}] | 进程号：{os.getpid()} | {datetime.now()}")

    try:
        await visual_service.initialize()
        logger.success(
            f"[{SERVICE_NAME}] initialized in {time.time() - start_time:.1f}s"
        )
    except Exception as e:
        logger.error(f"[{SERVICE_NAME}] init failed: {e}")
        if not DEBUG_MODE:
            sys.exit(1)

    yield

    logger.info(f"[{SERVICE_NAME}] shutting down...")
    try:
        await visual_service.shutdown()
    finally:
        await db_runtime.close()


app = FastAPIOffline(
    title=f"{SERVICE_NAME} API",
    description="ColQwen2 视觉嵌入服务 — 图片→向量",
    version=SERVICE_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if DEBUG_MODE else None,
    redoc_url="/redoc" if DEBUG_MODE else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(
    create_internal_auth_middleware(audience=SERVICE_NAME)
)
app.middleware("http")(log_requests)
app.include_router(create_model_management_router(category=ModelCategory.IMG_EMBEDDING.value))


@app.get("/health", tags=["System"], summary="健康检查")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "timestamp": datetime.now().isoformat(),
    }


@app.post(f"{INTERNAL_API_V1}/embed", response_model=VisualEmbeddingResponse, tags=["AI Service"])
async def embed_image(req: VisualEmbeddingRequest):
    """图片 → 视觉 embedding"""
    try:
        emb = await visual_service.embed(req.served_model_name, req.image_base64)
        return VisualEmbeddingResponse(
            embedding=emb,
            dimension=len(emb),
            served_model_name=req.served_model_name,
        )
    except Exception as e:
        logger.error(f"[VisualService] embed failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 先检查端口可用性，避免 EADDRINUSE 错误被 stderr 吞掉
    if not check_port_available(SERVICE_HOST, SERVICE_PORT, SERVICE_NAME):
        sys.exit(1)

    logger.info(f"服务开始监听 → {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(
        app,
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        log_config=None,
        loop="asyncio",
    )
