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
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from loguru import logger

from core.config.settings import get_visual_config, get_app_config
from core.logger import LogConfig, LogManager
from core.middleware.log_middleware import log_requests
from microservices.visual.visual_service import VisualService
from microservices.visual.schema import VisualEmbeddingRequest, VisualEmbeddingResponse
from microservices.common.security import create_internal_auth_middleware

# 加载环境变量
load_dotenv()

# 从 VisualConfig 读取视觉嵌入参数
config = get_visual_config()
SERVICE_NAME = config.service_name
SERVICE_VERSION = config.service_version
SERVICE_HOST = config.service_host
SERVICE_PORT = config.service_port

# 日志参数
app_config = get_app_config()
LOG_DIR = app_config.log.dir
LOG_LEVEL = app_config.log.level
LOG_ROTATION = app_config.log.rotation
LOG_RETENTION = app_config.log.retention
DEBUG_MODE = app_config.debug

# 全局服务实例
visual_service = VisualService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理服务生命周期"""
    app.state.service_name = SERVICE_NAME

    log_conf = LogConfig(
        service_name=SERVICE_NAME,
        log_dir=LOG_DIR,
        level=LOG_LEVEL,
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION,
    )
    LogManager(log_conf).setup()

    start_time = time.time()
    logger.info(f"Starting [{SERVICE_NAME}] | PID: {os.getpid()} | {datetime.now()}")

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
    await visual_service.shutdown()


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
app.middleware("http")(create_internal_auth_middleware())
app.middleware("http")(log_requests)


@app.get("/health", tags=["System"], summary="健康检查")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/v1/embed", response_model=VisualEmbeddingResponse, tags=["AI Service"])
async def embed_image(req: VisualEmbeddingRequest):
    """图片 → 视觉 embedding"""
    try:
        emb = await visual_service.embed(req.model_name, req.image_base64)
        return VisualEmbeddingResponse(
            embedding=emb,
            dimension=len(emb),
            model_name=req.model_name or "default",
        )
    except Exception as e:
        logger.error(f"[VisualService] embed failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info(f"Service starting → {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(
        app,
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        log_config=None,
        loop="asyncio",
    )
