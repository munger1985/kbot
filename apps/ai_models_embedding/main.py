"""Embedding microservice application.

This module provides a FastAPI-based microservice that exposes HTTP endpoints for interacting with various embedding providers.
It supports text vectorization, similarity calculation, and dynamic loading/unloading of models.
"""

import os
import sys
import signal
import time
import atexit
from datetime import datetime
from typing import Any
from contextlib import asynccontextmanager

import uvicorn
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from loguru import logger

from platform_core.config.settings import get_embed_config, get_app_config
from platform_core.contracts import INTERNAL_API_V1, PUBLIC_API_V1
from platform_core.dictionary import ModelCategory
from platform_core.logger import LogConfig, LogManager
from platform_core.middleware.log_middleware import log_requests
from platform_core.database.oracle import create_database_runtime
from model_serving.embedding.embed_service import EmbeddingService
from model_serving.embedding.schema import (
    EmbeddingRequest, OpenAIEmbeddingRequest, SimilarityRequest,
)
from model_serving.embedding.model import EmbeddingResponse
from platform_core.platform.port_check import check_port_available
from model_serving.common.management_router import create_model_management_router
from model_serving.common.openai_router import create_openai_models_router
from model_serving.common.openai_contracts import openai_error_response
from model_serving.common.model_registry import ModelRegistryService

# Load environment variables
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Extract service parameters from configuration
config = get_embed_config()
SERVICE_NAME = config.service_name
SERVICE_VERSION = config.service_version
SERVICE_HOST = config.service_host
SERVICE_PORT = config.service_port

# Log parameters
app_config = get_app_config()
LOG_DIR = app_config.log.dir
LOG_LEVEL = app_config.log.level
LOG_ROTATION = app_config.log.rotation
LOG_RETENTION = app_config.log.retention
DEBUG_MODE = app_config.debug

# Instantiate embedding service singleton
embedding_service = EmbeddingService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the application lifecycle.

    On service startup: Initialize logging system, load embedding model resources, and perform warmup.
    On service shutdown: Release GPU memory or RAM resources occupied by models.

    Args:
        app: FastAPI application instance.
    """
    # Set service name to app.state (used by middleware)
    app.state.service_name = SERVICE_NAME
    db_runtime = create_database_runtime()
    app.state.db_runtime = db_runtime
    embedding_service.bind_session_factory(db_runtime.session_factory)
    app.state.model_registry = ModelRegistryService(
        app_id=app_config.app_id,
        session_factory=db_runtime.session_factory,
        on_model_changed=embedding_service.invalidate_model,
    )

    # 1. Initialize logging system
    log_conf = LogConfig(
        service_name=SERVICE_NAME,
        log_dir=LOG_DIR,
        level=LOG_LEVEL,
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION
    )
    LogManager(log_conf).setup()

    # 2. Start initialization process
    start_time = time.time()
    logger.info(f"正在初始化 Embedding 服务 | 进程号：{os.getpid()} | 时间：{datetime.now()}")

    try:
        await embedding_service.initialize()
        await embedding_service.warmup()
        logger.info(f"Embedding 服务启动成功 | 耗时：{time.time() - start_time:.2f}s")
    except Exception as e:
        logger.exception(f"Failed to initialize embedding service: {e}")
        # In production environment, core service initialization failure should force exit
        if not DEBUG_MODE:
            sys.exit(1)

    yield  # --- Service running ---

    # 3. Shutdown cleanup process
    logger.info("正在停止 Embedding 服务并释放资源...")
    shutdown_start = time.time()
    try:
        await embedding_service.shutdown()
        logger.info(f"资源释放完成 | 停止耗时：{time.time() - shutdown_start:.2f}s")
    except Exception as e:
        logger.error(f"释放资源时发生异常：{e}")
    finally:
        await db_runtime.close()


# Create FastAPI application instance
app = FastAPIOffline(
    title="Embedding Microservice",
    description="Provides high-performance Text Embedding and vector similarity calculation services.",
    version=SERVICE_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if DEBUG_MODE else None,
    redoc_url="/redoc" if DEBUG_MODE else None
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
app.middleware("http")(log_requests)

# Internal service authentication middleware
from platform_core.security import (
    create_api_client_auth_middleware,
    create_internal_auth_middleware,
)
app.middleware("http")(
    create_internal_auth_middleware(
        audience=SERVICE_NAME, skip_prefixes=(PUBLIC_API_V1,),
    )
)
app.middleware("http")(create_api_client_auth_middleware())
app.include_router(create_model_management_router(category=ModelCategory.TXT_EMBEDDING.value))
app.include_router(create_openai_models_router(category=ModelCategory.TXT_EMBEDDING.value))

# --- Dependency Injection ---

def get_embed_service() -> EmbeddingService:
    """Provide dependency injection for embedding service singleton.

    Returns:
        EmbeddingService: Global embedding service instance.
    """
    return embedding_service


# --- API Endpoint Definitions ---

@app.get("/health", response_model=dict[str, Any], tags=["System"], summary="Health check endpoint")
async def health_check() -> dict[str, Any]:
    """Get the running status of the microservice and information about loaded models.

    Returns:
        Dictionary containing service status, model count, and timestamp.
    """
    loaded_models_count = 0
    if embedding_service._initialized and hasattr(embedding_service._model_pool, '_models'):
        loaded_models_count = len(embedding_service._model_pool._models)

    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "loaded_models_count": loaded_models_count,
        "timestamp": datetime.now().isoformat()
    }


@app.post(f"{INTERNAL_API_V1}/embeddings", response_model=EmbeddingResponse, tags=["AI Service"], summary="文本向量化")
async def handle_embed_texts(
    request: EmbeddingRequest,
    embed_service: EmbeddingService = Depends(get_embed_service)
) -> EmbeddingResponse:
    """Convert input text list to vector embeddings.

    Args:
        request: Embedding request parameters including model name, text list, batch size, etc.
        embed_service: Injected embedding service instance.

    Returns:
        Response object containing embedding vectors, indices, and token usage information.

    Raises:
        HTTPException: 500 error when any logical error occurs during processing.
    """
    try:
        logger.info(f"正在处理 Embedding 请求 | 模型：{request.served_model_name} | 文本数量：{len(request.texts)}")
        return await embed_service.embed_texts(
            served_model_name=request.served_model_name,
            texts=request.texts,
            batch_size=request.batch_size,
            is_query=request.is_query
        )
    except Exception as e:
        logger.exception(f"Text vectorization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding processing exception: {str(e)}")


@app.post(
    f"{PUBLIC_API_V1}/embeddings",
    response_model=EmbeddingResponse,
    tags=["OpenAI Compatible"],
)
async def openai_embeddings(
    request: OpenAIEmbeddingRequest,
    embed_service: EmbeddingService = Depends(get_embed_service),
) -> Any:
    """按 OpenAI 请求格式生成向量。"""
    texts = [request.input] if isinstance(request.input, str) else request.input
    if not texts:
        return openai_error_response(
            status_code=400,
            message="input 不能为空",
            code="invalid_input",
        )
    configured_dimension = get_embed_config().dimensions
    if (
        request.dimensions is not None
        and configured_dimension is not None
        and request.dimensions != configured_dimension
    ):
        return openai_error_response(
            status_code=400,
            message=f"dimensions 必须为 {configured_dimension}",
            code="invalid_dimensions",
        )
    try:
        return await embed_service.embed_texts(
            served_model_name=request.model,
            texts=texts,
            is_query=True,
        )
    except Exception as exc:
        logger.error(f"OpenAI Embedding 调用失败：{exc}")
        return openai_error_response(
            status_code=500,
            message="模型推理失败",
            code="model_inference_failed",
            error_type="server_error",
        )


@app.post(f"{INTERNAL_API_V1}/similarity", response_model=dict[str, Any], tags=["AI Service"], summary="计算文本相似度")
async def handle_compute_similarity(
    request: SimilarityRequest,
    embed_service: EmbeddingService = Depends(get_embed_service)
) -> dict[str, Any]:
    """Calculate similarity score between two specified texts.

    Args:
        request: Request object containing model name, text pair, and calculation method (cosine/dot).
        embed_service: Injected embedding service instance.

    Returns:
        Dictionary containing similarity score.

    Raises:
        HTTPException: 500 error when exception occurs during calculation.
    """
    try:
        logger.info(f"正在处理相似度请求 | 模型：{request.served_model_name} | 方法：{request.method}")
        model = await embed_service.get_embedding_model(request.served_model_name)
        score = await embed_service.compute_similarity(
            served_model_name=request.served_model_name,
            text1=request.text1,
            text2=request.text2,
            method=request.method
        )
        return {"similarity": score, "method": request.method}
    except Exception as e:
        logger.exception(f"Similarity calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calculation process exception: {str(e)}")


# --- Process Signal Management ---

def signal_handler(sig: int, frame: Any):
    """Handle termination signals sent by the operating system to ensure graceful shutdown.

    Args:
        sig: Signal number.
        frame: Current stack frame.
    """
    logger.warning(f"收到系统信号：{sig}，准备停止服务...")
    # sys.exit(0) triggers cleanup logic in atexit and lifespan
    sys.exit(0)


# Register exit hook
atexit.register(lambda: logger.info("微服务进程已安全退出"))

if __name__ == "__main__":
    # Register signal listeners
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 先检查端口可用性，避免 EADDRINUSE 错误被 stderr 吞掉
    if not check_port_available(SERVICE_HOST, SERVICE_PORT, "embedding"):
        sys.exit(1)

    logger.info(f"正在启动 Embedding 微服务，监听地址：{SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(
        app,
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        log_config=None  # Use loguru to take over all logging
    )
