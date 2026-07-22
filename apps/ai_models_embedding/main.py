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
from platform_core.dictionary import ModelCategory
from platform_core.logger import LogConfig, LogManager
from platform_core.middleware.log_middleware import log_requests
from platform_core.database.oracle import create_database_runtime
from model_serving.embedding.embed_service import EmbeddingService
from model_serving.embedding.schema import (
    EmbeddingRequest, SimilarityRequest, ToggleModelRequest
)
from model_serving.embedding.model import EmbeddingResponse
from platform_core.platform.port_check import check_port_available
from model_serving.common.management_router import create_model_management_router
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
        app_id=app_config.app_id, session_factory=db_runtime.session_factory,
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
    logger.info(f"Initializing embedding service | Process ID: {os.getpid()} | Time: {datetime.now()}")

    try:
        await embedding_service.initialize()
        await embedding_service.warmup()
        logger.info(f"Embedding service started successfully | Elapsed time: {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.exception(f"Failed to initialize embedding service: {e}")
        # In production environment, core service initialization failure should force exit
        if not DEBUG_MODE:
            sys.exit(1)

    yield  # --- Service running ---

    # 3. Shutdown cleanup process
    logger.info("Shutting down embedding service and releasing resources...")
    shutdown_start = time.time()
    try:
        await embedding_service.shutdown()
        logger.info(f"Resource release completed | Shutdown elapsed time: {time.time() - shutdown_start:.2f}s")
    except Exception as e:
        logger.error(f"Exception occurred while releasing resources: {e}")
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
from platform_core.platform.security import create_internal_auth_middleware
app.middleware("http")(create_internal_auth_middleware())
app.include_router(create_model_management_router(category=ModelCategory.TXT_EMBEDDING.value))

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


# @app.post("/load", response_model=dict[str, Any], tags=["Management"], summary="Dynamically manage model status")
# async def handle_toggle_model(request: ToggleModelRequest) -> dict[str, Any]:
#     """Load or unload a specific embedding model according to the instruction.

#     Args:
#         request: Request object containing model name and operation type (load/unload).

#     Returns:
#         Operation result status.

#     Raises:
#         HTTPException: 500 error when operation fails or model does not exist.
#     """
#     try:
#         if request.operation == "load":
#             logger.info(f"Executing model load task: {request.model_name}")
#             success = await embedding_service.load_model(request.model_name)
#         else:
#             logger.info(f"Executing model unload task: {request.model_name}")
#             success = await embedding_service.unload_model(request.model_name)

#         if not success:
#             raise ValueError(f"Failed to {request.operation} model {request.model_name}")

#         return {"status": "success", "model_name": request.model_name, "operation": request.operation}

#     except Exception as e:
#         logger.error(f"Model management operation exception: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings", response_model=EmbeddingResponse, tags=["AI Service"], summary="Text vectorization endpoint")
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
        logger.info(f"Processing embedding request | Model: {request.model_name} | Text count: {len(request.texts)}")
        return await embed_service.embed_texts(
            model_name=request.model_name,
            texts=request.texts,
            batch_size=request.batch_size,
            is_query=request.is_query
        )
    except Exception as e:
        logger.exception(f"Text vectorization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding processing exception: {str(e)}")


@app.post("/v1/similarity", response_model=dict[str, Any], tags=["AI Service"], summary="Calculate text similarity endpoint")
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
        logger.info(f"Processing similarity request | Model: {request.model_name} | Method: {request.method}")
        model = await embed_service.get_embedding_model(request.model_name)
        score = await embed_service.compute_similarity(
            model_name=request.model_name,
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
    logger.warning(f"Received system signal: {sig}, preparing to shutdown service...")
    # sys.exit(0) triggers cleanup logic in atexit and lifespan
    sys.exit(0)


# Register exit hook
atexit.register(lambda: logger.info("Microservice process exited safely"))

if __name__ == "__main__":
    # Register signal listeners
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 先检查端口可用性，避免 EADDRINUSE 错误被 stderr 吞掉
    if not check_port_available(SERVICE_HOST, SERVICE_PORT, "embedding"):
        sys.exit(1)

    logger.info(f"Starting embedding microservice, listening on: {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(
        app,
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        log_config=None  # Use loguru to take over all logging
    )
