"""Reranker Microservice Application.

This module provides a FastAPI-based Reranker service for semantic reordering of candidate document lists.
It supports multi-model management, dynamic loading/unloading, and standardized reordering API endpoints.
"""

import os
import sys
import signal
import subprocess
import time
import atexit
from datetime import datetime
from typing import Any
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from loguru import logger

from core.config.settings import get_reranker_config, get_app_config
from core.logger import LogConfig, LogManager
from core.middleware.log_middleware import log_requests
from microservices.reranker.reranker_service import RerankerService
from microservices.reranker.schema import (
    RerankerRequest,
    RerankerResponse,
    ToggleModelRequest
)

# Load environment variables
load_dotenv()

# Get service configuration from config center
config = get_reranker_config()
SERVICE_NAME: str = config.service_name
SERVICE_VERSION: str = config.service_version
SERVICE_HOST: str = config.service_host
SERVICE_PORT: int = config.service_port

# Get general application configuration
app_config = get_app_config()
DEBUG: bool = app_config.debug
LOG_DIR: str = app_config.log.dir
LOG_LEVEL: str = app_config.log.level
LOG_ROTATION: str = app_config.log.rotation
LOG_RETENTION: str = app_config.log.retention

# Initialize Reranker logic service instance
reranker_service = RerankerService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the Reranker application.

    Args:
        app: FastAPI application instance.
    """
    # Set service name to app.state (used by middleware)
    app.state.service_name = SERVICE_NAME

    # 1. Initialize logging configuration
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
    logger.info(f"Initializing Reranker service | PID: {os.getpid()} | Time: {datetime.now()}")
    
    try:
        await reranker_service.initialize()
        await reranker_service.warmup()
        logger.info(f"Reranker service initialized successfully | Elapsed time: {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to start Reranker service: {e}")
        if not DEBUG:
            sys.exit(1)
    
    yield  # --- Service running ---
    
    # 3. Perform shutdown cleanup
    logger.info("Shutting down Reranker service...")
    shutdown_start = time.time()
    try:
        await reranker_service.shutdown()
        logger.info(f"Reranker service shut down safely | Shutdown elapsed time: {time.time() - shutdown_start:.2f}s")
    except Exception as e:
        logger.error(f"Error while shutting down service: {e}")


# Create FastAPI instance
app = FastAPIOffline(
    title="Reranker Microservice",
    description="Provides text semantic reordering service based on deep learning models",
    version=SERVICE_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None
)

# Configure CORS middleware
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
from microservices.common.security import create_internal_auth_middleware
app.middleware("http")(create_internal_auth_middleware())

def get_reranker_service() -> RerankerService:
    """Dependency for getting Reranker service singleton."""
    return reranker_service


@app.get("/health", response_model=dict, tags=["System"], summary="Health Check Endpoint")
async def health_check() -> dict[str, Any]:
    """Check microservice health status and model loading status.

    Returns:
        Dictionary containing status, number of loaded models, and timestamp.
    """
    loaded_models_count = 0
    if reranker_service._initialized and hasattr(reranker_service._model_pool, '_models'):
        loaded_models_count = len(reranker_service._model_pool._models)
    
    return {
        "status": "ok",
        "loaded_models_count": loaded_models_count,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/load", response_model=dict, tags=["Management"], summary="Dynamic Model Load/Unload")
async def toggle_model(request: ToggleModelRequest) -> dict[str, str]:
    """Dynamically load or unload the specified model from memory based on request.

    Args:
        request: Object containing model name and operation type (load/unload).

    Returns:
        Operation status response.

    Raises:
        HTTPException: 500 error when operation fails.
    """
    model_name = request.model_name
    try:
        if request.operation == "load":
            logger.info(f"Executing model load command: {model_name}")
            success = await reranker_service.load_model(model_name)
        else:
            logger.info(f"Executing model unload command: {model_name}")
            success = await reranker_service.unload_model(model_name)
            
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to {request.operation} model {model_name}")
            
        return {"status": "success", "model_name": model_name}
    except Exception as e:
        logger.exception(f"Exception occurred during model operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rerank", response_model=RerankerResponse, tags=["Reranker"], summary="Perform Reordering")
async def rerank_documents(
    request: RerankerRequest,
    service: RerankerService = Depends(get_reranker_service)
) -> RerankerResponse:
    """Reorder document list by relevance to query statement.

    Args:
        request: Contains parameters such as query, documents, and top_k.
        service: Reranker logic service instance.

    Returns:
        Response object containing reordering result list (with scores).

    Raises:
        HTTPException: 500 error when errors occur during processing.
    """
    try:
        logger.info(
            f"Received reordering request | Model: {request.model_name} | "
            f"Document count: {len(request.documents)} | top_k: {request.top_k}"
        )
        results = await service.rerank(
            model_name=request.model_name,
            query=request.query,
            documents=request.documents,
            top_k=request.top_k
        )
        
        logger.info(f"Reordering calculation completed | Result count: {len(results)}")
        return RerankerResponse(rerankers=results)
    
    except Exception as e:
        logger.error(f"Reordering calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Rerank Error: {e}")


# --- Process Management and Signal Handling ---

reranker_process: subprocess.Popen | None = None


def stop_reranker_standalone():
    """Terminate microservice running as independent process."""
    global reranker_process
    if reranker_process:
        logger.info("Safely terminating Reranker standalone process...")
        try:
            reranker_process.terminate()
            reranker_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Process termination timeout, force killing...")
            reranker_process.kill()
        reranker_process = None


def handle_exit_signal(sig: int, frame: Any):
    """Callback handler after catching exit signal."""
    logger.info(f"Received system signal: {sig}, preparing to exit...")
    stop_reranker_standalone()
    sys.exit(0)


# Register global exit handling logic
atexit.register(stop_reranker_standalone)

if __name__ == "__main__":
    # Check if running in standalone mode and set up signal listeners
    if os.environ.get("RERANKER_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, handle_exit_signal)
        signal.signal(signal.SIGTERM, handle_exit_signal)
    
    logger.info(f"Reranker microservice started | Listening on: {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)