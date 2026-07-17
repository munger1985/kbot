"""Document Parsing Microservice Application."""

import os
import sys
import signal
import time
from datetime import datetime
from typing import Any
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
from pathlib import Path
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from loguru import logger

from core.config.settings import get_parser_config, get_app_config
from core.logger import LogConfig, LogManager
from core.middleware.log_middleware import log_requests
from microservices.file_processor.services import FileParseEngine

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# Get service configuration from config center
config = get_parser_config()
SERVICE_NAME = config.service_name
SERVICE_VERSION = config.service_version
SERVICE_HOST = config.service_host
SERVICE_PORT = config.service_port

# Get general application configuration
app_config = get_app_config()
DEBUG: bool = app_config.debug
LOG_DIR: str = app_config.log.dir
LOG_LEVEL: str = app_config.log.level
LOG_ROTATION: str = app_config.log.rotation
LOG_RETENTION: str = app_config.log.retention

# Initialize global singleton
parallel_workers = config.parser_parallel
db_check_interval = config.db_check_interval
parse_engine = FileParseEngine(parallel_workers=parallel_workers, check_interval=db_check_interval)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle: initialize logging and parsing singleton."""

    # Set service name to app.state (used by middleware)
    app.state.service_name = SERVICE_NAME

    # 1. Initialize logging system (following LLM microservice implementation)
    log_conf = LogConfig(
        service_name=SERVICE_NAME, 
        log_dir=LOG_DIR, 
        level=LOG_LEVEL, 
        rotation=LOG_ROTATION, 
        retention=LOG_RETENTION
    )
    LogManager(log_conf).setup()
    
    start_time = time.time()
    logger.info(f"Starting [{SERVICE_NAME}] | PID: {os.getpid()} | Time: {datetime.now()}")

    try:
        # 2. Initialize parsing service
        logger.info("Starting file parsing engine...")
        await parse_engine.start()
        logger.success(f"File parsing engine loaded successfully | Elapsed time: {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to start file parsing engine: {e}")
        if not DEBUG:
            sys.exit(1)
    
    yield  # --- At this point, both Web service and background polling tasks are running in the main process ---
    
    # 3. Cleanup phase
    logger.info("Application is shutting down, executing cleanup tasks...")
    try:
        # Stop background tasks
        await parse_engine.stop()
        logger.info("File parsing engine has been stopped")

    except Exception as e:
        logger.error(f"Exception occurred while cleaning up resources: {e}")
    

# Create application instance
app = FastAPIOffline(
    title=f"{SERVICE_NAME} API",
    description="Multi-format parsing service based on Docling, supporting OCR and dynamic VLM semantic enhancement.",
    version=SERVICE_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None
)

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

# --- API Endpoints ---

@app.get("/health", tags=["System"], summary="Health Check")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "timestamp": datetime.now().isoformat()
    }

# --- Startup Logic ---

if __name__ == "__main__":
    logger.info(f"Service starting up -> {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(
        app, 
        host=SERVICE_HOST, 
        port=SERVICE_PORT,
        log_config=None,
        # Ensure uvicorn can control exit flow when using Ctrl+C
        loop="asyncio" 
    )