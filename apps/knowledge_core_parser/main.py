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

from platform_core.config.settings import get_parser_config, get_app_config
from platform_core.logger import LogConfig, LogManager
from platform_core.middleware.log_middleware import log_requests
from knowledge_core.parsing.converter import KcDoclingConverter
from knowledge_core.parsing.pipeline import KcParsingPipeline
from knowledge_core.workers.parser.client import KcParseClient
from knowledge_core.workers.parser.worker import KcParserWorker
from knowledge_core.workers.parser.visual_enricher import KcVisualEnricher
from platform_core.platform.port_check import check_port_available

# Load environment variables
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

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

# Initialize the V2 lease worker. It never polls or writes V1 File/Chunk tables.
parse_worker = KcParserWorker(
    client=KcParseClient(base_url=config.knowledge_core_url, timeout_seconds=config.timeout),
    converter=KcDoclingConverter(artifacts_path=config.local_artifacts_path),
    pipeline=KcParsingPipeline(parser_version=SERVICE_VERSION),
    worker_id=config.worker_id,
    lease_seconds=config.lease_seconds,
    poll_interval=config.claim_interval_seconds,
    evidence_batch_size=config.evidence_batch_size,
    visual_enricher=KcVisualEnricher(),
)

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
    logger.info(f"正在启动 [{SERVICE_NAME}] | 进程号: {os.getpid()} | 时间: {datetime.now()}")

    try:
        logger.info("正在启动 KC 解析 Worker...")
        await parse_worker.start()
        logger.success(f"KC V2 parser worker loaded | Elapsed time: {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"文件解析引擎启动失败：{e}")
        if not DEBUG:
            sys.exit(1)
    
    yield  # --- At this point, both Web service and background polling tasks are running in the main process ---
    
    # 3. Cleanup phase
    logger.info("应用正在停止，开始清理资源...")
    try:
        await parse_worker.stop()
        logger.info("KC 解析 Worker 已停止")

    except Exception as e:
        logger.error(f"清理资源时发生异常：{e}")
    

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
from platform_core.platform.security import create_internal_auth_middleware
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
    # 先检查端口可用性，避免 EADDRINUSE 错误被 stderr 吞掉
    if not check_port_available(SERVICE_HOST, SERVICE_PORT, SERVICE_NAME):
        sys.exit(1)

    logger.info(f"服务开始监听 -> {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(
        app, 
        host=SERVICE_HOST, 
        port=SERVICE_PORT,
        log_config=None,
        # Ensure uvicorn can control exit flow when using Ctrl+C
        loop="asyncio" 
    )
