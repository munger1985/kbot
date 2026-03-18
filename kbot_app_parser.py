"""文档解析微服务应用程序。"""

import os
import sys
import signal
import time
from datetime import datetime
from typing import Any
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from loguru import logger

from core.config.settings import get_parser_config, get_app_config
from core.logger import LogConfig, LogManager
from core.middleware.log_middleware import log_requests
from microservices.file_processor.services import FileParseEngine

# 加载环境变量
load_dotenv()

# 从配置中心获取服务配置
config = get_parser_config()
SERVICE_NAME = config.service_name
SERVICE_VERSION = config.service_version
SERVICE_HOST = config.service_host
SERVICE_PORT = config.service_port

# 获取通用应用配置
app_config = get_app_config()
DEBUG: bool = app_config.debug
LOG_DIR: str = app_config.log.dir
LOG_LEVEL: str = app_config.log.level
LOG_ROTATION: str = app_config.log.rotation
LOG_RETENTION: str = app_config.log.retention

# 初始化全局单例
parallel_workers = config.parser_parallel
db_check_interval = config.db_check_interval
parse_engine = FileParseEngine(parallel_workers=parallel_workers, check_interval=db_check_interval)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理服务生命周期：初始化日志和解析单例。"""

    # 设置服务名称到 app.state（供中间件使用）
    app.state.service_name = SERVICE_NAME

    # 1. 初始化日志系统 (对应 LLM 微服务做法)
    log_conf = LogConfig(
        service_name=SERVICE_NAME, 
        log_dir=LOG_DIR, 
        level=LOG_LEVEL, 
        rotation=LOG_ROTATION, 
        retention=LOG_RETENTION
    )
    LogManager(log_conf).setup()
    
    start_time = time.time()
    logger.info(f"正在启动 [{SERVICE_NAME}] | PID: {os.getpid()} | 时间: {datetime.now()}")

    try:
        # 2. 初始化解析服务
        logger.info("正在启动文件解析引擎...")
        await parse_engine.start()
        logger.success(f"文件解析引擎加载成功 | 耗时: {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"文件解析引擎启动失败: {e}")
        if not DEBUG:
            sys.exit(1)
    
    yield  # --- 此时 Web 服务和后台轮询任务都在主进程中运行 ---
    
    # 3. 清理阶段
    logger.info("应用正在关闭，执行清理任务...")
    try:
        # 停止后台任务
        await parse_engine.stop()
        logger.info("文件解析引擎已停止")

    except Exception as e:
        logger.error(f"清理资源时发生异常: {e}")
    

# 创建应用实例
app = FastAPIOffline(
    title=f"{SERVICE_NAME} API",
    description="基于 Docling 的多格式解析服务，支持 OCR 和动态 VLM 语义增强。",
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

# 请求日志中间件
app.middleware("http")(log_requests)

# --- API 端点 ---

@app.get("/health", tags=["System"], summary="健康检查")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "timestamp": datetime.now().isoformat()
    }

# --- 启动逻辑 ---

if __name__ == "__main__":
    logger.info(f"服务启动中 -> {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(
        app, 
        host=SERVICE_HOST, 
        port=SERVICE_PORT,
        log_config=None,
        # 确保在使用 Ctrl+C 时 uvicorn 能够控制退出流程
        loop="asyncio" 
    )