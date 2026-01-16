"""Reranker 微服务应用程序。

本模块提供基于 FastAPI 的 Reranker 服务，用于对候选文档列表进行语义重排序。
支持多模型管理、动态加载/卸载以及标准化的重排序 API 接口。
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

# 加载环境变量
load_dotenv()

# 从配置中心获取服务配置
config = get_reranker_config()
SERVICE_NAME: str = config.service_name
SERVICE_VERSION: str = config.service_version
SERVICE_HOST: str = config.service_host
SERVICE_PORT: int = config.service_port

# 获取通用应用配置
app_config = get_app_config()
DEBUG: bool = app_config.debug
LOG_DIR: str = app_config.log.dir
LOG_LEVEL: str = app_config.log.level
LOG_ROTATION: str = app_config.log.rotation
LOG_RETENTION: str = app_config.log.retention

# 初始化 Reranker 逻辑服务实例
reranker_service = RerankerService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理 Reranker 应用程序的生命周期。

    Args:
        app: FastAPI 应用程序实例。
    """
    # 设置服务名称到 app.state（供中间件使用）
    app.state.service_name = SERVICE_NAME

    # 1. 初始化日志配置
    log_conf = LogConfig(
        service_name=SERVICE_NAME, 
        log_dir=LOG_DIR, 
        level=LOG_LEVEL, 
        rotation=LOG_ROTATION, 
        retention=LOG_RETENTION
    )
    LogManager(log_conf).setup()
    
    # 2. 启动初始化过程
    start_time = time.time()
    logger.info(f"正在初始化 Reranker 服务 | PID: {os.getpid()} | 时间: {datetime.now()}")
    
    try:
        await reranker_service.initialize()
        await reranker_service.warmup()
        logger.info(f"Reranker 服务初始化成功 | 耗时: {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Reranker 服务启动失败: {e}")
        if not DEBUG:
            sys.exit(1)
    
    yield  # --- 服务运行中 ---
    
    # 3. 执行关闭清理
    logger.info("正在关闭 Reranker 服务...")
    shutdown_start = time.time()
    try:
        await reranker_service.shutdown()
        logger.info(f"Reranker 服务已安全关闭 | 停机耗时: {time.time() - shutdown_start:.2f}s")
    except Exception as e:
        logger.error(f"关闭服务时出错: {e}")


# 创建 FastAPI 实例
app = FastAPIOffline(
    title="Reranker 微服务",
    description="提供基于深度学习模型的文本语义重排序服务",
    version=SERVICE_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None
)

# 配置跨域中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求日志中间件
app.middleware("http")(log_requests)


def get_reranker_service() -> RerankerService:
    """获取 Reranker 服务单例的依赖项。"""
    return reranker_service


@app.get("/health", response_model=dict, tags=["System"], summary="健康检查接口")
async def health_check() -> dict[str, Any]:
    """检查微服务健康状态及模型加载情况。

    Returns:
        包含状态 (status)、已加载模型数 (loaded_models_count) 和时间戳的字典。
    """
    loaded_models_count = 0
    if reranker_service._initialized and hasattr(reranker_service._model_pool, '_models'):
        loaded_models_count = len(reranker_service._model_pool._models)
    
    return {
        "status": "ok",
        "loaded_models_count": loaded_models_count,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/load", response_model=dict, tags=["Management"], summary="模型动态加载/卸载")
async def toggle_model(request: ToggleModelRequest) -> dict[str, str]:
    """根据请求动态加载或从内存中卸载指定模型。

    Args:
        request: 包含模型名称和操作类型 (load/unload) 的对象。

    Returns:
        操作状态响应。

    Raises:
        HTTPException: 操作失败时抛出 500 错误。
    """
    model_name = request.model_name
    try:
        if request.operation == "load":
            logger.info(f"执行模型加载指令: {model_name}")
            success = await reranker_service.load_model(model_name)
        else:
            logger.info(f"执行模型卸载指令: {model_name}")
            success = await reranker_service.unload_model(model_name)
            
        if not success:
            raise HTTPException(status_code=500, detail=f"模型 {model_name} {request.operation} 操作未成功")
            
        return {"status": "success", "model_name": model_name}
    except Exception as e:
        logger.exception(f"执行模型操作时发生异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rerank", response_model=RerankerResponse, tags=["Reranker"], summary="执行重排序")
async def rerank_documents(
    request: RerankerRequest,
    service: RerankerService = Depends(get_reranker_service)
) -> RerankerResponse:
    """根据查询语句对文档列表进行相关性重排序。

    Args:
        request: 包含 query, documents 以及 top_k 等参数。
        service: Reranker 逻辑服务实例。

    Returns:
        包含重排序结果列表（带得分）的响应对象。

    Raises:
        HTTPException: 处理过程中发生错误时抛出 500 错误。
    """
    try:
        logger.info(
            f"收到重排序请求 | 模型: {request.model_name} | "
            f"文档数: {len(request.documents)} | top_k: {request.top_k}"
        )
        
        results = await service.rerank(
            model_name=request.model_name,
            query=request.query,
            documents=request.documents,
            top_k=request.top_k
        )
        
        logger.info(f"重排序计算完成 | 结果数: {len(results)}")
        return RerankerResponse(rerankers=results)
    
    except Exception as e:
        logger.error(f"重排序计算失败: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Rerank Error: {e}")


# --- 进程管理与信号处理 ---

reranker_process: subprocess.Popen | None = None


def stop_reranker_standalone():
    """终止作为独立进程运行的微服务。"""
    global reranker_process
    if reranker_process:
        logger.info("正在安全终止 Reranker 独立进程...")
        try:
            reranker_process.terminate()
            reranker_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("进程终止超时，正在强制 Kill...")
            reranker_process.kill()
        reranker_process = None


def handle_exit_signal(sig: int, frame: Any):
    """捕获退出信号后的回调处理。"""
    logger.info(f"收到系统信号: {sig}，准备退出...")
    stop_reranker_standalone()
    sys.exit(0)


# 注册全局退出处理逻辑
atexit.register(stop_reranker_standalone)

if __name__ == "__main__":
    # 检查是否为独立运行模式，并设置信号监听
    if os.environ.get("RERANKER_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, handle_exit_signal)
        signal.signal(signal.SIGTERM, handle_exit_signal)
    
    logger.info(f"Reranker 微服务已启动 | 监听地址: {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)