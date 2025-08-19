"""Reranker microservice application.

This module provides a FastAPI application that exposes HTTP endpoints for interacting
with various reranker providers. It supports text rerank.

该模块提供 FastAPI 微服务应用程序，用于公开与各种嵌入提供者交互的 HTTP 端点。它支持文本嵌入。

"""

import os
import sys
import signal
import subprocess
import time
import atexit
import uvicorn
from datetime import datetime
from dotenv import load_dotenv
from typing import Any
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from reranker_service import RerankerService
from ms_core import nacos_manager, load_config, AppConfig, ModelConfig, LogManager, LogConfig


# 加载环境变量配置
load_dotenv()

try:
    # 从 nacos 获取 reranker 服务配置
    config = load_config("model_config")
    if not isinstance(config, ModelConfig):
        raise ValueError
    service_name = config.reranker.service_name or "reranker-service" # 全局微服务名称
    service_version = config.reranker.service_version or "1.0.0" # 微服务版本
    service_host = config.reranker.service_host or "0.0.0.0" # 微服务地址
    service_port = config.reranker.service_port or 9203 # 微服务通信端口
except Exception as e:
    # 如果从 nacos 获取 reranker 服务配置失败，则使用默认配置
    logger.warning("Failed to get reranker service config from nacos: {}".format(e))
    service_name = "reranker-service"
    service_version = "1.0.0"
    service_host = "0.0.0.0"
    service_port = 9203

# 创建reranker服务实例
reranker_service = RerankerService()

# 定义 lifespan 上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager. //应用程序生命周期上下文管理器"""
    # 通过 nacos_manager 获取 logger 配置
    try:
        log_config = load_config("app_config")
        if not isinstance(log_config, AppConfig):
            raise ValueError
        
        log_dir = log_config.kbot.log.dir or "logs/"
        log_level = log_config.kbot.log.level or "DEBUG"
        rotation = log_config.kbot.log.rotation or "10 MB"
        retention = log_config.kbot.log.retention or "20 days"
        
    except Exception as e:
        # 如果获取 logger 配置失败，则使用默认配置
        logger.warning(f"Failed to get logger config from nacos: {str(e)}")
        log_dir = "logs/"
        log_level = "DEBUG"
        rotation = "10 MB"
        retention = "10 days"
    
    # 初始化日志
    conf = LogConfig(service_name=service_name, log_dir=log_dir, level=log_level, rotation=rotation, retention=retention)
    LogManager(conf).setup()
    
    # 启动事件
    start_time = time.time()
    logger.info(f"Initializing reranker service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    logger.info(f"Process ID: {os.getpid()}")

    
    # 初始化reranker服务
    try:
        await reranker_service.initialize()
        logger.info(f"Reranker service started successfully, elapsed time: {time.time() - start_time:.2f} seconds")

        # 注册服务到 Nacos
        nacos_manager.register_service(service_name=service_name, service_host=service_host, service_port=service_port)
        logger.info("Reranker service registered to Nacos.")

    except Exception as e:
        logger.error(f"Reranker service initialization failed: {e}")
        # 在生产环境中，可能需要在这里退出应用程序
        current_env = os.getenv("NACOS_GROUP", "dev")
        if current_env == "prod":
            sys.exit(1)
    
    yield  # 服务运行期间
    
    # 关闭事件
    logger.info(f"Closing reranker service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    shutdown_start = time.time()
    
    try:
        await reranker_service.shutdown()
        logger.info("reranker service is closed.")
    except Exception as e:
        logger.error(f"reranker service shutdown failed: {e}")
    
    logger.info(f"reranker service closed successfully, elapsed time: {time.time() - shutdown_start:.2f} seconds")
    logger.info(f"Total running time: {time.time() - start_time:.2f} seconds")

# 创建 FastAPI 应用
app = FastAPI(
    title="reranker service",
    description="Provides text reranker services.",
    version=service_version,
    lifespan=lifespan,
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求模型
class RerankerRequest(BaseModel):
    model_unique_name: str = Field(..., description="Reranker model ID")
    query: str = Field(..., description="query")
    documents: list[str] = Field(..., description="List of documents to be reranked.")
    top_k: int | None = Field(10, description="Number of top documents to return (None for all)")

# 定义响应模型
class RerankerResponse(BaseModel):
    rerankers: list[dict[str, Any]] = Field(..., description="List of reranked documents.")

# 依赖项：获取reranker服务实例
def get_reranker_service():
    return reranker_service

@app.get("/health", response_model=dict, tags=["Reranker"])
async def health() -> dict[str, Any]:
    """Health check endpoint. //微服务接口健康检查
    Returns:
        Loaded models count. //已加载的模型数量
    """
    
    # 获取已加载的模型信息
    loaded_models = {}
    if reranker_service._initialized and hasattr(reranker_service._model_pool, '_models'):
        loaded_models = reranker_service._model_pool._models
    
    # 返回已加载模型数量
    return {
        "status": "ok",
        "loaded_models_count": len(loaded_models),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/v1/rerank", response_model=RerankerResponse, tags=["Reranker"])
async def rerank_texts(
    request: RerankerRequest,
    reranker_service: RerankerService = Depends(get_reranker_service)
    ) -> RerankerResponse:
    """
    将文本列表进行rerank
    - **model_unique_name**: Model ID to use for reranking.
    - **query**: Query text to be reranked.
    - **documents**: list of documents to be reranked.
    - **top_k**: Number of top documents to return (None for all)
    """

    try:
        logger.info(f"Received reranker request: model={request.model_unique_name}, query={request.query}, documents={len(request.documents)}, top_k={request.top_k}")
        
        # 使用嵌入服务将文本转换为向量
        rerankers = await reranker_service.rerank(
            model_unique_name=request.model_unique_name,
            query=request.query,
            documents=request.documents,
            top_k=request.top_k
        )
        
        logger.info(f"Rerank completed: number of rerankers={len(rerankers)}, top_k={request.top_k}")
        return RerankerResponse(rerankers=rerankers)
    
    except Exception as e:
        logger.error(f"Error occurred during reranker.: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error occurred during reranker.: {str(e)}")



# 全局变量，用于存储微服务进程
reranker_service_process = None

def start_reranker_service():
    """Start the reranker microservice as an independent process."""

    # 启动reranker微服务，使用环境变量中的端口并设置为独立模式
    try:
        logger.info("Start the reranker microservice as an independent process.")
        reranker_service_path = os.path.abspath(__file__)
        
        process = subprocess.Popen(
            [sys.executable, reranker_service_path],
            env={**os.environ, "RERANKER_SERVICE_STANDALONE": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 检查进程是否成功启动
        if process.poll() is not None:
            stderr = process.stderr.read().decode('utf-8') if process.stderr else ""
            raise RuntimeError(f"Failed to start LLM service: {stderr}")
            
        logger.success(f"LLM service started successfully with PID {process.pid}")
        return process
        
    except Exception as e:
        logger.exception(f"Error starting LLM service: {str(e)}")
        raise

def shutdown_reranker_service():
    """Terminate the reranker microservice process."""
    global reranker_service_process
    if reranker_service_process:
        logger.info("Terminating the reranker microservice process...")
        try:
            reranker_service_process.terminate()
            reranker_service_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("The reranker microservice process failed to terminate properly; forcing shutdown...")
            reranker_service_process.kill()
        reranker_service_process = None


def signal_handler(sig, frame):
    """Handling termination signal."""
    logger.info(f"Signal received: {sig}, shutting down....")
    shutdown_reranker_service()
    sys.exit(0)

# 注册退出处理程序，确保在应用程序退出时关闭微服务
atexit.register(shutdown_reranker_service)

if __name__ == "__main__":
    # 如果是作为独立进程启动，则注册信号处理器
    if os.environ.get("RERANKER_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Started the reranker microservice, listening on {service_host}:{service_port}")
    uvicorn.run(app, host=service_host, port=service_port)