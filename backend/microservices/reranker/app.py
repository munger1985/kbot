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
import platform
from datetime import datetime
from typing import List, Optional, Any, Dict
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
    
from microservices.reranker.reranker_service import RerankerService
from core.config import settings

# 确保日志目录存在
log_dir = settings["logger"]["dir"]
os.makedirs(log_dir, exist_ok=True)

# 配置日志 - 使用 loguru，覆盖日志文件路径
logger.add(
    os.path.join(log_dir, "reranker_service.log"),
    rotation=settings["logger"]["rotation"],
    retention=settings["logger"]["retention"],
    level=settings["logger"]["level"]
)

# 创建reranker服务实例
reranker_service = RerankerService()

# 定义 lifespan 上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager. //应用程序生命周期上下文管理器"""
    # 启动事件
    start_time = time.time()
    logger.info(f"Initializing reranker service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Process ID: {os.getpid()}")
    
    # 初始化LLM服务
    try:
        await reranker_service.initialize()
        logger.info(f"Reranker service started successfully, elapsed time: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Reranker service initialization failed: {e}")
        # 在生产环境中，可能需要在这里退出应用程序
        current_env = os.getenv("KBOT_ENV")
        if current_env == "production":
            sys.exit(1)
    
    yield  # 服务运行期间
    
    # 关闭事件
    logger.info("Closing reranker service...")
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
    description="Provides text reranker services to convert text into vector representations.",
    version=settings["app"]["version"],
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
    model_id: int = Field(..., description="reranker model ID")
    query: str = Field(..., description="query")
    documents: List[str] = Field(..., description="List of documents to be reranked.")
    top_k: Optional[int] = Field(10, description="Number of top documents to return (None for all)")

# 定义响应模型
class RerankerResponse(BaseModel):
    rerankers: List[Dict[str, Any]] = Field(..., description="List of reranked documents.")

# 依赖项：获取reranker服务实例
def get_reranker_service():
    return reranker_service

@app.get("/health", response_model=dict, tags=["Reranker"])
async def health() -> Dict[str, Any]:
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

@app.post("/rerank", response_model=RerankerResponse, tags=["Reranker"])
async def rerank_texts(
    request: RerankerRequest,
    reranker_service: RerankerService = Depends(get_reranker_service)
    ) -> RerankerResponse:
    """
    将文本列表进行rerank
    - **model_id**: Model ID to use for reranking.
    - **query**: Query text to be reranked.
    - **documents**: List of documents to be reranked.
    - **top_k**: Number of top documents to return (None for all)
    """

    try:
        logger.info(f"Received reranker request: model={request.model_id}, query={request.query}, documents={len(request.documents)}, top_k={request.top_k}")
        
        # 使用嵌入服务将文本转换为向量
        rerankers = await reranker_service.rerank(
            model_id=request.model_id,
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
    logger.info("Start the reranker microservice as an independent process.")
    reranker_service_path = os.path.abspath(__file__)
    
    # 启动reranker微服务，使用环境变量中的端口并设置为独立模式
    process = subprocess.Popen(
        [sys.executable, reranker_service_path],
        env={**os.environ, "RERANKER_SERVICE_STANDALONE": "1"}
    )
    return process

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
    import uvicorn
    # 从环境变量获取主机和端口，如果没有设置，则使用默认值
    host = os.environ.get("KBOT_RERANKER_HOST", "0.0.0.0")
    port = int(os.environ.get("KBOT_RERANKER_PORT", 8003))
    
    # 如果是作为独立进程启动，则注册信号处理器
    if os.environ.get("RERANKER_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Started the reranker microservice, listening on {host}:{port}")
    uvicorn.run(app, host=host, port=port)