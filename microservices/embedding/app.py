"""Embedding microservice application.

This module provides a FastAPI application that exposes HTTP endpoints for interacting
with various embedding providers. It supports text embedding.

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
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
    
from microservices.embedding.embed_service import EmbeddingService
from models.embedding.base import EmbeddingResponse, EmbeddingDataItem
from core.config import settings
from core.log.logger import setup_logging

# 初始化日志
setup_logging(service_name="embedding")

# 创建embedding服务实例
embedding_service = EmbeddingService()

# 定义 lifespan 上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager. //应用程序生命周期上下文管理器"""
    # 启动事件
    start_time = time.time()
    logger.info(f"Initializing embedding service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...") 
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Process ID: {os.getpid()}")
    
    # 初始化LLM服务
    try:
        await embedding_service.initialize()
        logger.info(f"Embedding service started successfully, elapsed time: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Embedding service initialization failed: {e}")
        # 在生产环境中，可能需要在这里退出应用程序
        current_env = os.getenv("KBOT_ENV")
        if current_env == "production":
            sys.exit(1)
    
    yield  # 服务运行期间
    
    # 关闭事件
    logger.info("Closing embedding service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    shutdown_start = time.time()
    
    try:
        await embedding_service.shutdown()
        logger.info("Embedding service is closed.")
    except Exception as e:
        logger.error(f"Embedding service shutdown failed: {e}")
    
    logger.info(f"Embedding service closed successfully, elapsed time: {time.time() - shutdown_start:.2f} seconds")
    logger.info(f"Total running time: {time.time() - start_time:.2f} seconds")

# 创建 FastAPI 应用
app = FastAPI(
    title="Embedding service",
    description="Provides text embedding services to convert text into vector representations.",
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
class EmbeddingRequest(BaseModel):
    model_unique_name: str = Field(..., description="Embedding model unique name")
    texts: List[str] = Field(..., description="List of texts to be embedded.")
    batch_size: Optional[int] = Field(32, description="Batch size")


# 依赖项：获取嵌入服务实例
def get_embed_service():
    return embedding_service

@app.get("/health", response_model=dict, tags=["Embedding"])
async def health() -> Dict[str, Any]:
    """Health check endpoint. //微服务接口健康检查
    Returns:
        Loaded models count. //已加载的模型数量
    """
    
    # 获取已加载的模型信息
    loaded_models = {}
    if embedding_service._initialized and hasattr(embedding_service._model_pool, '_models'):
        loaded_models = embedding_service._model_pool._models
    
    # 返回已加载模型数量
    return {
        "status": "ok",
        "loaded_models_count": len(loaded_models),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/embed", response_model=EmbeddingResponse, tags=["Embedding"])
@app.post("/v1/embeddings", response_model=EmbeddingResponse, tags=["Embedding"])
async def embed_texts(
    request: EmbeddingRequest,
    embed_service: EmbeddingService = Depends(get_embed_service)
    ) -> EmbeddingResponse:
    """
    将文本列表转换为嵌入向量
    
    - **model_unique_name**: 要使用的嵌入模型的ID
    - **texts**: 要嵌入的文本列表
    - **batch_size**: 批处理大小（可选）
    
    返回:
    - **data**: 嵌入向量列表
    - **usage**: 使用情况信息，包括总令牌数和提示令牌数
    - **model**: 使用的嵌入模型ID
    - **object**: 响应对象类型，固定为 "list"
    
    Raises:
    - **HTTPException**: 如果模型ID不存在或嵌入失败
    - **RuntimeError**: 如果模型创建失败
    - **Exception**: 如果发生其他错误
    """
    try:
        logger.info(f"Received embedding request: model={request.model_unique_name}, Number of texts.={len(request.texts)}")
        
        # 使用嵌入服务将文本转换为向量
        embeddings = await embed_service.embed_texts(
            model_unique_name=request.model_unique_name,
            texts=request.texts,
            batch_size=request.batch_size # type: ignore
        )
        
        return embeddings
    
    except Exception as e:
        logger.error(f"Error occurred during embedding.: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error occurred during embedding.: {str(e)}")



# 全局变量，用于存储微服务进程
embedding_service_process = None

def start_embedding_service():
    """Start the embedding microservice as an independent process."""
    logger.info("Start the embedding microservice as an independent process.")
    embedding_service_path = os.path.abspath(__file__)
    
    # 启动嵌入微服务，使用环境变量中的端口并设置为独立模式
    process = subprocess.Popen(
        [sys.executable, embedding_service_path],
        env={**os.environ, "EMBEDDING_SERVICE_STANDALONE": "1"}
    )
    return process

def shutdown_embedding_service():
    """Terminate the embedding microservice process."""
    global embedding_service_process
    if embedding_service_process:
        logger.info("Terminating the embedding microservice process...")
        try:
            embedding_service_process.terminate()
            embedding_service_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("The embedding microservice process failed to terminate properly; forcing shutdown...")
            embedding_service_process.kill()
        embedding_service_process = None

def signal_handler(sig, frame):
    """Handling termination signal."""
    logger.info(f"Signal received: {sig}, shutting down....")
    shutdown_embedding_service()
    sys.exit(0)

# 注册退出处理程序，确保在应用程序退出时关闭微服务
atexit.register(shutdown_embedding_service)

if __name__ == "__main__":
    import uvicorn
    # 从环境变量获取主机和端口，如果没有设置，则使用默认值
    host = os.environ.get("KBOT_EMBED_HOST", "0.0.0.0")
    port = int(os.environ.get("KBOT_EMBED_PORT", 8001))
    
    # 如果是作为独立进程启动，则注册信号处理器
    if os.environ.get("EMBEDDING_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Started the embedding microservice, listening on {host}:{port}")
    uvicorn.run(app, host=host, port=port)