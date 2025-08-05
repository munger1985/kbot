"""VLM microservice application.

This module provides a FastAPI application that exposes HTTP endpoints for interacting
with various VLM providers. It supports text VLM.

该模块提供 FastAPI 微服务应用程序，用于公开与各种VLM提供者交互的 HTTP 端点。它支持文本VLM。

"""

import os
import sys
import signal
import subprocess
import time
import atexit
import platform
from datetime import datetime
from PIL import Image
from typing import Any
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
    
from microservices.vlm.vlm_service import VLMService
from core.config import settings
from core.log.logger import setup_logging

# 初始化日志
setup_logging(service_name="VLM")

# 创建VLM服务实例
vlm_service = VLMService()

# 定义 lifespan 上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager. //应用程序生命周期上下文管理器"""
    # 启动事件
    start_time = time.time()
    logger.info(f"Initializing VLM service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...") 
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Process ID: {os.getpid()}")
    
    # 初始化LLM服务
    try:
        await vlm_service.initialize()
        logger.info(f"VLM service started successfully, elapsed time: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"VLM service initialization failed: {e}")
        # 在生产环境中，可能需要在这里退出应用程序
        current_env = os.getenv("KBOT_ENV")
        if current_env == "production":
            sys.exit(1)
    
    yield  # 服务运行期间
    
    # 关闭事件
    logger.info(f"Closing VLM service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    shutdown_start = time.time()
    
    try:
        await vlm_service.shutdown()
        logger.info("VLM service is closed.")
    except Exception as e:
        logger.error(f"VLM service shutdown failed: {e}")
    
    logger.info(f"VLM service closed successfully, elapsed time: {time.time() - shutdown_start:.2f} seconds")
    logger.info(f"Total running time: {time.time() - start_time:.2f} seconds")

# 创建 FastAPI 应用
app = FastAPI(
    title="VLM service",
    description="Provides text VLM services to convert text into vector representations.",
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
class VLMRequest(BaseModel):
    model_unique_name: str = Field(..., description="VLM model unique name")
    text: str = Field(..., description="text to convert to VLM")
    image: str | Image.Image = Field(..., description="image of the text")

class VLMResponse(BaseModel):
    response: str = Field(..., description="VLM model response")


# 依赖项：获取VLM服务实例
def get_vlm_service():
    return vlm_service

@app.get("/health", response_model=dict, tags=["VLM"])
async def health() -> dict[str, Any]:
    """Health check endpoint. //微服务接口健康检查
    Returns:
        Loaded models count. //已加载的模型数量
    """
    
    # 获取已加载的模型信息
    loaded_models = {}
    if vlm_service._initialized and hasattr(vlm_service._model_pool, '_models'):
        loaded_models = vlm_service._model_pool._models
    
    # 返回已加载模型数量
    return {
        "status": "ok",
        "loaded_models_count": len(loaded_models),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/v1/VLMs", response_model=VLMResponse, tags=["VLM"])
async def inference(
    request: VLMRequest,
    vlm_service: VLMService = Depends(get_vlm_service)
    ) -> VLMResponse:
    """
    将文本列表转换为VLM向量
    
    - **model_unique_name**: 要使用的VLM模型的ID
    - **texts**: 要VLM的文本列表
    - **batch_size**: 批处理大小（可选）
    
    返回:
    - **data**: VLM向量列表
    - **usage**: 使用情况信息，包括总令牌数和提示令牌数
    - **model**: 使用的VLM模型ID
    - **object**: 响应对象类型，固定为 "list"
    
    Raises:
    - **HTTPException**: 如果模型ID不存在或VLM失败
    - **RuntimeError**: 如果模型创建失败
    - **Exception**: 如果发生其他错误
    """
    try:
        logger.info(f"Received VLM request: model={request.model_unique_name}")
        
        # 使用VLM服务将文本转换为向量
        vlm = await vlm_service.inference(
            model_unique_name=request.model_unique_name,
            text=request.text,
            image=request.image
        )
        
        return VLMResponse(
            response=vlm
        )
    
    except Exception as e:
        logger.error(f"Error occurred during VLM.: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error occurred during VLM.: {str(e)}")



# 全局变量，用于存储微服务进程
vlm_service_process = None

def start_vlm_service():
    """Start the VLM microservice as an independent process."""
    logger.info("Start the VLM microservice as an independent process.")
    vlm_service_path = os.path.abspath(__file__)
    
    # 启动VLM微服务，使用环境变量中的端口并设置为独立模式
    process = subprocess.Popen(
        [sys.executable, vlm_service_path],
        env={**os.environ, "vlm_service_STANDALONE": "1"}
    )
    return process

def shutdown_vlm_service():
    """Terminate the VLM microservice process."""
    global vlm_service_process
    if vlm_service_process:
        logger.info("Terminating the VLM microservice process...")
        try:
            vlm_service_process.terminate()
            vlm_service_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("The VLM microservice process failed to terminate properly; forcing shutdown...")
            vlm_service_process.kill()
        vlm_service_process = None

def signal_handler(sig, frame):
    """Handling termination signal."""
    logger.info(f"Signal received: {sig}, shutting down....")
    shutdown_vlm_service()
    sys.exit(0)

# 注册退出处理程序，确保在应用程序退出时关闭微服务
atexit.register(shutdown_vlm_service)

if __name__ == "__main__":
    import uvicorn
    # 从环境变量获取主机和端口，如果没有设置，则使用默认值
    host = os.environ.get("KBOT_EMBED_HOST", "0.0.0.0")
    port = int(os.environ.get("KBOT_EMBED_PORT", 8001))
    
    # 如果是作为独立进程启动，则注册信号处理器
    if os.environ.get("vlm_service_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Started the VLM microservice, listening on {host}:{port}")
    uvicorn.run(app, host=host, port=port)