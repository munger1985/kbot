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
import configparser
import uvicorn
import socket
from dotenv import load_dotenv
from datetime import datetime
from typing import Any
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from nacos import NacosClient
from nacos_manager import nacos_manager # type: ignore
from logger_manager import LogManager, LogConfig # type: ignore
from embed_service import EmbeddingService
from model.base import EmbeddingResponse


# 加载环境变量配置
load_dotenv()

nacos_addr = os.getenv("NACOS_SERVER_ADDR") # Nacos服务器地址
nacos_namespace = "public" # Nacos命名空间
nacos_group = os.getenv("NACOS_GROUP") or "DEV_GROUP" # Nacos分组
nacos_username = os.getenv("NACOS_USERNAME") # Nacos账号名称
nacos_password = os.getenv("NACOS_PASSWORD") # Nacos账号密码

try:
    # 从 nacos 获取 embedding 服务配置
    config_parser = configparser.ConfigParser()
    nacos_config = nacos_manager.get_config("embedding", nacos_group)
    config_parser.read_string(f"[{nacos_group}]\n{nacos_config}")
    service_name = config_parser.get(nacos_group, "service_name") or "embedding-service" # 全局微服务名称
    service_version = config_parser.get(nacos_group, "service_version") or "1.0.0" # 微服务版本
    service_host = config_parser.get(nacos_group, "service_host") or "0.0.0.0" # 微服务地址
    service_port = int(config_parser.get(nacos_group, "service_port")) or 9201 # 微服务通信端口
except Exception as e:
    # 如果从 nacos 获取 embedding 服务配置失败，则使用默认配置
    logger.warning("Failed to get embedding service config from nacos: {}".format(e))
    service_name = "embedding-service"
    service_version = "1.0.0"
    service_host = "0.0.0.0"
    service_port = 9201



# Nacos 服务注册
def register_service():
    client = NacosClient(
        server_addresses=nacos_addr,
        namespace=nacos_namespace
        # username='nacos',
        # password='nacos'
        )
    client.add_naming_instance(
        service_name=service_name,
        group_name=nacos_group,
        ip=service_host,
        port=service_port,
        ephemeral=True,
        healthy=True
    )
    # nacos 心跳发送器
    while True:
        if signal.SIGINT or signal.SIGTERM:
            break
        try:
            # 健康检查：检测服务端口是否存活
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((service_host, service_port))
            is_healthy = (result == 0)
            sock.close()

            # 更新实例健康状态
            client.send_heartbeat(
                service_name=service_name,
                group_name=nacos_group,
                ip=service_host,
                port=service_port
            )
            
            logger.info(f"Heartbeat sent, healthy: {is_healthy}")
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            break
        
        time.sleep(10)  # 间隔需小于Nacos心跳超时时间（默认15秒）


# 创建embedding服务实例
embedding_service = EmbeddingService()

# 定义 lifespan 上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager. //应用程序生命周期上下文管理器"""
    # 通过 nacos_manager 获取logger配置
    try:
        log_config = nacos_manager.get_config("logger", nacos_group)
        config_parser.read_string(f"[{nacos_group}]\n{log_config}")
        log_dir = config_parser.get(nacos_group, "dir") or "logs/"
        log_level = config_parser.get(nacos_group, "level") or "DEBUG"
        rotation = config_parser.get(nacos_group, "rotation") or "10 MB"
        retention = config_parser.get(nacos_group, "retention") or "20 days"
        
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
    logger.info(f"Initializing embedding service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...") 
    logger.info(f"Process ID: {os.getpid()}")
    
    
     
    # 初始化微服务
    try:
        await embedding_service.initialize()
        logger.info(f"Embedding service started successfully, elapsed time: {time.time() - start_time:.2f} seconds")

        # 注册服务到 Nacos
        register_service()
        logger.info("Embedding service registered to Nacos.")

    except Exception as e:
        logger.exception(f"Embedding service initialization failed: {e}")
        # 在生产环境中，可能需要在这里退出应用程序
        current_env = nacos_namespace
        if current_env == "prod":
            sys.exit(1)
    
    yield  # 服务运行期间
    
    # 关闭事件
    logger.info(f"Closing embedding service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    shutdown_start = time.time()
    
    try:
        await embedding_service.shutdown()
        logger.info("Embedding service is closed.")
    except Exception as e:
        logger.exception(f"Embedding service shutdown failed: {e}")
    
    logger.info(f"Embedding service closed successfully, elapsed time: {time.time() - shutdown_start:.2f} seconds")
    logger.info(f"Total running time: {time.time() - start_time:.2f} seconds")

# 创建 FastAPI 应用
app = FastAPI(
    title=service_name,
    description="Provides text embedding services to convert text into vector representations.",
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
class EmbeddingRequest(BaseModel):
    model_unique_name: str = Field(..., description="Embedding model unique name")
    texts: list[str] = Field(..., description="list of texts to be embedded.")
    batch_size: int | None = Field(32, description="Batch size")


# 依赖项：获取嵌入服务实例
def get_embed_service():
    return embedding_service

@app.get("/health", response_model=dict, tags=["Embedding"])
async def health() -> dict[str, Any]:
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
        logger.exception(f"Error occurred during embedding.: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error occurred during embedding.: {str(e)}")


# 全局变量，用于存储微服务进程
embedding_service_process = None

def start_embedding_service():
    """Start the embedding microservice as an independent process."""
    try:
        logger.info("Starting embedding microservice as independent process...")
        embedding_service_path = os.path.abspath(__file__)
        
        process = subprocess.Popen(
            [sys.executable, embedding_service_path],
            env={**os.environ, "EMBEDDING_SERVICE_STANDALONE": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 检查进程是否成功启动
        if process.poll() is not None:
            stderr = process.stderr.read().decode('utf-8') if process.stderr else ""
            raise RuntimeError(f"Failed to start embedding service: {stderr}")
            
        logger.success(f"Embedding service started successfully with PID {process.pid}")
        return process
        
    except Exception as e:
        logger.exception(f"Error starting embedding service: {str(e)}")
        raise

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
    # 如果是作为独立进程启动，则注册信号处理器
    if os.environ.get("EMBEDDING_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Started the embedding microservice, listening on {service_host}:{service_port}")
    uvicorn.run(app, host=service_host, port=service_port)