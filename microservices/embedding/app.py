"""嵌入微服务应用程序。

该模块提供 FastAPI 微服务应用程序，用于公开与各种嵌入提供者交互的 HTTP 端点。支持文本嵌入功能。
"""

import os
import sys
import signal
import subprocess
import time
import atexit
import uvicorn
from dotenv import load_dotenv
from datetime import datetime
from typing import Any
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from ms_core import LogConfig, LogManager, nacos_manager, ConfigManager
from embed_service import EmbeddingService
from model.base import EmbeddingResponse


# 加载环境变量配置
load_dotenv()

config = ConfigManager.get_model_config()
service_name = config.embed.service_name
service_version = config.embed.service_version
service_host = config.embed.service_host
service_port = config.embed.service_port


# 创建嵌入服务实例
embedding_service = EmbeddingService()

# 定义生命周期上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期上下文管理器。"""

    log_config = ConfigManager.get_app_config()
    log_dir = log_config.kbot.log.dir
    log_level = log_config.kbot.log.level
    rotation = log_config.kbot.log.rotation
    retention = log_config.kbot.log.retention
        
    # 初始化日志配置
    conf = LogConfig(service_name=service_name, log_dir=log_dir, level=log_level, rotation=rotation, retention=retention)
    LogManager(conf).setup()
    
    # 启动事件处理
    start_time = time.time()
    logger.info(f"正在初始化嵌入服务，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 
    logger.info(f"进程ID: {os.getpid()}")
    
    # 初始化微服务
    try:
        await embedding_service.initialize()
        await embedding_service.warmup()
        logger.info(f"嵌入服务启动成功，耗时: {time.time() - start_time:.2f} 秒")

        # 注册服务到 Nacos
        nacos_manager.register_service(service_name=service_name, service_host=service_host, service_port=service_port)
        logger.info("嵌入服务已注册到 Nacos。")

    except Exception as e:
        logger.exception(f"嵌入服务初始化失败: {e}")
        # 生产环境初始化失败时退出应用
        current_env = os.getenv("NACOS_GROUP", "dev")
        if current_env == "prod":
            sys.exit(1)
    
    yield  # 服务运行期间
    
    # 关闭事件处理
    logger.info(f"正在关闭嵌入服务，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    shutdown_start = time.time()
    
    try:
        await embedding_service.shutdown()
        logger.info("嵌入服务已关闭。")
    except Exception as e:
        logger.exception(f"嵌入服务关闭失败: {e}")
    
    logger.info(f"嵌入服务关闭完成，耗时: {time.time() - shutdown_start:.2f} 秒")
    logger.info(f"总运行时间: {time.time() - start_time:.2f} 秒")

# 创建 FastAPI 应用实例
app = FastAPI(
    title=service_name,
    description="提供文本嵌入服务，将文本转换为向量表示。",
    version=service_version,
    lifespan=lifespan,
)

# 添加 CORS 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为特定源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义嵌入请求模型
class EmbeddingRequest(BaseModel):
    """嵌入请求参数模型。"""
    
    model_id: int = Field(..., description="模型唯一标识符")
    texts: list[str] = Field(..., description="待嵌入的文本列表")
    batch_size: int | None = Field(32, description="批处理大小")
    is_query: bool = Field(True, description="是否为查询文本")

class ToggleModelRequest(BaseModel):
    """启用或禁用模型请求表单。"""
    model_id: int = Field(..., description="模型唯一标识符")
    operation: str = Field(..., description="操作类型，'load' 或 'unload'")

def get_embed_service():
    """获取嵌入服务实例依赖项。
    
    Returns:
        EmbeddingService: 嵌入服务实例
    """
    return embedding_service

@app.get("/health", response_model=dict, tags=["Embedding"])
async def health() -> dict[str, Any]:
    """微服务健康检查接口。
    
    Returns:
        dict: 包含服务状态、已加载模型数量和时间戳的响应数据
    """
    
    # 获取已加载的模型信息
    loaded_models = {}
    if embedding_service._initialized and hasattr(embedding_service._model_pool, '_models'):
        loaded_models = embedding_service._model_pool._models
    
    return {
        "status": "ok",
        "loaded_models_count": len(loaded_models),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/load", response_model=dict, tags=["Embedding"])
async def load_model(request: ToggleModelRequest) -> dict:
    """通过模型ID加载模型到内存中。
    
    Args:
        request: 启用或禁用模型请求表单，包含模型唯一名称和操作类型
        
    Returns:
        dict: 包含操作状态和模型ID的响应数据
        
    Raises:
        HTTPException: 当模型加载失败时抛出500错误
    """
    model_name = embedding_service._model_pool._model_names.get(request.model_id, str(request.model_id))
    try:
        if request.operation == "load":
            logger.info(f"接收到指令：加载模型 {model_name}")
            success = await embedding_service.load_model(request.model_id)
        else:
            logger.info(f"接收到指令：卸载模型 {model_name}")
            success = await embedding_service.unload_model(request.model_id)
        if not success:
            raise HTTPException(status_code=500, detail=f"模型 {model_name} 操作失败")
        return {"status": "success", "model_name": model_name}
    except Exception as e:
        logger.exception(f"操作模型 {model_name} 时发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    
@app.post("/v1/embeddings", response_model=EmbeddingResponse, tags=["Embedding"])
async def embed_texts(
    request: EmbeddingRequest,
    embed_service: EmbeddingService = Depends(get_embed_service)
    ) -> EmbeddingResponse:
    """将文本列表转换为嵌入向量。
    
    Args:
        request: 嵌入请求参数，包含模型名称、文本列表和批处理大小
        embed_service: 嵌入服务实例依赖注入
        
    Returns:
        EmbeddingResponse: 嵌入响应数据，包含向量数据和使用情况信息
        
    Raises:
        HTTPException: 当嵌入处理过程中发生错误时抛出500错误
    """
    model_name = embedding_service._model_pool._model_names.get(request.model_id, str(request.model_id))
    try:
        logger.info(f"收到嵌入请求: 模型 {model_name}, 文本数量：{len(request.texts)}")
        
        # 使用嵌入服务将文本转换为向量
        embeddings = await embed_service.embed_texts(
            model_id=request.model_id,
            texts=request.texts,
            batch_size=request.batch_size, # type: ignore
            is_query=request.is_query
        )
        
        return embeddings
    
    except Exception as e:
        logger.exception(f"嵌入处理过程中发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"嵌入处理过程中发生错误: {str(e)}")


# 全局变量，用于存储微服务进程
embedding_service_process = None

def start_embedding_service():
    """启动嵌入微服务作为独立进程。
    
    Returns:
        subprocess.Popen: 微服务进程对象
        
    Raises:
        RuntimeError: 当进程启动失败时抛出运行时错误
    """
    try:
        logger.info("正在启动嵌入微服务独立进程...")
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
            raise RuntimeError(f"启动嵌入服务失败: {stderr}")
            
        logger.success(f"嵌入服务启动成功，进程ID: {process.pid}")
        return process
        
    except Exception as e:
        logger.exception(f"启动嵌入服务时发生错误: {str(e)}")
        raise

def shutdown_embedding_service():
    """终止嵌入微服务进程。"""
    global embedding_service_process
    if embedding_service_process:
        logger.info("正在终止嵌入微服务进程...")
        try:
            embedding_service_process.terminate()
            embedding_service_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("嵌入微服务进程终止超时，强制关闭中...")
            embedding_service_process.kill()
        embedding_service_process = None

def signal_handler(sig, frame):
    """处理终止信号。
    
    Args:
        sig: 信号类型
        frame: 当前堆栈帧
    """
    logger.info(f"收到信号: {sig}，正在关闭服务...")
    shutdown_embedding_service()
    sys.exit(0)

# 注册退出处理程序，确保应用程序退出时关闭微服务
atexit.register(shutdown_embedding_service)

if __name__ == "__main__":
    # 如果是作为独立进程启动，则注册信号处理器
    if os.environ.get("EMBEDDING_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"嵌入微服务已启动，监听地址: {service_host}:{service_port}")
    uvicorn.run(app, host=service_host, port=service_port)