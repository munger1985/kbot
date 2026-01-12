"""嵌入微服务应用程序。

本模块提供基于 FastAPI 的微服务，用于公开与各种嵌入提供者（Embedding Providers）交互的 HTTP 端点。
支持文本向量化、相似度计算以及模型的动态加载与卸载。
"""

import os
import sys
import signal
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

from core.config.settings import get_embed_config, get_app_config
from core.logger import LogConfig, LogManager
from core.middleware.log_middleware import log_requests
from microservices.embedding.embed_service import EmbeddingService
from microservices.embedding.schema import (
    EmbeddingRequest, SimilarityRequest, ToggleModelRequest
)
from microservices.embedding.model import EmbeddingResponse

# 加载环境变量
load_dotenv()

# 从配置中提取服务参数
config = get_embed_config()
SERVICE_NAME = config.service_name
SERVICE_VERSION = config.service_version
SERVICE_HOST = config.service_host
SERVICE_PORT = config.service_port

# 日志参数
app_config = get_app_config()
LOG_DIR = app_config.log.dir
LOG_LEVEL = app_config.log.level
LOG_ROTATION = app_config.log.rotation
LOG_RETENTION = app_config.log.retention
DEBUG_MODE = app_config.debug

# 实例化嵌入服务单例
embedding_service = EmbeddingService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用程序的生命周期。

    在服务启动时：初始化日志系统、加载嵌入模型资源并执行预热。
    在服务关闭时：释放模型占用的显存或内存资源。

    Args:
        app: FastAPI 应用程序实例。
    """
    # 设置服务名称到 app.state（供中间件使用）
    app.state.service_name = SERVICE_NAME

    # 1. 初始化日志系统
    log_conf = LogConfig(
        service_name=SERVICE_NAME,
        log_dir=LOG_DIR,
        level=LOG_LEVEL,
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION
    )
    LogManager(log_conf).setup()

    # 2. 启动初始化流程
    start_time = time.time()
    logger.info(f"正在初始化嵌入服务 | 进程ID: {os.getpid()} | 时间: {datetime.now()}")

    try:
        await embedding_service.initialize()
        await embedding_service.warmup()
        logger.info(f"嵌入服务启动成功 | 耗时: {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.exception(f"嵌入服务初始化失败: {e}")
        # 在生产环境下，核心服务初始化失败应强制退出
        if not DEBUG_MODE:
            sys.exit(1)

    yield  # --- 服务运行中 ---

    # 3. 关闭清理流程
    logger.info("正在关闭嵌入服务并释放资源...")
    shutdown_start = time.time()
    try:
        await embedding_service.shutdown()
        logger.info(f"资源释放完成 | 销毁耗时: {time.time() - shutdown_start:.2f}s")
    except Exception as e:
        logger.error(f"释放资源时发生异常: {e}")


# 创建 FastAPI 应用程序实例
app = FastAPIOffline(
    title="Embedding 微服务",
    description="提供高性能文本嵌入（Text Embedding）与向量相似度计算服务。",
    version=SERVICE_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if DEBUG_MODE else None,
    redoc_url="/redoc" if DEBUG_MODE else None
)

# CORS 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求日志中间件
app.middleware("http")(log_requests)


# --- 依赖注入 ---

def get_embed_service() -> EmbeddingService:
    """提供嵌入服务单例的依赖项注入。

    Returns:
        EmbeddingService: 全局嵌入服务实例。
    """
    return embedding_service


# --- API 端点定义 ---

@app.get("/health", response_model=dict[str, Any], tags=["System"], summary="健康检查接口")
async def health_check() -> dict[str, Any]:
    """获取微服务的运行状态和已加载模型信息。

    Returns:
        包含服务状态、模型计数和时间戳的字典。
    """
    loaded_models_count = 0
    if embedding_service._initialized and hasattr(embedding_service._model_pool, '_models'):
        loaded_models_count = len(embedding_service._model_pool._models)

    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "loaded_models_count": loaded_models_count,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/load", response_model=dict[str, Any], tags=["Management"], summary="动态管理模型状态")
async def handle_toggle_model(request: ToggleModelRequest) -> dict[str, Any]:
    """根据指令加载或卸载特定的嵌入模型。

    Args:
        request: 包含模型名称和操作类型（load/unload）的请求对象。

    Returns:
        操作结果状态。

    Raises:
        HTTPException: 当操作失败或模型不存在时抛出 500 错误。
    """
    try:
        if request.operation == "load":
            logger.info(f"执行模型加载任务: {request.model_id}")
            success = await embedding_service.load_model(request.model_id)
        else:
            logger.info(f"执行模型卸载任务: {request.model_id}")
            success = await embedding_service.unload_model(request.model_id)

        if not success:
            raise ValueError(f"模型 {request.model_id} 执行 {request.operation} 失败")

        return {"status": "success", "model_id": request.model_id, "operation": request.operation}

    except Exception as e:
        logger.error(f"模型管理操作异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings", response_model=EmbeddingResponse, tags=["AI Service"], summary="文本向量化接口")
async def handle_embed_texts(
    request: EmbeddingRequest,
    embed_service: EmbeddingService = Depends(get_embed_service)
) -> EmbeddingResponse:
    """将输入的文本列表转换为向量嵌入。

    Args:
        request: 嵌入请求参数，包括模型名、文本列表、批处理大小等。
        embed_service: 注入的嵌入服务实例。

    Returns:
        包含嵌入向量、索引和 Token 使用情况的响应对象。

    Raises:
        HTTPException: 处理过程中发生任何逻辑错误时抛出 500 错误。
    """
    try:
        logger.info(f"处理嵌入请求 | 模型: {request.model_id} | 文本量: {len(request.texts)}")
        return await embed_service.embed_texts(
            model_id=request.model_id,
            texts=request.texts,
            batch_size=request.batch_size or 2,
            is_query=request.is_query
        )
    except Exception as e:
        logger.exception(f"文本向量化失败: {e}")
        raise HTTPException(status_code=500, detail=f"嵌入处理异常: {str(e)}")


@app.post("/v1/similarity", response_model=dict[str, Any], tags=["AI Service"], summary="计算文本相似度接口")
async def handle_compute_similarity(
    request: SimilarityRequest,
    embed_service: EmbeddingService = Depends(get_embed_service)
) -> dict[str, Any]:
    """计算两个指定文本之间的相似度得分。

    Args:
        request: 包含模型名、文本对以及计算方法（cosine/dot）的请求对象。
        embed_service: 注入的嵌入服务实例。

    Returns:
        包含相似度分数的字典。

    Raises:
        HTTPException: 计算过程中发生异常时抛出 500 错误。
    """
    try:
        logger.info(f"处理相似度请求 | 模型: {request.model_id} | 方法: {request.method}")
        score = await embed_service.compute_similarity(
            model_id=request.model_id,
            text1=request.text1,
            text2=request.text2,
            method=request.method
        )
        return {"similarity": score, "method": request.method}
    except Exception as e:
        logger.exception(f"相似度计算失败: {e}")
        raise HTTPException(status_code=500, detail=f"计算过程异常: {str(e)}")


# --- 进程信号管理 ---

def signal_handler(sig: int, frame: Any):
    """处理操作系统发送的终止信号，确保优雅停机。

    Args:
        sig: 信号编号。
        frame: 当前堆栈帧。
    """
    logger.warning(f"接收到系统信号: {sig}，准备关闭服务...")
    # sys.exit(0) 会触发 atexit 和 lifespan 的清理逻辑
    sys.exit(0)


# 注册退出钩子
atexit.register(lambda: logger.info("微服务进程已安全退出"))

if __name__ == "__main__":
    # 注册信号监听
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"启动嵌入微服务，监听地址: {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(
        app,
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        log_config=None  # 使用 loguru 接管所有日志
    )