"""Reranker 微服务应用程序。

该模块提供了一个 FastAPI 应用程序，用于暴露与各种 reranker 提供商交互的 HTTP 端点。
它支持文本重排序功能。

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
from ms_core import nacos_manager, ConfigManager, LogManager, LogConfig


# 加载环境变量配置
load_dotenv()

# 从 nacos 获取 reranker 服务配置
config = ConfigManager.get_model_config()
service_name = config.reranker.service_name
service_version = config.reranker.service_version
service_host = config.reranker.service_host
service_port = config.reranker.service_port

# 创建 reranker 服务实例
reranker_service = RerankerService()

# 定义 lifespan 上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期上下文管理器"""
    # 通过 nacos_manager 获取 logger 配置
    log_config = ConfigManager.get_app_config()
    log_dir = log_config.kbot.log.dir
    log_level = log_config.kbot.log.level
    rotation = log_config.kbot.log.rotation
    retention = log_config.kbot.log.retention
    
    # 初始化日志
    conf = LogConfig(service_name=service_name, log_dir=log_dir, level=log_level, rotation=rotation, retention=retention)
    LogManager(conf).setup()
    
    # 启动事件
    start_time = time.time()
    logger.info(f"正在初始化 reranker 服务，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    logger.info(f"进程 ID: {os.getpid()}")

    
    # 初始化 reranker 服务
    try:
        await reranker_service.initialize()
        await reranker_service.warmup()
        logger.info(f"Reranker 服务启动成功，耗时: {time.time() - start_time:.2f} 秒")

        # 注册服务到 Nacos
        nacos_manager.register_service(service_name=service_name, service_host=service_host, service_port=service_port)
        logger.info("Reranker 服务已注册到 Nacos")

    except Exception as e:
        logger.error(f"Reranker 服务初始化失败: {e}")
        # 在生产环境中，可能需要在这里退出应用程序
        current_env = os.getenv("NACOS_GROUP", "dev")
        if current_env == "prod":
            sys.exit(1)
    
    yield  # 服务运行期间
    
    # 关闭事件
    logger.info(f"正在关闭 reranker 服务，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    shutdown_start = time.time()
    
    try:
        await reranker_service.shutdown()
        logger.info("Reranker 服务已关闭")
    except Exception as e:
        logger.error(f"Reranker 服务关闭失败: {e}")
    
    logger.info(f"Reranker 服务关闭成功，耗时: {time.time() - shutdown_start:.2f} 秒")
    logger.info(f"总运行时间: {time.time() - start_time:.2f} 秒")

# 创建 FastAPI 应用
app = FastAPI(
    title="Reranker 服务",
    description="提供文本重排序服务",
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
    model_unique_name: str = Field(..., description="Reranker 模型唯一名称")
    query: str = Field(..., description="查询文本")
    documents: list[str] = Field(..., description="需要重排序的文档列表")
    top_k: int | None = Field(10, description="返回的顶部文档数量（None 表示返回所有）")

class ToggleModelRequest(BaseModel):
    """启用或禁用模型请求表单。"""
    model_unique_name: str = Field(..., description="模型唯一标识符")
    operation: str = Field(..., description="操作类型，'load' 或 'unload'")

# 定义响应模型
class RerankerResponse(BaseModel):
    rerankers: list[dict[str, Any]] = Field(..., description="重排序后的文档列表")

# 依赖项：获取 reranker 服务实例
def get_reranker_service():
    return reranker_service

@app.get("/health", response_model=dict, tags=["Reranker"])
async def health() -> dict[str, Any]:
    """健康检查端点
    
    Returns:
        已加载的模型数量
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

@app.post("/load", response_model=dict, tags=["Reranker"])
async def load_model(request: ToggleModelRequest) -> dict:
    """通过模型ID加载模型到内存中。
    
    Args:
        request: 启用或禁用模型请求表单，包含模型唯一名称和操作类型
        
    Returns:
        dict: 包含操作状态和模型ID的响应数据
        
    Raises:
        HTTPException: 当模型加载失败时抛出500错误
    """
    try:
        if request.operation == "load":
            logger.info(f"接收到指令：加载模型 {request.model_unique_name}")
            success = await reranker_service.load_model(request.model_unique_name)
        else:
            logger.info(f"接收到指令：卸载模型 {request.model_unique_name}")
            success = await reranker_service.unload_model(request.model_unique_name)
        if not success:
            raise HTTPException(status_code=500, detail=f"模型 {request.model_unique_name} 操作失败")
        return {"status": "success", "model_unique_name": request.model_unique_name}
    except Exception as e:
        logger.exception(f"操作模型 {request.model_unique_name} 时发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
      
@app.post("/v1/rerank", response_model=RerankerResponse, tags=["Reranker"])
async def rerank_texts(
    request: RerankerRequest,
    reranker_service: RerankerService = Depends(get_reranker_service)
    ) -> RerankerResponse:
    """
    将文本列表进行重排序
    
    - **model_unique_name**: 用于重排序的模型唯一名称
    - **query**: 查询文本
    - **documents**: 需要重排序的文档列表
    - **top_k**: 返回的顶部文档数量（None 表示返回所有）
    """

    try:
        logger.info(f"收到重排序请求，模型：{request.model_unique_name}, 查询：{request.query}, 文档数量：{len(request.documents)}, top_k：{request.top_k}")
        
        # 使用重排序服务将文本列表进行重排序
        rerankers = await reranker_service.rerank(
            model_id=request.model_unique_name,
            query=request.query,
            documents=request.documents,
            top_k=request.top_k
        )
        
        logger.info(f"重排序完成，重排序结果数量：{len(rerankers)}, top_k：{request.top_k}")
        return RerankerResponse(rerankers=rerankers)
    
    except Exception as e:
        logger.error(f"重排序过程中发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重排序过程中发生错误: {str(e)}")



# 全局变量，用于存储微服务进程
reranker_service_process = None

def start_reranker_service():
    """启动 reranker 微服务作为独立进程"""

    # 启动 reranker 微服务，使用环境变量中的端口并设置为独立模式
    try:
        logger.info("正在启动 reranker 微服务作为独立进程")
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
            raise RuntimeError(f"启动 reranker 服务失败: {stderr}")
            
        logger.success(f"Reranker 服务启动成功，进程 ID: {process.pid}")
        return process
        
    except Exception as e:
        logger.exception(f"启动 reranker 服务时出错: {str(e)}")
        raise

def shutdown_reranker_service():
    """终止 reranker 微服务进程"""
    global reranker_service_process
    if reranker_service_process:
        logger.info("正在终止 reranker 微服务进程...")
        try:
            reranker_service_process.terminate()
            reranker_service_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Reranker 微服务进程未能正常终止; 强制关闭中...")
            reranker_service_process.kill()
        reranker_service_process = None


def signal_handler(sig, frame):
    """处理终止信号"""
    logger.info(f"收到信号: {sig}, 正在关闭...")
    shutdown_reranker_service()
    sys.exit(0)

# 注册退出处理程序，确保在应用程序退出时关闭微服务
atexit.register(shutdown_reranker_service)

if __name__ == "__main__":
    # 如果是作为独立进程启动，则注册信号处理器
    if os.environ.get("RERANKER_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"已启动 reranker 微服务，监听地址: {service_host}:{service_port}")
    uvicorn.run(app, host=service_host, port=service_port)