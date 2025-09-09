"""同义词微服务应用程序。

该模块提供了一个基于FastAPI的微服务，通过HTTP端点与各种同义词提供者进行交互。
主要功能包括单词同义词查询、缓存管理和服务健康状态监控等。
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
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from synonym_service import FastTextSynonymExpander
from ms_core import nacos_manager, ConfigManager, LogManager, LogConfig


# 加载环境变量配置
load_dotenv()

# 从nacos获取同义词服务配置
config = ConfigManager.get_model_config()
service_name = config.synonym.service_name
service_version = config.synonym.service_version
service_host = config.synonym.service_host
service_port = config.synonym.service_port

# 创建同义词服务实例
synonym_service = FastTextSynonymExpander()

# 定义应用生命周期上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期上下文管理器"""
    # 通过nacos_manager获取日志配置
    log_config = ConfigManager.get_app_config()
    log_dir = log_config.kbot.log.dir
    log_level = log_config.kbot.log.level
    rotation = log_config.kbot.log.rotation
    retention = log_config.kbot.log.retention
    
    # 初始化日志系统
    conf = LogConfig(service_name=service_name, log_dir=log_dir, level=log_level, rotation=rotation, retention=retention)
    LogManager(conf).setup()
    
    # 启动事件
    start_time = time.time()
    logger.info(f"正在初始化同义词服务，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    logger.info(f"进程ID: {os.getpid()}")

    
    # 初始化同义词服务
    try:
        # 加载模型和词向量
        await synonym_service.load_model()
        # 预热缓存
        await synonym_service.preload_cache()
        logger.info(f"同义词服务启动成功，耗时: {time.time() - start_time:.2f} 秒")

        # 注册服务到Nacos
        nacos_manager.register_service(service_name=service_name, service_host=service_host, service_port=service_port)
        logger.info("同义词服务已注册到Nacos.")

    except Exception as e:
        logger.error(f"同义词服务初始化失败: {e}")
        # 在生产环境中，可能需要在这里退出应用程序
        current_env = os.getenv("NACOS_GROUP", "dev")
        if current_env == "prod":
            sys.exit(1)
    
    yield  # 服务运行期间
    
    # 关闭事件
    logger.info(f"正在关闭同义词服务，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    shutdown_start = time.time()
    
    try:
        await synonym_service.shutdown()
        logger.info("同义词服务已关闭.")
    except Exception as e:
        logger.error(f"同义词服务关闭失败: {e}")
    
    logger.info(f"同义词服务关闭完成，耗时: {time.time() - shutdown_start:.2f} 秒")
    logger.info(f"总运行时间: {time.time() - start_time:.2f} 秒")

# 创建FastAPI应用
app = FastAPI(
    title="同义词服务",
    description="提供文本同义词查询服务。",
    version=service_version,
    lifespan=lifespan,
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求模型
class SynonymRequest(BaseModel):
    words: list[str] = Field(..., description="需要查询同义词的单词列表")
    top_k: int | None = Field(3, description="返回的同义词数量（None表示返回所有）")
    threshold: float | None = Field(0.6, description="同义词过滤阈值")


# 定义响应模型
class SynonymResponse(BaseModel):
    synonyms: dict[str, list[str]] = Field(..., description="每个单词的同义词字典")

# 依赖项：获取同义词服务实例
def get_synonym_service():
    return synonym_service

@app.get("/cache_info", response_model=dict, tags=["synonym"])
async def get_cache_info() -> dict:
    """获取缓存统计信息接口"""
    return await synonym_service.get_cache_info()

@app.post("/synonym", response_model=SynonymResponse, tags=["synonym"])
async def synonym(
    request: SynonymRequest,
    synonym_service: FastTextSynonymExpander = Depends(get_synonym_service)
    ) -> SynonymResponse:
    """
    为输入的文本生成同义词
    
    参数:
        request (SynonymRequest): 请求对象，包含要查询的单词列表和top_k参数、threshold参数
        synonym_service (FastTextSynonymExpander): 同义词服务实例
    返回:
        SynonymResponse: 同义词响应对象，包含每个单词的同义词列表
    """

    try:
        logger.info(f"收到同义词查询请求: {request}")
        synonym_dict = {}
        # 为每一个输入的词生成同义词
        for word in request.words:
            synonyms = synonym_service.get_synonym(word, top_k=request.top_k, threshold=request.threshold)
            synonym_dict[word] = synonyms
            logger.info(f"为单词 '{word}' 生成同义词: {synonyms}")
        
        logger.info(f"同义词查询完成: 单词数量={len(request.words)}, top_k={request.top_k}, threshold={request.threshold}")
        return SynonymResponse(synonyms=synonym_dict)
    
    except Exception as e:
        logger.error(f"同义词查询过程中发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"同义词查询过程中发生错误: {str(e)}")



# 全局变量，用于存储微服务进程
synonym_service_process = None

def start_synonym_service():
    """启动同义词微服务作为独立进程"""

    # 启动同义词微服务，使用环境变量中的端口并设置为独立模式
    try:
        logger.info("正在启动同义词微服务作为独立进程")
        synonym_service_path = os.path.abspath(__file__)
        
        process = subprocess.Popen(
            [sys.executable, synonym_service_path],
            env={**os.environ, "SYNONYM_SERVICE_STANDALONE": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 检查进程是否成功启动
        if process.poll() is not None:
            stderr = process.stderr.read().decode('utf-8') if process.stderr else ""
            raise RuntimeError(f"启动同义词服务失败: {stderr}")
            
        logger.success(f"同义词服务启动成功，进程ID: {process.pid}")
        return process
        
    except Exception as e:
        logger.exception(f"启动同义词服务时发生错误: {str(e)}")
        raise

def shutdown_synonym_service():
    """终止同义词微服务进程"""
    global synonym_service_process
    if synonym_service_process:
        logger.info("正在终止同义词微服务进程...")
        try:
            synonym_service_process.terminate()
            synonym_service_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("同义词微服务进程未能正常终止，正在强制关闭...")
            synonym_service_process.kill()
        synonym_service_process = None


def signal_handler(sig, frame):
    """处理终止信号"""
    logger.info(f"收到信号: {sig}, 正在关闭服务...")
    shutdown_synonym_service()
    sys.exit(0)

# 注册退出处理程序，确保在应用程序退出时关闭微服务
atexit.register(shutdown_synonym_service)

if __name__ == "__main__":
    # 如果是作为独立进程启动，则注册信号处理器
    if os.environ.get("SYNONYM_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"同义词微服务已启动，监听地址: {service_host}:{service_port}")
    uvicorn.run(app, host=service_host, port=service_port)