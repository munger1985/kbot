"""文档解析微服务应用程序。"""

import os
import sys
import signal
import time
import multiprocess.resource_tracker as rt
from datetime import datetime
from typing import Any
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from loguru import logger

from core.config.settings import get_parser_config, get_app_config
from core.logger import LogConfig, LogManager
from core.middleware.log_middleware import log_requests
from microservices.docparser.docling_service import ParserService
from microservices.docparser.parser_schema import ParserParams

# 加载环境变量
load_dotenv()

# 从配置中心获取服务配置
config = get_parser_config()
SERVICE_NAME = config.service_name
SERVICE_VERSION = config.service_version
SERVICE_HOST = config.service_host
SERVICE_PORT = config.service_port

# 获取通用应用配置
app_config = get_app_config()
DEBUG: bool = app_config.debug
LOG_DIR: str = app_config.log.dir
LOG_LEVEL: str = app_config.log.level
LOG_ROTATION: str = app_config.log.rotation
LOG_RETENTION: str = app_config.log.retention

# 全局服务单例
doc_parser_service: ParserService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理服务生命周期：初始化日志和解析单例。"""
    global doc_parser_service

    # 设置服务名称到 app.state（供中间件使用）
    app.state.service_name = SERVICE_NAME

    # --- 核心修复：针对 Python 3.12 的终极补丁 ---
    # 这里的 *args, **kwargs 是关键，能自动兼容 use_blocking_lock 等参数
    def patched_stop(self, *args, **kwargs):
        """修复 Python 3.12 兼容性并兼容不同版本的参数传递"""
        # 注意：这里我们使用 self._lock 而不是直接去读锁的内部属性
        try:
            with self._lock:
                if self._fd is None:
                    return
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None
        except Exception:
            # 在退出阶段，如果锁已经被销毁，直接静默退出
            pass

    # 应用补丁
    rt.ResourceTracker._stop = patched_stop
    # -------------------------------------------

    # 1. 初始化日志系统 (对应 LLM 微服务做法)
    log_conf = LogConfig(
        service_name=SERVICE_NAME, 
        log_dir=LOG_DIR, 
        level=LOG_LEVEL, 
        rotation=LOG_ROTATION, 
        retention=LOG_RETENTION
    )
    LogManager(log_conf).setup()
    
    start_time = time.time()
    logger.info(f"正在启动 [{SERVICE_NAME}] | PID: {os.getpid()} | 时间: {datetime.now()}")
    
    try:
        # 2. 初始化解析服务 (保持原样，直接实例化)
        doc_parser_service = ParserService(
            en_model_path=config.tokenizer.en,
            zh_model_path=config.tokenizer.zh
        )
        logger.info(f"解析服务引擎加载成功 | 耗时: {time.time() - start_time:.2f}s")

    except Exception as e:
        logger.exception(f"初始化阶段发生致命错误: {e}")
        if not DEBUG:
            sys.exit(1)
    
    yield  # --- 服务运行中 ---
    
    # 3. 清理阶段
    logger.info("服务正在关闭...")
    

# 创建应用实例
app = FastAPIOffline(
    title=f"{SERVICE_NAME} API",
    description="基于 Docling 的多格式解析服务，支持 OCR 和动态 VLM 语义增强。",
    version=SERVICE_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None
)

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

def get_parser_service() -> ParserService:
    if doc_parser_service is None:
        raise HTTPException(status_code=503, detail="Service Uninitialized")
    return doc_parser_service

# --- API 端点 ---

@app.get("/health", tags=["System"], summary="健康检查")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/v1/parse/file", tags=["Parser"], summary="文档解析接口")
async def handle_parse_file(
    file: UploadFile = File(...),
    params: ParserParams = Depends(ParserParams.as_form),
    parser_service: ParserService = Depends(get_parser_service)
) -> dict:
    """处理文档解析并返回内容。"""
    
    # 1. 落地上传的原始文件
    temp_input_path = f"/tmp/in_{time.time()}_{file.filename}"
    try:
        content = await file.read()
        with open(temp_input_path, "wb") as f:
            f.write(content)
        
        params.file_path = temp_input_path
        
        # 2. 调用 Service 进行解析
        result = await parser_service.parse_file(params)
        
        # 3. 统一返回内容
        return {
            "status": "success",
            "format": params.output_format,
            "result": result
        }
        
    finally:
        # 清理输入的临时文件
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

# --- 启动逻辑 ---

def signal_handler(sig, frame):
    """捕捉系统信号实现优雅停机。"""
    logger.warning(f"接收到信号 {sig}，正在准备退出...")
    sys.exit(0)

if __name__ == "__main__":
    # 仅在独立模式下注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"LLM 解析适配层已就绪 -> {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(
        app, 
        host=SERVICE_HOST, 
        port=SERVICE_PORT,
        log_config=None  # 完全由 Loguru 接管
    )