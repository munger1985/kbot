"""主程序启动入口。

本模块负责初始化 FastAPI 应用、加载全局配置、管理文件解析服务的生命周期，
并启动 Uvicorn 服务器。
"""
# --- 修复：屏蔽 jieba 警告 ---
# jieba 内部使用了旧版本的 setuptools 接口导致日志第一行显示的 UserWarning: pkg_resources is deprecated
import warnings
# 屏蔽 jieba 引起的 pkg_resources 弃用警告
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")
# ----------------------------

import asyncio
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from loguru import logger

from api.routers import router
from core.config.settings import get_app_config
from core.logger import LogConfig, LogManager
from core.middleware.log_middleware import log_requests

# --- 环境初始化 ---
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用程序的生命周期。

    在应用启动时初始化文件解析管理器并启动服务；
    在应用关闭时确保资源安全回收。

    Args:
        app: FastAPI 实例。
    """
    # 设置服务名称到 app.state（供中间件使用）
    app.state.service_name = get_app_config().service_name

    # 启动阶段
    logger.info("应用正在启动，执行初始化任务...")

    yield  # 应用运行中

    # 关闭阶段
    logger.info("应用正在关闭，执行清理任务...")
    


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用程序。

    读取全局配置，初始化日志系统，挂载路由及中间件。

    Returns:
        FastAPI: 配置完成的应用实例。
    """
    try:
        app_config = get_app_config()

        # 1. 初始化日志中心
        log_conf = LogConfig(
            service_name=app_config.service_name,
            log_dir=app_config.log.dir,
            level=app_config.log.level,
            rotation=app_config.log.rotation,
            retention=app_config.log.retention,
        )
        LogManager(log_conf).setup()

        logger.debug("正在配置 FastAPI 实例...")

        # 2. 实例化应用 (支持离线文档)
        app = FastAPIOffline(
            title=app_config.title,
            description=app_config.description,
            version=app_config.service_version,
            debug=app_config.debug,
            lifespan=lifespan,  # 注入生命周期管理器
            docs_url="/docs" if app_config.debug else None,
            redoc_url="/redoc" if app_config.debug else None,
        )

        # 3. 中间件配置
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 4. 请求日志中间件
        app.middleware("http")(log_requests)

        # 5. 路由注册
        app.include_router(router)

        return app

    except Exception as e:
        logger.critical(f"应用创建失败: {e}")
        raise


def handle_exit_signal(sig, frame):
    """处理系统退出信号 (SIGINT, SIGTERM)。

    Args:
        sig: 信号编号。
        frame: 当前堆栈帧。
    """
    logger.warning(f"接收到信号 {sig}，准备强制退出...")
    # 注意：在 lifespan 模式下，uvicorn 会优雅处理正常关闭，
    # 这里的 sys.exit 主要是针对双击 Ctrl+C 等强制情况
    sys.exit(0)


async def start_server():
    """配置并运行 Uvicorn 异步服务器。"""
    app = create_app()

    SERVICE_HOST = get_app_config().service_host
    SERVICE_PORT = get_app_config().service_port

    logger.info(f"服务启动于: http://{SERVICE_HOST}:{SERVICE_PORT}")

    config = uvicorn.Config(
        app=app,
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        log_level="info",
        access_log=False,  # 交由 loguru 处理以减少冗余
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # 注册退出信号处理
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(f"服务器异常崩溃: {e}")