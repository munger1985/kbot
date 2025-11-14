import os
import atexit
import asyncio
import uvicorn
import signal
import sys
from fastapi import FastAPI
from fastapi_offline import FastAPIOffline
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from dotenv import load_dotenv
from core.logger_manager import LogManager, LogConfig
from core.config.settings import get_app_config
from api.routers import router
from services.dataparse.file_parser_manger import FileParserManager


# 加载环境变量
from pathlib import Path
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

service_name = os.getenv("KBOT_SERVICE_NAME") or "main"
service_host = os.getenv("KBOT_HOST") or "0.0.0.0"
service_port = int(os.getenv("KBOT_PORT") or 8000)

# 注册退出时的清理函数
def cleanup():
    """在应用程序退出时关闭文件解析服务"""
    # shutdown_services("Application exiting, shutting down file parse service")
    fp_manager = FileParserManager()
    fp_manager.shutdown_service("应用退出，关闭文件解析服务")


atexit.register(cleanup)


def create_app() -> FastAPI:
    """创建并配置FastAPI应用程序

    Returns:
        FastAPI: 配置好的应用实例
    """
    try:
        # 读取日志配置
        app_config = get_app_config()
        
        log_dir = app_config.log.dir
        log_level = app_config.log.level
        rotation = app_config.log.rotation
        retention = app_config.log.retention
        title = app_config.title
        description = app_config.description
        version = app_config.version
        debug = app_config.debug

        
        # 初始化日志
        conf = LogConfig(service_name=service_name, log_dir=log_dir, level=log_level, rotation=rotation, retention=retention)
        LogManager(conf).setup()

        logger.debug("开始创建FastAPI应用...")

        # 使用 FastAPIOffline 的默认配置，它会自动处理离线文档
        app = FastAPIOffline(
            title=title,
            description=description,
            version=version,
            debug=debug,
            # 这些参数让 FastAPIOffline 自动处理静态文件
            docs_url="/docs" if debug else None,  # 生产环境可禁用
            redoc_url="/redoc" if debug else None
        )

        # Add middleware with safer defaults
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )

        # Add routers
        app.include_router(router)

        return app

    except Exception as e:
        logger.critical(f"无法创建FastAPI应用: {str(e)}")
        raise

def signal_handler(sig, frame):
    """处理终止信号"""
    logger.info(f"收到信号 {sig}, 正在关闭应用...")
    sys.exit(0)


async def main():

    # 创建文件解析服务管理器实例
    fp_manager = FileParserManager()
    # 启动所有文件解析服务
    fp_manager.start_service()

    # 创建主应用程序
    app = create_app()
    logger.info("应用创建完成")

    # 配置uvicorn服务器
    logger.debug(f"应用正在使用主机: {service_host}, 端口: {service_port}")
    config = uvicorn.Config(app, host=service_host, port=service_port)
    server = uvicorn.Server(config)
    logger.info("开始启动Uvicorn服务器...")

    # 运行服务器
    await server.serve()
    logger.info("Uvicorn服务器启动完成")
    


if __name__ == "__main__":

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    asyncio.run(main())