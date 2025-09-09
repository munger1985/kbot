import os
import atexit
import asyncio
import uvicorn
import signal
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from loguru import logger
from dotenv import load_dotenv
from configuration import ConfigManager
from core.nacos_manager import nacos_manager
from core.logger_manager import LogManager, LogConfig
from api.routers import router
from services.dataparse.file_parser_manger import FileParserManager


# 加载环境变量
from pathlib import Path
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

service_name = "main"
service_host = os.getenv("KBOT_HOST") or "0.0.0.0"
service_port = int(os.getenv("KBOT_PORT") or 8000)

# 注册退出时的清理函数
def cleanup():
    """在应用程序退出时关闭文件解析服务"""
    # shutdown_services("Application exiting, ")
    fp_manager = FileParserManager()
    fp_manager.shutdown_service("Application exiting, ")


atexit.register(cleanup)


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """
    try:
        # Initiate loguru configuration
        # 通过 nacos_manager 获取 app 配置
        app_config = ConfigManager.get_app_config()
        
        log_dir = app_config.kbot.log.dir
        log_level = app_config.kbot.log.level
        rotation = app_config.kbot.log.rotation
        retention = app_config.kbot.log.retention
        title = app_config.kbot.title
        description = app_config.kbot.description
        version = app_config.kbot.version
        debug = app_config.kbot.debug
        
        # 初始化日志
        conf = LogConfig(service_name=service_name, log_dir=log_dir, level=log_level, rotation=rotation, retention=retention)
        LogManager(conf).setup()

        logger.debug("Starting application initialization...")

        app = FastAPI(
            title=title,
            description=description,
            version=version,
            debug=debug
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

        # Add API documentation endpoint
        @app.get("/", include_in_schema=False)
        @app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui_html():
            return get_swagger_ui_html(
                openapi_url="/openapi.json",
                title=app.title + " - Swagger UI",
                swagger_js_url="/static/swagger-ui-bundle.js",
                swagger_css_url="/static/swagger-ui.css"
            )

        # Add API documentation endpoint
        @app.get("/redoc", include_in_schema=False)
        async def redoc_html():
            return get_redoc_html(
                openapi_url="/openapi.json",
                title=app.title + " - ReDoc",
                redoc_js_url="/static/redoc.standalone.js"
            )

        return app

    except Exception as e:
        logger.critical(f"Failed to create application: {str(e)}")
        raise

def signal_handler(sig, frame):
    """Processing termination signal."""
    logger.info(f"Received signal {sig}, shutting down...")
    sys.exit(0)


async def main():

    # 创建文件解析服务管理器实例
    fp_manager = FileParserManager()
    # 启动所有文件解析服务
    fp_manager.start_service()

    # 创建主应用程序
    app = create_app()
    logger.info("Application created...")

    # 配置uvicorn服务器
    logger.debug(f"Main application is using host: {service_host}, port: {service_port}")
    config = uvicorn.Config(app, host=service_host, port=service_port)
    server = uvicorn.Server(config)
    logger.info("Starting Uvicorn server...")

    # 注册到nacos
    logger.debug("Registering application to Nacos...")
    nacos_manager.register_service(service_name, service_host, service_port)

    # 运行服务器
    await server.serve()
    


if __name__ == "__main__":

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    asyncio.run(main())