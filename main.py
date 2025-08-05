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
from api.routers import router
from core.log.logger import setup_logging
from core.config import settings
from microservices.microservice_manger import MicroserviceManager


# 加载环境变量
load_dotenv()

# 注册退出时的清理函数
def cleanup():
    """在应用程序退出时关闭微服务"""
    # shutdown_services("Application exiting, ")
    microservice_manager = MicroserviceManager()
    microservice_manager.shutdown_all_services("Application exiting, ")

atexit.register(cleanup)

def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """
    try:
        # Initiate loguru configuration
        setup_logging(service_name="main")
        logger.debug("Starting application initialization")

        app = FastAPI(
            title=settings["app"]["name"],
            description=settings["app"]["description"],
            version=settings["app"]["version"],
            debug=settings["app"]["debug"]
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
    logger.info(f"Received signal {sig}, shutting down")
    sys.exit(0)


async def main():

    # 创建微服务管理器实例
    service_manager = MicroserviceManager()
    # 启动所有微服务
    service_manager.start_all_services()

    # 创建主应用程序
    app = create_app()
    logger.info("Application created")

    # 配置uvicorn服务器
    host = os.environ.get("KBOT_HOST", "0.0.0.0")
    port = int(os.environ.get("KBOT_PORT", 8000))
    logger.debug(f"Main application is using host: {host}, port: {port}")
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    logger.info("Starting Uvicorn server")

    # 运行服务器
    await server.serve()


if __name__ == "__main__":

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    asyncio.run(main())