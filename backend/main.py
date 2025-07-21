import os
import atexit
import asyncio
import uvicorn
import signal
import sys
import multiprocessing
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html
)
from loguru import logger
from dotenv import load_dotenv
from api.routers import router
from api.routers.kb_router import router as kb_router
from core.log.logger import setup_logging
from core.config import settings
from services.dataparse.parse_file import start_file_parse_service
from microservices.embedding.app import start_embedding_service, shutdown_embedding_service
from microservices.llm.app import start_llm_service, shutdown_llm_service

# 全局变量，用于存储微服务进程和后台子进程状态
embedding_service_process = None
llm_service_process = None
file_parse_service_process = None

def run_file_parse_service():
    """
    在子进程中运行文件解析服务。
    这个函数作为multiprocessing.Process的目标函数。
    """
    asyncio.run(start_file_parse_service())

# 注册退出时的清理函数
def cleanup():
    """在应用程序退出时关闭微服务"""
    global embedding_service_process, file_parse_service_process, llm_service_process
    if embedding_service_process:
        logger.info("Application exiting, shutting down the embedding microservice.")
        shutdown_embedding_service()
        embedding_service_process = None

    if llm_service_process:
        logger.info("Application exiting, shutting down the LLM microservice.")
        start_llm_service()
        llm_service_process = None
    
    if file_parse_service_process:
        logger.info("Application exiting, shutting down the file parsing service.")
        file_parse_service_process.terminate()
        file_parse_service_process.join()
        file_parse_service_process = None

atexit.register(cleanup)

def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """
    try:
        # Initiate loguru configuration
        setup_logging()
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
        app.include_router(kb_router)

        # Add health check endpoint
        @app.get("/health", tags=["health"])
        async def health_check() -> JSONResponse:
            return JSONResponse({"status": "ok"})

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
    # 确保在退出前关闭微服务
    global embedding_service_process, file_parse_service_process, llm_service_process
    if embedding_service_process:
        logger.info("Shutting down the embedding microservice...")
        # 关闭嵌入微服务
        shutdown_embedding_service()
        embedding_service_process = None

    if llm_service_process:
        logger.info("Shutting down the LLM microservice...")
        # 关闭LLM微服务
        shutdown_llm_service()
        llm_service_process = None
    
    if file_parse_service_process:
        logger.info("Shutting down the file parsing service...")
        file_parse_service_process.terminate()
        file_parse_service_process.join()
        file_parse_service_process = None
    
    sys.exit(0)


async def main():
    global embedding_service_process, file_parse_service_process
    
    # 启动嵌入微服务
    embedding_service_process = start_embedding_service()
    logger.info(f"Embedding microservice started with PID {embedding_service_process.pid}")

    # 启动LLM微服务
    llm_service_process = start_llm_service()
    logger.info(f"LLM microservice started with PID {llm_service_process.pid}")
    
    # 启动文件解析服务
    file_parse_service_process = multiprocessing.Process(target=run_file_parse_service)
    file_parse_service_process.daemon = True  # 设置为守护进程，这样主进程退出时，子进程也会退出
    file_parse_service_process.start()
    logger.info(f"File parse service started with PID {file_parse_service_process.pid}")

    # 创建主应用程序
    app = create_app()
    logger.info("Application created")

    # 配置uvicorn服务器
    host = os.environ.get("KBOT_HOST", "0.0.0.0")
    port = int(os.environ.get("KBOT_PORT", 8000))
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    logger.info("Starting Uvicorn server")

    # 运行服务器
    await server.serve()


if __name__ == "__main__":
    load_dotenv()
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    asyncio.run(main())