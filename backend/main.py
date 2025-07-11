import os
import uvicorn
import signal
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
)
from loguru import logger
from api.routers import router
from api.routers.kb_upload_router import router as kb_upload_router
from core.log.logger import setup_logging
from core.config import settings

# 导入微服务的启动和关闭函数
from microservices.embedding.app import start_embedding_service, shutdown_embedding_service
import atexit

# 全局变量，用于存储微服务进程
embedding_service_process = None

# 注册退出时的清理函数
def cleanup():
    """在应用程序退出时关闭微服务"""
    global embedding_service_process
    if embedding_service_process:
        logger.info("应用程序退出，关闭嵌入微服务")
        shutdown_embedding_service()
        embedding_service_process = None

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
            debug=settings["app"]["debug"],
        )
        
        # Add middleware with safer defaults
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routers
        app.include_router(router)
        app.include_router(kb_upload_router)
        
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
                swagger_css_url="/static/swagger-ui.css",
            )
        
        # Add API documentation endpoint
        @app.get("/redoc", include_in_schema=False)
        async def redoc_html():
            return get_redoc_html(
                openapi_url="/openapi.json",
                title=app.title + " - ReDoc",
                redoc_js_url="/static/redoc.standalone.js",
            )

        return app
        
    except Exception as e:
        logger.critical(f"Failed to create application: {str(e)}")
        raise

def signal_handler(sig, frame):
    """处理终止信号"""
    logger.info(f"Received signal {sig}, shutting down")
    # 确保在退出前关闭微服务
    global embedding_service_process
    if embedding_service_process:
        logger.info("关闭嵌入微服务")
        shutdown_embedding_service()
        embedding_service_process = None
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动嵌入微服务
    embedding_service_process = start_embedding_service()
    logger.info(f"Embedding microservice started with PID {embedding_service_process.pid}")
    
    # 创建并启动主应用程序
    app = create_app()
    logger.info("Application created, starting Uvicorn server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    # 当作为模块导入时，创建应用程序
    app = create_app()
    logger.info("Application created as module")
    
    # 启动嵌入微服务
    embedding_service_process = start_embedding_service()
    logger.info(f"Embedding microservice started with PID {embedding_service_process.pid}")