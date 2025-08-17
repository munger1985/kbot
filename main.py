import os
import atexit
import asyncio
import uvicorn
import signal
import sys
import time
import socket
import configparser
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from loguru import logger
from dotenv import load_dotenv
from nacos import NacosClient
from nacos_manager import nacos_manager # type: ignore
from logger_manager import LogManager, LogConfig # type: ignore
from api.routers import router
from core.config import settings
from services.dataparse.file_parser_manger import FileParserManager


# 加载环境变量
load_dotenv()
service_name = "main"
nacos_addr = os.getenv("NACOS_SERVER_ADDR") # Nacos服务器地址
nacos_namespace = os.getenv("NACOS_NAMESPACE") or "public" # Nacos命名空间
nacos_group = os.getenv("NACOS_GROUP") or "DEV_GROUP" # Nacos分组
nacos_username = os.getenv("NACOS_USERNAME") # Nacos账号名称
nacos_password = os.getenv("NACOS_PASSWORD") # Nacos账号密码
service_host = os.getenv("KBOT_HOST", "0.0.0.0") # 服务地址
service_port = int(os.getenv("KBOT_PORT", 9000)) # 服务端口

# Nacos 服务注册
def register_service():
    client = NacosClient(
        server_addresses=nacos_addr,
        namespace=nacos_namespace
        # username='nacos',
        # password='nacos'
        )
    client.add_naming_instance(
        service_name=service_name,
        group_name=nacos_group,
        ip=service_host,
        port=service_port,
        ephemeral=True,
        healthy=True
    )
    # nacos 心跳发送器
    while True:
        if signal.SIGINT or signal.SIGTERM:
            break
        try:
            # 健康检查：检测服务端口是否存活
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((service_host, service_port))
            is_healthy = (result == 0)
            sock.close()

            # 更新实例健康状态
            client.send_heartbeat(
                service_name=service_name,
                group_name=nacos_group,
                ip=service_host,
                port=service_port
            )
            
            logger.info(f"Heartbeat sent, healthy: {is_healthy}")
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            break
        
        time.sleep(10)  # 间隔需小于Nacos心跳超时时间（默认15秒）

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
        # 通过 nacos_manager 获取logger配置
        try:
            config_parser = configparser.ConfigParser()
            log_config = nacos_manager.get_config("logger", nacos_group)
            config_parser.read_string(f"[{nacos_group}]\n{log_config}")
            log_dir = config_parser.get(nacos_group, "dir") or "logs/"
            log_level = config_parser.get(nacos_group, "level") or "DEBUG"
            rotation = config_parser.get(nacos_group, "rotation") or "10 MB"
            retention = config_parser.get(nacos_group, "retention") or "20 days"
            
        except Exception as e:
            # 如果获取 logger 配置失败，则使用默认配置
            logger.warning(f"Failed to get logger config from nacos: {str(e)}")
            log_dir = "logs/"
            log_level = "DEBUG"
            rotation = "10 MB"
            retention = "10 days"
            
        # 初始化日志
        conf = LogConfig(service_name=service_name, log_dir=log_dir, level=log_level, rotation=rotation, retention=retention)
        LogManager(conf).setup()

        logger.debug("Starting application initialization...")
        try:
            app_conf = nacos_manager.get_config("app", nacos_group)
            config_parser.read_string(f"[{nacos_group}]\n{app_conf}")
            title = config_parser.get(nacos_group, "title") or "KBot API 3.0"
            description = config_parser.get(nacos_group, "description") or "KBot API 3.0"
            version = config_parser.get(nacos_group, "version") or "3.0.0"
            debug = config_parser.get(nacos_group, "debug").lower() == "true" or False
        except Exception as e:
            # 如果获取app配置失败，则使用默认配置
            logger.warning(f"Failed to get app config from nacos: {str(e)}")
            title = "KBot API 3.0"
            description = "KBot API 3.0"
            version = "1.0.0"
            debug = True

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
    register_service()

    # 运行服务器
    await server.serve()


if __name__ == "__main__":

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    asyncio.run(main())