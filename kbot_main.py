"""Main application entry point.

This module is responsible for initializing the FastAPI application, loading global configurations,
managing the lifecycle of the file parsing service, and starting the Uvicorn server.
"""
# --- Fix: Suppress jieba warnings ---
# jieba internally uses deprecated setuptools interfaces causing UserWarning: pkg_resources is deprecated
# to appear on the first line of logs
import warnings
# Suppress pkg_resources deprecation warnings caused by jieba
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

# --- Environment Initialization ---
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle.

    Initialize the file parsing manager and start services when the application launches;
    ensure safe resource recovery when the application shuts down.

    Args:
        app: FastAPI instance.
    """
    # Set service name to app.state (used by middleware)
    app.state.service_name = get_app_config().service_name

    # Startup phase
    logger.info("Application is starting up, executing initialization tasks...")

    yield  # Application running

    # Shutdown phase
    logger.info("Application is shutting down, executing cleanup tasks...")
    


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Read global configurations, initialize logging system, mount routers and middleware.

    Returns:
        FastAPI: Configured application instance.
    """
    try:
        app_config = get_app_config()

        # 1. Initialize logging center
        log_conf = LogConfig(
            service_name=app_config.service_name,
            log_dir=app_config.log.dir,
            level=app_config.log.level,
            rotation=app_config.log.rotation,
            retention=app_config.log.retention,
        )
        LogManager(log_conf).setup()

        logger.debug("Configuring FastAPI instance...")

        # 2. Instantiate application (supports offline documentation)
        app = FastAPIOffline(
            title=app_config.title,
            description=app_config.description,
            version=app_config.service_version,
            debug=app_config.debug,
            lifespan=lifespan,  # Inject lifecycle manager
            docs_url="/docs" if app_config.debug else None,
            redoc_url="/redoc" if app_config.debug else None,
        )

        # 3. Middleware configuration
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 4. Request logging middleware
        app.middleware("http")(log_requests)

        # 5. Router registration
        app.include_router(router)

        # 6. Slack router (registered at root level — Slack requires exact
        #    paths like /slack/events without an /api prefix).
        from api.routers.slack_router import router as slack_router
        app.include_router(slack_router)

        return app

    except Exception as e:
        logger.critical(f"Failed to create application: {e}")
        raise


def handle_exit_signal(sig, frame):
    """Handle system exit signals (SIGINT, SIGTERM).

    Args:
        sig: Signal number.
        frame: Current stack frame.
    """
    logger.warning(f"Received signal {sig}, preparing for forced exit...")
    # Note: In lifespan mode, uvicorn handles graceful shutdown properly.
    # sys.exit here is mainly for forced scenarios like double Ctrl+C
    sys.exit(0)


async def start_server():
    """Configure and run the Uvicorn async server."""
    app = create_app()

    SERVICE_HOST = get_app_config().service_host
    SERVICE_PORT = get_app_config().service_port

    logger.info(f"Service started at: http://{SERVICE_HOST}:{SERVICE_PORT}")

    config = uvicorn.Config(
        app=app,
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        log_level="info",
        access_log=False,  # Handled by loguru to reduce redundancy
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # Register exit signal handlers
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(f"Server crashed unexpectedly: {e}")