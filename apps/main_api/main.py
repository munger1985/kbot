"""KBot 4.0 Main API/BFF 进程入口。"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from fastapi import FastAPI
from loguru import logger

from main_api.app import create_main_api_app
from main_api.application import DomainValidationService
from main_api.config import get_main_api_settings
from main_api.persistence import create_main_api_uow
from platform_clients import (
    AIOpsClientAuth,
    AIOpsManagementClient,
    AgentRuntimeClient,
    KnowledgeCoreClient,
)
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager
from platform_core.platform.port_check import check_port_available
from platform_core.security import (
    create_auth_context_codec,
    create_service_identity_codec,
)


settings = get_main_api_settings()
config = settings.api


@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化 Main API 自有数据库和跨服务 Client。"""
    app.state.service_name = config.service_name
    app.state.main_api_settings = settings
    LogManager(LogConfig(
        service_name=config.service_name,
        log_dir=settings.log.dir,
        level=settings.log.level,
        rotation=settings.log.rotation,
        retention=settings.log.retention,
    )).setup()
    db_runtime = create_database_runtime()
    app.state.db_runtime = db_runtime
    app.state.domain_validation_service = DomainValidationService(
        app_id=settings.platform.app_id,
        uow_factory=create_main_api_uow(db_runtime.session_factory),
    )
    client_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(
            total=settings.knowledge_core.timeout_seconds,
        )
    )
    app.state.knowledge_core_client = KnowledgeCoreClient(
        base_url=settings.knowledge_core.base_url,
        caller_service=config.service_name,
        audience=settings.knowledge_core.audience,
        timeout_seconds=settings.knowledge_core.timeout_seconds,
        session=client_session,
    )
    app.state.agent_runtime_client = AgentRuntimeClient(
        base_url=settings.agent_runtime.base_url,
        caller_service=config.service_name,
        audience=settings.agent_runtime.audience,
        timeout_seconds=settings.agent_runtime.timeout_seconds,
        session=client_session,
    )
    app.state.aiops_client = AIOpsManagementClient(
        base_url=settings.aiops.base_url,
        auth=AIOpsClientAuth(
            caller_service=config.service_name,
            audience=settings.aiops.audience,
            scopes=("aiops.manage",),
            auth_context_codec=create_auth_context_codec(),
            service_identity_codec=create_service_identity_codec(),
        ),
        timeout_seconds=settings.aiops.timeout_seconds,
        session=client_session,
    )
    logger.info("Main API 已启动，公开前缀=/api/v1")
    try:
        yield
    finally:
        await client_session.close()
        await db_runtime.close()
        logger.info("Main API 已停止")


app = create_main_api_app(lifespan=lifespan)


if __name__ == "__main__":
    if not check_port_available(
        config.service_host,
        config.service_port,
        config.service_name,
    ):
        sys.exit(1)
    uvicorn.run(
        app,
        host=config.service_host,
        port=config.service_port,
        log_config=None,
        loop="asyncio",
    )
