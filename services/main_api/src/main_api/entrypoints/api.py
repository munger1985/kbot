"""KBot 4.0 Main API/BFF 进程入口。"""

from __future__ import annotations

import sys
import os
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from fastapi import FastAPI
from loguru import logger

from main_api.app import create_main_api_app
from main_api.application import (
    AccessManagementService,
    AccessControlService,
    AppApiKeyService,
    DomainManagementService,
    DomainValidationService,
    NotificationCenterService,
    UserAuthService,
    create_user_token_codec,
)
from main_api.config import get_main_api_settings
from main_api.persistence import create_main_api_uow
from platform_clients import (
    AIOpsClientAuth,
    AIOpsManagementClient,
    AgentRuntimeClient,
    AIModelConfigClient,
    KnowledgeCoreClient,
    KnowledgeRetrievalAppClient,
    KmAssetClient,
    DataQueryClient,
)
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager
from platform_core.platform.port_check import check_port_available
from platform_core.security import (
    create_auth_context_codec,
    create_service_identity_codec,
)
from platform_core.security.runtime import DEFAULT_DEV_API_KEY_PEPPER


settings = get_main_api_settings()
config = settings.api


def _configure_model_config_clients(app: FastAPI) -> None:
    """把四类模型配置服务接入 Main API 的公开目录聚合。"""
    dependencies = (
        settings.model_embedding,
        settings.model_llm,
        settings.model_visual,
        settings.model_vlm,
    )
    app.state.model_config_clients = tuple(
        AIModelConfigClient(
            base_url=dependency.base_url,
            timeout=dependency.timeout_seconds,
            caller_service=config.service_name,
            audience=dependency.audience,
        )
        for dependency in dependencies
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化 Main API 自有数据库和跨服务 Client。"""
    app.state.service_name = config.service_name
    app.state.main_api_settings = settings
    LogManager(LogConfig(
        service="main_api",
        process="api",
        log_dir=settings.log.dir,
        level=settings.log.level,
        rotation=settings.log.rotation,
        retention=settings.log.retention,
    )).setup()
    db_runtime = create_database_runtime()
    app.state.db_runtime = db_runtime
    uow_factory = create_main_api_uow(db_runtime.session_factory)
    app.state.domain_validation_service = DomainValidationService(
        uow_factory=uow_factory,
    )
    app.state.domain_management_service = DomainManagementService(
        uow_factory=uow_factory,
    )
    app.state.access_control_service = AccessControlService(
        uow_factory=uow_factory,
    )
    app.state.access_management_service = AccessManagementService(
        uow_factory=uow_factory,
    )
    app.state.user_auth_service = UserAuthService(
        uow_factory=uow_factory,
        codec=create_user_token_codec(settings=settings),
    )
    api_key_pepper = os.getenv(settings.security.api_key_pepper_env)
    if not api_key_pepper:
        if settings.is_production():
            raise RuntimeError(
                f"生产环境必须设置 {settings.security.api_key_pepper_env}"
            )
        logger.warning("当前使用开发 App API Key Pepper")
        api_key_pepper = DEFAULT_DEV_API_KEY_PEPPER
    app.state.app_api_key_service = AppApiKeyService(
        uow_factory=uow_factory,
        pepper=api_key_pepper,
    )
    app.state.notification_center_service = NotificationCenterService(
        uow_factory=uow_factory,
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
    app.state.knowledge_retrieval_app_client = KnowledgeRetrievalAppClient(
        base_url=settings.knowledge_retrieval_app.base_url,
        caller_service=config.service_name,
        audience=settings.knowledge_retrieval_app.audience,
        timeout_seconds=settings.knowledge_retrieval_app.timeout_seconds,
        session=client_session,
    )
    app.state.km_asset_client = KmAssetClient(
        base_url=settings.km_asset_app.base_url,
        caller_service=config.service_name,
        audience=settings.km_asset_app.audience,
        timeout_seconds=settings.km_asset_app.timeout_seconds,
        session=client_session,
    )
    app.state.agent_runtime_client = AgentRuntimeClient(
        base_url=settings.agent_runtime.base_url,
        caller_service=config.service_name,
        audience=settings.agent_runtime.audience,
        timeout_seconds=settings.agent_runtime.timeout_seconds,
        session=client_session,
    )
    app.state.data_query_client = DataQueryClient(
        base_url=settings.data_query.base_url,
        caller_service=config.service_name,
        audience=settings.data_query.audience,
        timeout_seconds=settings.data_query.timeout_seconds,
        session=client_session,
    )
    _configure_model_config_clients(app)
    app.state.aiops_client = AIOpsManagementClient(
        base_url=settings.aiops.base_url,
        auth=AIOpsClientAuth(
            caller_service=config.service_name,
            audience=settings.aiops.audience,
            scopes=(
                "aiops.manage",
                "aiops.run",
                "aiops.hitl",
                "aiops.approve",
                "aiops.monitor.intake",
            ),
            auth_context_codec=create_auth_context_codec(),
            service_identity_codec=create_service_identity_codec(),
        ),
        timeout_seconds=settings.aiops.timeout_seconds,
        upload_timeout_seconds=settings.aiops.upload_timeout_seconds,
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
