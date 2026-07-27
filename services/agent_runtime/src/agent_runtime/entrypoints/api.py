"""Agent Runtime 内部 API 入口。

当前基础阶段只提供进程与数据库就绪检查。Run 命令路由会在 Repository/UoW
完成后注册，避免暴露假成功接口。
"""

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import HTTPException, Request
from fastapi_offline import FastAPIOffline
from loguru import logger
from sqlalchemy import text

from agent_runtime.api import (
    agent_router,
    conversation_router,
    data_router,
    internal_router,
    memory_router,
    task_router,
)
from agent_runtime.application import (
    AgentDefinitionService,
    AgentRuntimeService,
    ConversationService,
    ConversationAttachmentStore,
    MemoryRecallService,
)
from agent_runtime.application.model_resolution import (
    AgentModelCatalogResolver,
)
from agent_runtime.config import get_agent_runtime_settings
from agent_runtime.domain.planning import PlanLimits, PlanValidator
from agent_runtime.domain.skills import SkillRegistry
from agent_runtime.persistence import create_agent_runtime_uow
from agent_runtime.specialists import register_builtin_manifests
from agent_runtime.specialists.root import RootAgentPlanner
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager
from platform_core.middleware.log_middleware import log_requests
from platform_core.platform.port_check import check_port_available
from platform_core.security import create_internal_auth_middleware
from platform_clients import AIModelClient, AIModelConfigClient, MCPDataClient
from platform_core.dictionary import ModelCategory
from platform_core.prompts import PromptResolver, load_prompt_catalog
from pathlib import Path


settings = get_agent_runtime_settings()
config = settings.api
SERVICE_NAME = config.service_name
SERVICE_VERSION = config.service_version


@asynccontextmanager
async def lifespan(app: FastAPIOffline):
    """创建本进程独占的日志和数据库运行时。"""
    LogManager(LogConfig(
        service="agent_runtime",
        process="api",
        log_dir=settings.log.dir,
        level=settings.log.level,
        rotation=settings.log.rotation,
        retention=settings.log.retention,
    )).setup()
    db_runtime = create_database_runtime()
    app.state.db_runtime = db_runtime
    app.state.platform_app_id = settings.platform.app_id
    app.state.agent_runtime_budget = {
        "max_tasks": settings.worker.max_tasks_per_run,
        "max_parallel_tasks": settings.worker.max_parallel_tasks,
        "max_total_retries": settings.worker.max_total_retries,
        "max_task_timeout_seconds": (
            settings.worker.max_task_timeout_seconds
        ),
    }
    skill_registry = register_builtin_manifests(SkillRegistry())
    app.state.agent_runtime_skill_registry = skill_registry
    uow_factory = create_agent_runtime_uow(db_runtime.session_factory)
    model_resolver = AgentModelCatalogResolver(
        {
            ModelCategory.LLM: AIModelConfigClient(
                base_url=settings.llm.base_url,
                timeout=settings.llm.timeout_seconds,
                caller_service=SERVICE_NAME,
                audience=settings.llm.audience,
            ),
            ModelCategory.TXT_EMBEDDING: AIModelConfigClient(
                base_url=settings.embedding.base_url,
                timeout=settings.embedding.timeout_seconds,
                caller_service=SERVICE_NAME,
                audience=settings.embedding.audience,
            ),
            ModelCategory.VLM: AIModelConfigClient(
                base_url=settings.vlm.base_url,
                timeout=settings.vlm.timeout_seconds,
                caller_service=SERVICE_NAME,
                audience=settings.vlm.audience,
            ),
        }
    )
    app.state.agent_definition_service = AgentDefinitionService(
        uow_factory=uow_factory,
        model_resolver=model_resolver,
    )
    model_client = AIModelClient(
        caller_service=SERVICE_NAME,
        embedding_config=settings.embedding,
        llm_config=settings.llm,
    )
    prompt_resolver = PromptResolver(
        session_factory=db_runtime.session_factory,
        catalog=load_prompt_catalog(),
    )
    try:
        mcp_api_key = settings.ask_data_api.require_api_key()
    except RuntimeError:
        app.state.mcp_data_client = None
        logger.warning("未配置问数 API Key，问数与 Profile 列表暂不可用")
    else:
        app.state.mcp_data_client = MCPDataClient(
            api_endpoint=settings.ask_data_api.api_endpoint,
            profiles_endpoint=settings.ask_data_api.profiles_endpoint,
            api_key=mcp_api_key,
            timeout_seconds=settings.ask_data_api.timeout,
            max_rows=settings.ask_data_api.max_rows,
            max_response_bytes=(
                settings.ask_data_api.max_response_bytes
            ),
        )
    runtime_service = AgentRuntimeService(
        uow_factory=uow_factory,
        plan_validator=PlanValidator(
            skill_exists=skill_registry.contains,
            capability_exists=lambda service, capability: (
                service == "aiops_agent" and capability == "diagnosis"
            ),
            public_artifact_types={"GROUNDED_ANSWER"},
        ),
        plan_limits=PlanLimits(
            max_tasks=settings.worker.max_tasks_per_run,
            max_parallel_tasks=settings.worker.max_parallel_tasks,
            max_total_retries=settings.worker.max_total_retries,
            max_task_timeout_seconds=(
                settings.worker.max_task_timeout_seconds
            ),
        ),
        skill_registry=skill_registry,
        root_planner=RootAgentPlanner(
            model_client=model_client,
            prompt_resolver=prompt_resolver,
        ),
        model_resolver=model_resolver,
    )
    app.state.agent_runtime_service = runtime_service
    app.state.conversation_service = ConversationService(
        uow_factory=uow_factory,
        runtime_service=runtime_service,
        memory_recall_service=MemoryRecallService(
            uow_factory=uow_factory,
            model_client=model_client,
        ),
        attachment_store=ConversationAttachmentStore(
            Path(settings.attachments.local_storage_path)
        ),
    )
    logger.info("正在启动服务 [{}]，进程号={}", SERVICE_NAME, os.getpid())
    try:
        yield
    finally:
        await db_runtime.close()
        logger.info("正在停止服务 [{}]", SERVICE_NAME)


app = FastAPIOffline(
    title="KBot Agent Runtime Internal API",
    description="Agent Run、Task、Artifact 和 Event 的内部命令边界。",
    version=SERVICE_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.platform.debug else None,
    redoc_url="/redoc" if settings.platform.debug else None,
)
app.middleware("http")(log_requests)
app.middleware("http")(
    create_internal_auth_middleware(audience=SERVICE_NAME)
)
app.include_router(internal_router)
app.include_router(task_router)
app.include_router(agent_router)
app.include_router(data_router)
app.include_router(conversation_router)
app.include_router(memory_router)


@app.get("/health", tags=["System"])
async def health() -> dict[str, Any]:
    """存活检查不访问外部依赖。"""
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/healthz", tags=["System"])
async def healthz() -> dict[str, Any]:
    return await health()


@app.get("/readyz", tags=["System"])
async def readyz(request: Request) -> dict[str, Any]:
    """确认数据库连接可用；尚未部署 Schema 时返回 503。"""
    checks: dict[str, str] = {}
    try:
        async with request.app.state.db_runtime.session_factory() as session:
            await session.execute(
                text("SELECT 1 FROM KBOT_AGENT_RUN WHERE 1 = 0")
            )
        checks["agent_schema"] = "ok"
    except Exception as exc:
        checks["agent_schema"] = type(exc).__name__
    ready = all(value == "ok" for value in checks.values())
    payload = {
        "status": "ready" if ready else "not_ready",
        "service": SERVICE_NAME,
        "checks": checks,
    }
    if not ready:
        raise HTTPException(status_code=503, detail=payload)
    return payload


if __name__ == "__main__":
    if not check_port_available(
        config.service_host,
        config.service_port,
        SERVICE_NAME,
    ):
        sys.exit(1)
    uvicorn.run(
        app,
        host=config.service_host,
        port=config.service_port,
        log_config=None,
        loop="asyncio",
    )
