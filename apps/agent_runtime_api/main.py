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

from agent_runtime.api import internal_router, task_router
from agent_runtime.application import AgentRuntimeService
from agent_runtime.config import get_agent_runtime_settings
from agent_runtime.domain.planning import PlanLimits, PlanValidator
from agent_runtime.domain.skills import SkillRegistry
from agent_runtime.persistence import create_agent_runtime_uow
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager
from platform_core.middleware.log_middleware import log_requests
from platform_core.platform.port_check import check_port_available
from platform_core.security import create_internal_auth_middleware


settings = get_agent_runtime_settings()
config = settings.api
SERVICE_NAME = config.service_name
SERVICE_VERSION = config.service_version


@asynccontextmanager
async def lifespan(app: FastAPIOffline):
    """创建本进程独占的日志和数据库运行时。"""
    LogManager(LogConfig(
        service_name=SERVICE_NAME,
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
    skill_registry = SkillRegistry()
    app.state.agent_runtime_skill_registry = skill_registry
    app.state.agent_runtime_service = AgentRuntimeService(
        uow_factory=create_agent_runtime_uow(db_runtime.session_factory),
        plan_validator=PlanValidator(
            skill_exists=skill_registry.contains,
            capability_exists=lambda service, capability: False,
            public_artifact_types={
                "FINAL_RESPONSE",
                "GROUNDED_ANSWER",
            },
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
