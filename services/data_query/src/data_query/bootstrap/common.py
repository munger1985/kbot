"""Data Query 进程共享生命周期与系统探针。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from collections.abc import Awaitable, Callable

from fastapi import HTTPException
from fastapi.responses import PlainTextResponse
from fastapi_offline import FastAPIOffline

from data_query.config import DataQuerySettings
from data_query.persistence import DataQueryUnitOfWork
from platform_core.database import DatabaseRuntime
from platform_core.logger import LogConfig, LogManager
from platform_core.middleware.log_middleware import log_requests


@dataclass
class DataQueryProcessRuntime:
    """单个 Data Query 进程的基础运行时资源。"""

    settings: DataQuerySettings
    service_name: str
    database_runtime: DatabaseRuntime
    uow_factory: Callable[[], DataQueryUnitOfWork] | None = None
    on_start: Callable[[], Awaitable[None]] | None = None
    on_close: Callable[[], Awaitable[None]] | None = None

    async def start(self) -> None:
        if self.on_start is not None:
            await self.on_start()

    async def close(self) -> None:
        if self.on_close is not None:
            await self.on_close()
        await self.database_runtime.close()

    async def check_database(self) -> dict[str, str]:
        try:
            if self.uow_factory is None:
                return {"database": "not_configured"}
            async with self.uow_factory() as uow:
                assert uow.health is not None
                ready = await uow.health.is_ready()
                await uow.commit()
            return {"database": "ok" if ready else "unexpected_result"}
        except Exception as exc:
            return {"database": type(exc).__name__}


def configure_process_logging(
    settings: DataQuerySettings, *, process: str,
) -> None:
    LogManager(
        LogConfig(
            service="kbot_data_query",
            process=process,
            log_dir=settings.log.dir,
            level=settings.log.level,
            rotation=settings.log.rotation,
            retention=settings.log.retention,
        )
    ).setup()


def create_process_app(
    *,
    title: str,
    description: str,
    runtime: DataQueryProcessRuntime,
    debug: bool,
) -> FastAPIOffline:
    """创建仅包含系统探针的 Data Query 进程应用。"""

    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = runtime
        await runtime.start()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPIOffline(
        title=title,
        description=description,
        version=runtime.settings.api.service_version,
        lifespan=lifespan,
        docs_url="/docs" if debug else None,
        redoc_url="/redoc" if debug else None,
    )
    app.state.service_name = runtime.service_name
    app.middleware("http")(log_requests)

    @app.get("/live", tags=["System"])
    async def live() -> dict[str, str]:
        return {
            "status": "live",
            "service": runtime.service_name,
            "version": runtime.settings.api.service_version,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    @app.get("/ready", tags=["System"])
    async def ready() -> dict[str, object]:
        checks = await runtime.check_database()
        if not all(value == "ok" for value in checks.values()):
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "service": runtime.service_name,
                    "checks": checks,
                },
            )
        return {"status": "ready", "service": runtime.service_name, "checks": checks}

    @app.get("/metrics", tags=["System"], response_class=PlainTextResponse)
    async def metrics() -> str:
        return (
            "# HELP kbot_process_live Data Query 进程是否存活\n"
            "# TYPE kbot_process_live gauge\n"
            f'kbot_process_live{{service="{runtime.service_name}"}} 1\n'
        )

    return app
