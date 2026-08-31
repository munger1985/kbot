"""AIOps 进程共享的生命周期与系统探针。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Awaitable, Callable

from fastapi import HTTPException
from fastapi.responses import PlainTextResponse
from fastapi_offline import FastAPIOffline
from sqlalchemy import text

from aiops_agent.config import AIOpsSettings
from aiops_agent.persistence import AIOpsUnitOfWork
from platform_core.database.oracle import DatabaseRuntime
from platform_core.logger import LogConfig, LogManager
from platform_core.middleware.log_middleware import log_requests


ReadyCheck = Callable[[], Awaitable[dict[str, str]]]


@dataclass
class AIOpsProcessRuntime:
    """单个进程独占且可显式关闭的资源集合。"""

    settings: AIOpsSettings
    service_name: str
    database_runtime: DatabaseRuntime | None = None
    uow_factory: Callable[[], AIOpsUnitOfWork] | None = None
    components: dict[str, bool] = field(default_factory=dict)

    async def start(self) -> None:
        """步骤 0 不在启动时连接外部 Provider 或创建后台任务。"""

    async def close(self) -> None:
        if self.database_runtime is not None:
            await self.database_runtime.close()

    async def check_aiops_schema(self) -> dict[str, str]:
        if self.database_runtime is None:
            return {
                "aiops_schema": "database_not_configured",
                "aiops_schema_integrity": "not_checked",
            }
        try:
            async with self.database_runtime.session_factory() as session:
                version_ready = (
                    await session.execute(
                        text(
                            """
                            SELECT 1
                            FROM KBOT_V_OPS_SCHEMA_VERSION
                            WHERE component = 'AIOPS'
                              AND schema_version = 16
                              AND contract_version = 'aiops-oracle-v6'
                            """
                        )
                    )
                ).scalar_one_or_none()
                if version_ready != 1:
                    return {
                        "aiops_schema": "version_mismatch",
                        "aiops_schema_integrity": "not_checked",
                    }
                required_columns = (
                    await session.execute(
                        text(
                            """
                            SELECT COUNT(*)
                            FROM USER_TAB_COLUMNS
                            WHERE NULLABLE = 'N'
                              AND (
                                (TABLE_NAME = 'KBOT_OPS_TASK'
                                 AND COLUMN_NAME = 'TASK_TYPE')
                                OR (TABLE_NAME = 'KBOT_OPS_CHANGE_PROPOSAL'
                                    AND COLUMN_NAME = 'TURN_ID')
                                OR (TABLE_NAME = 'KBOT_OPS_CONVERSATION_TURN'
                                    AND COLUMN_NAME = 'CURRENT_PLAN_REVISION')
                                OR (TABLE_NAME = 'KBOT_OPS_INVESTIGATION_REVISION'
                                    AND COLUMN_NAME = 'REVISION_ID')
                                OR (TABLE_NAME = 'KBOT_OPS_PLAYBOOK_INVOCATION'
                                    AND COLUMN_NAME = 'PLAYBOOK_INVOCATION_ID')
                                OR (TABLE_NAME = 'KBOT_OPS_TOOL_INVOCATION'
                                    AND COLUMN_NAME = 'TOOL_INVOCATION_ID')
                                OR (TABLE_NAME = 'KBOT_OPS_TURN_EVIDENCE'
                                    AND COLUMN_NAME = 'EVIDENCE_ROLE')
                              )
                            """
                        )
                    )
                ).scalar_one_or_none()
                task_type_constraint = (
                    await session.execute(
                        text(
                            """
                            SELECT COUNT(*)
                            FROM USER_CONSTRAINTS
                            WHERE TABLE_NAME = 'KBOT_OPS_TASK'
                              AND CONSTRAINT_NAME = 'CK_OPS_TASK_TYPE'
                              AND CONSTRAINT_TYPE = 'C'
                              AND STATUS = 'ENABLED'
                              AND VALIDATED = 'VALIDATED'
                              AND SEARCH_CONDITION_VC LIKE '%CONTEXT_BUILD%'
                              AND SEARCH_CONDITION_VC LIKE '%PLAYBOOK_INVOKE%'
                              AND SEARCH_CONDITION_VC LIKE '%PROPOSAL%'
                              AND SEARCH_CONDITION_VC NOT LIKE '%INTENT_ROUTE%'
                              AND SEARCH_CONDITION_VC NOT LIKE '%SKILL_PLAN%'
                              AND SEARCH_CONDITION_VC NOT LIKE '%SKILL_INVOKE%'
                            """
                        )
                    )
                ).scalar_one_or_none()
                integrity_ready = (
                    required_columns == 7 and task_type_constraint == 1
                )
            return {
                "aiops_schema": "ok",
                "aiops_schema_integrity": (
                    "ok" if integrity_ready else "contract_mismatch"
                ),
            }
        except Exception as exc:
            return {
                "aiops_schema": type(exc).__name__,
                "aiops_schema_integrity": "not_checked",
            }

    async def check_executor_components(self) -> dict[str, str]:
        return {
            name: "ok" if ready else "not_configured"
            for name, ready in self.components.items()
        }


def configure_process_logging(
    settings: AIOpsSettings,
    *,
    process: str,
) -> None:
    LogManager(
        LogConfig(
            service="aiops_agent",
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
    service_name: str,
    service_version: str,
    debug: bool,
    lifespan,
) -> FastAPIOffline:
    """创建不带 CORS 和领域路由的内部进程 App。"""
    app = FastAPIOffline(
        title=title,
        description=description,
        version=service_version,
        lifespan=lifespan,
        docs_url="/docs" if debug else None,
        redoc_url="/redoc" if debug else None,
    )
    app.state.service_name = service_name
    app.middleware("http")(log_requests)

    @app.get("/live", tags=["System"])
    async def live() -> dict[str, str]:
        return {
            "status": "live",
            "service": service_name,
            "version": service_version,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    @app.get("/ready", tags=["System"])
    async def ready() -> dict[str, object]:
        runtime: AIOpsProcessRuntime = app.state.runtime
        checks = await app.state.ready_check()
        is_ready = bool(checks) and all(
            value == "ok" for value in checks.values()
        )
        payload = {
            "status": "ready" if is_ready else "not_ready",
            "service": runtime.service_name,
            "checks": checks,
        }
        if not is_ready:
            raise HTTPException(status_code=503, detail=payload)
        return payload

    @app.get(
        "/metrics",
        tags=["System"],
        response_class=PlainTextResponse,
    )
    async def metrics() -> str:
        return (
            "# HELP kbot_process_live AIOps 进程是否存活\n"
            "# TYPE kbot_process_live gauge\n"
            f'kbot_process_live{{service="{service_name}"}} 1\n'
        )

    return app
