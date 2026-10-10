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


_REQUIRED_AIOPS_SCHEMA_NOT_NULL_COLUMNS = frozenset(
    {
        ("KBOT_OPS_TASK", "TASK_TYPE"),
        ("KBOT_OPS_CHANGE_PROPOSAL", "TURN_ID"),
        ("KBOT_OPS_CONVERSATION_TURN", "CURRENT_PLAN_REVISION"),
        ("KBOT_OPS_INVESTIGATION_REVISION", "REVISION_ID"),
        ("KBOT_OPS_PLAYBOOK_INVOCATION", "PLAYBOOK_INVOCATION_ID"),
        ("KBOT_OPS_TOOL_INVOCATION", "TOOL_INVOCATION_ID"),
        ("KBOT_OPS_TURN_EVIDENCE", "EVIDENCE_ROLE"),
        ("KBOT_OPS_INSPECTION_PLAN", "AGENT_ID"),
        ("KBOT_OPS_INSPECTION_PLAN", "INSPECTION_TEMPLATE_ID"),
        (
            "KBOT_OPS_INSPECTION_PLAN",
            "INSPECTION_TEMPLATE_VERSION_ID",
        ),
        ("KBOT_OPS_INSPECTION_FIRE", "INSPECTION_TEMPLATE_ID"),
        (
            "KBOT_OPS_INSPECTION_FIRE",
            "INSPECTION_TEMPLATE_VERSION_ID",
        ),
        ("KBOT_OPS_INSPECTION_TEMPLATE", "CURRENT_VERSION_ID"),
        ("KBOT_OPS_INSPECTION_TEMPLATE_VER", "DEFINITION_JSON"),
        ("KBOT_OPS_TARGET", "IMPORTANCE_LEVEL"),
        ("KBOT_OPS_KNOWLEDGE_ASSET", "ASSET_KIND"),
        ("KBOT_OPS_KNOWLEDGE_VERSION", "SOURCE_HASH"),
        ("KBOT_OPS_RECOVERY_PROFILE", "RPO_SECONDS"),
        ("KBOT_OPS_RECOVERY_PROFILE", "RTO_SECONDS"),
        (
            "KBOT_OPS_RECOVERY_PROFILE",
            "REQUIRED_ASSURANCE_LEVEL",
        ),
        (
            "KBOT_OPS_RECOVERY_PROFILE",
            "REQUIRED_BACKUP_SOURCE_TYPES_JSON",
        ),
        ("KBOT_OPS_RECOVERY_DRILL", "RECOVERY_MARKER_JSON"),
        ("KBOT_OPS_RECOVERY_DRILL", "EVIDENCE_JSON"),
        ("KBOT_OPS_RECOVERY_DRILL", "SOURCE_TRUST_LEVEL"),
        ("KBOT_OPS_WORK_ITEM", "WORK_ITEM_ID"),
        ("KBOT_OPS_WORK_ITEM_OCCURRENCE", "OCCURRENCE_ID"),
        ("KBOT_OPS_WORK_ITEM_LINK", "WORK_ITEM_LINK_ID"),
        ("KBOT_OPS_WORK_ITEM_ACTIVITY", "ACTIVITY_ID"),
    }
)


def _required_schema_columns_statement():
    predicates = " OR ".join(
        (
            f"(TABLE_NAME = '{table_name}' "
            f"AND COLUMN_NAME = '{column_name}')"
        )
        for table_name, column_name in sorted(
            _REQUIRED_AIOPS_SCHEMA_NOT_NULL_COLUMNS
        )
    )
    return text(
        "SELECT TABLE_NAME, COLUMN_NAME, NULLABLE "
        "FROM USER_TAB_COLUMNS WHERE " + predicates
    )


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
                              AND schema_version = 37
                              AND contract_version = 'aiops-oracle-v27'
                            """
                        )
                    )
                ).scalar_one_or_none()
                if version_ready != 1:
                    return {
                        "aiops_schema": "version_mismatch",
                        "aiops_schema_integrity": "not_checked",
                    }
                required_column_rows = (
                    await session.execute(
                        _required_schema_columns_statement()
                    )
                ).all()
                required_columns = frozenset(
                    (str(row[0]).upper(), str(row[1]).upper())
                    for row in required_column_rows
                    if str(row[2]).upper() == "N"
                )
                report_summary_column = (
                    await session.execute(
                        text(
                            """
                            SELECT COUNT(*)
                            FROM USER_TAB_COLUMNS
                            WHERE TABLE_NAME = 'KBOT_OPS_REPORT'
                              AND COLUMN_NAME = 'SUMMARY'
                              AND DATA_TYPE = 'CLOB'
                            """
                        )
                    )
                ).scalar_one_or_none()
                report_source_table = (
                    await session.execute(
                        text(
                            """
                            SELECT COUNT(*)
                            FROM USER_TABLES
                            WHERE TABLE_NAME = 'KBOT_OPS_REPORT_SOURCE'
                            """
                        )
                    )
                ).scalar_one_or_none()
                business_check_constraints = (
                    await session.execute(
                        text(
                            """
                            SELECT COUNT(*)
                            FROM USER_CONSTRAINTS
                            WHERE TABLE_NAME LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\'
                              AND CONSTRAINT_TYPE = 'C'
                              AND GENERATED = 'USER NAME'
                            """
                        )
                    )
                ).scalar_one_or_none()
                integrity_ready = (
                    required_columns
                    == _REQUIRED_AIOPS_SCHEMA_NOT_NULL_COLUMNS
                    and report_summary_column == 1
                    and report_source_table == 1
                    and business_check_constraints == 0
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
