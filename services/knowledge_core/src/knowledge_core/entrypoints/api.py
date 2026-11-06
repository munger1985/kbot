"""纯计算正式入口：只接受固定 App 的受控源，不创建任何业务数据库连接。"""

from pathlib import Path
import uvicorn
from knowledge_core.config import get_knowledge_core_settings
from knowledge_core.adapters.app_sources import FixedAppSourceRegistry
from knowledge_core.composition.parse_jobs import create_parse_computation_api
from platform_clients.knowledge_computation import ControlledAppSourceClient
from platform_core.logger import LogConfig, LogManager


def create_app(settings=None):
    settings = settings or get_knowledge_core_settings()
    caller = settings.api.service_name
    registry = FixedAppSourceRegistry(caller_service=caller, clients={
        owner: ControlledAppSourceClient(owner=owner, caller_service=caller, base_url=config.base_url,
            audience=config.audience, timeout_seconds=config.timeout_seconds)
        for owner, config in {"knowledge_retrieval": settings.knowledge_retrieval, "aiops": settings.aiops}.items()})
    budgets = settings.compute.model_dump(exclude={"spool_root"})
    app = create_parse_computation_api(audience=caller, spool_root=Path(settings.compute.spool_root),
        source_registry=registry, **budgets)
    app.state.settings = settings
    return app


app = create_app()


if __name__ == "__main__":
    settings = app.state.settings
    LogManager(LogConfig(service="knowledge_core", process="api", log_dir=settings.log.dir,
        level=settings.log.level, rotation=settings.log.rotation, retention=settings.log.retention)).setup()
    uvicorn.run(app, host=settings.api.service_host, port=settings.api.service_port, log_config=None, loop="asyncio")
