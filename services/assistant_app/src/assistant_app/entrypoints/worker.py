"""智能工作台 X Search / 文生图 Worker。"""

import asyncio

from assistant_app.adapters import AssistantLocalObjectStore
from assistant_app.application import AssistantRunWorker
from assistant_app.config import get_assistant_app_settings
from assistant_app.persistence import create_assistant_app_uow
from platform_clients import GenerativeResponsesClient
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager


async def main() -> None:
    settings = get_assistant_app_settings()
    LogManager(LogConfig(
        service="assistant_app", process="worker", log_dir=settings.log.dir,
        level=settings.log.level, rotation=settings.log.rotation,
        retention=settings.log.retention,
    )).setup()
    runtime = create_database_runtime(settings)
    worker = AssistantRunWorker(
        uow_factory=create_assistant_app_uow(runtime.session_factory),
        generative_client=GenerativeResponsesClient(
            base_url=settings.llm.base_url,
            caller_service=settings.worker.service_name,
            audience=settings.llm.audience,
            timeout_seconds=settings.llm.timeout_seconds,
        ),
        object_store=AssistantLocalObjectStore(settings.storage.local_object_storage_path),
        poll_seconds=settings.worker.poll_interval_seconds,
        lease_seconds=settings.worker.lease_seconds,
    )
    try:
        await worker.run_forever()
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
