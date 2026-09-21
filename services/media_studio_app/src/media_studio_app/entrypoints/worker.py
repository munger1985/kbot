"""多媒体创作工作台 图片生成 Worker。"""

import asyncio

from media_studio_app.adapters import MediaStudioLocalObjectStore
from media_studio_app.application import MediaStudioRunWorker
from media_studio_app.config import get_media_studio_app_settings
from media_studio_app.persistence import create_media_studio_app_uow
from platform_clients import GenerativeResponsesClient
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager


async def main() -> None:
    settings = get_media_studio_app_settings()
    LogManager(LogConfig(
        service="media_studio_app", process="worker", log_dir=settings.log.dir,
        level=settings.log.level, rotation=settings.log.rotation,
        retention=settings.log.retention,
    )).setup()
    runtime = create_database_runtime(settings)
    worker = MediaStudioRunWorker(
        uow_factory=create_media_studio_app_uow(runtime.session_factory),
        generative_client=GenerativeResponsesClient(
            base_url=settings.llm.base_url,
            caller_service=settings.worker.service_name,
            audience=settings.llm.audience,
            timeout_seconds=settings.llm.timeout_seconds,
        ),
        object_store=MediaStudioLocalObjectStore(settings.storage.local_object_storage_path),
        poll_seconds=settings.worker.poll_interval_seconds,
        lease_seconds=settings.worker.lease_seconds,
    )
    try:
        await worker.run_forever()
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
