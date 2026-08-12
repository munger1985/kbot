"""KM Asset 持久任务 Worker。"""

import asyncio

from km_asset_app.application import KmAssetService, KmCredentialService
from km_asset_app.application.worker import KmAssetWorker
from km_asset_app.config import get_km_asset_settings
from km_asset_app.persistence import create_km_asset_uow
from platform_clients import KnowledgeCoreClient
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager
from platform_core.managed_credentials import ManagedCredentialCipher


async def main() -> None:
    settings = get_km_asset_settings()
    LogManager(LogConfig(service="km_asset_app", process="worker", log_dir=settings.log.dir, level=settings.log.level, rotation=settings.log.rotation, retention=settings.log.retention)).setup()
    runtime = create_database_runtime(settings)
    uow_factory = create_km_asset_uow(runtime.session_factory)
    credentials = KmCredentialService(cipher=ManagedCredentialCipher.from_environment())
    asset_service = KmAssetService(uow_factory=uow_factory, credential_service=credentials)
    worker = KmAssetWorker(uow_factory=uow_factory, credential_service=credentials, asset_service=asset_service, knowledge_core_client=KnowledgeCoreClient(base_url=settings.knowledge_core.base_url, caller_service=settings.worker.service_name, audience=settings.knowledge_core.audience, timeout_seconds=settings.knowledge_core.timeout_seconds), poll_seconds=settings.worker.poll_interval_seconds, lease_seconds=settings.worker.lease_seconds)
    try:
        await worker.run_forever()
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
