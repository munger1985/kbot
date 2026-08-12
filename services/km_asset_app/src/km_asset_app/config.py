"""KM Asset App 配置。"""

from functools import lru_cache

from pydantic import Field

from platform_core.config import ServiceConfig, ServiceDependencyConfig, Settings, load_settings


class KmAssetApiConfig(ServiceConfig):
    service_name: str = "kbot-km-asset-app-api"
    service_port: int = 18160


class KmAssetWorkerConfig(ServiceConfig):
    service_name: str = "kbot-km-asset-app-worker"
    service_port: int = 18161
    poll_interval_seconds: int = 5
    lease_seconds: int = 120


class KmAssetAppSettings(Settings):
    api: KmAssetApiConfig = Field(default_factory=KmAssetApiConfig)
    worker: KmAssetWorkerConfig = Field(default_factory=KmAssetWorkerConfig)
    knowledge_core: ServiceDependencyConfig = Field(default_factory=lambda: ServiceDependencyConfig(base_url="http://127.0.0.1:18090", audience="kbot-knowledge-core-api", timeout_seconds=300))
    data_query: ServiceDependencyConfig = Field(default_factory=lambda: ServiceDependencyConfig(base_url="http://127.0.0.1:18140", audience="kbot-data-query-api", timeout_seconds=300))


@lru_cache(maxsize=1)
def get_km_asset_settings() -> KmAssetAppSettings:
    return load_settings(KmAssetAppSettings, service="km_asset_app")
