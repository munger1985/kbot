"""多媒体创作工作台服务配置。"""

from functools import lru_cache

from pydantic import BaseModel, Field

from platform_core.config import ServiceConfig, ServiceDependencyConfig, Settings, load_settings


class MediaStudioAppApiConfig(ServiceConfig):
    service_name: str = "kbot-media-studio-app-api"
    service_port: int = 18170


class MediaStudioAppWorkerConfig(ServiceConfig):
    service_name: str = "kbot-media-studio-app-worker"
    service_port: int = 18171
    poll_interval_seconds: int = Field(default=5, ge=1, le=60)
    lease_seconds: int = Field(default=360, ge=15, le=600)


class MediaStudioAppStorageConfig(BaseModel):
    local_object_storage_path: str = "var/data/media_studio_app"


class MediaStudioAppSettings(Settings):
    api: MediaStudioAppApiConfig = Field(default_factory=MediaStudioAppApiConfig)
    worker: MediaStudioAppWorkerConfig = Field(default_factory=MediaStudioAppWorkerConfig)
    storage: MediaStudioAppStorageConfig = Field(default_factory=MediaStudioAppStorageConfig)
    llm: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18092",
            audience="kbot-model-llm",
            timeout_seconds=300,
        )
    )


@lru_cache(maxsize=1)
def get_media_studio_app_settings() -> MediaStudioAppSettings:
    return load_settings(MediaStudioAppSettings, service="media_studio_app")
