"""智能工作台服务配置。"""

from functools import lru_cache

from pydantic import BaseModel, Field

from platform_core.config import ServiceConfig, ServiceDependencyConfig, Settings, load_settings


class AssistantAppApiConfig(ServiceConfig):
    service_name: str = "kbot-assistant-app-api"
    service_port: int = 18170


class AssistantAppWorkerConfig(ServiceConfig):
    service_name: str = "kbot-assistant-app-worker"
    service_port: int = 18171
    poll_interval_seconds: int = Field(default=5, ge=1, le=60)
    lease_seconds: int = Field(default=360, ge=15, le=600)


class AssistantAppStorageConfig(BaseModel):
    local_object_storage_path: str = "var/data/assistant_app"


class AssistantAppSettings(Settings):
    api: AssistantAppApiConfig = Field(default_factory=AssistantAppApiConfig)
    worker: AssistantAppWorkerConfig = Field(default_factory=AssistantAppWorkerConfig)
    storage: AssistantAppStorageConfig = Field(default_factory=AssistantAppStorageConfig)
    llm: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18092",
            audience="kbot-model-llm",
            timeout_seconds=300,
        )
    )


@lru_cache(maxsize=1)
def get_assistant_app_settings() -> AssistantAppSettings:
    return load_settings(AssistantAppSettings, service="assistant_app")
