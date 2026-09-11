"""智能工作台服务配置。"""

from functools import lru_cache

from pydantic import Field

from platform_core.config import ServiceConfig, Settings, load_settings


class AssistantAppApiConfig(ServiceConfig):
    service_name: str = "kbot-assistant-app-api"
    service_port: int = 18170


class AssistantAppSettings(Settings):
    api: AssistantAppApiConfig = Field(default_factory=AssistantAppApiConfig)


@lru_cache(maxsize=1)
def get_assistant_app_settings() -> AssistantAppSettings:
    return load_settings(AssistantAppSettings, service="assistant_app")
