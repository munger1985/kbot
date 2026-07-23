"""Main API 服务专属配置。"""

from functools import lru_cache

from pydantic import Field

from platform_core.config import (
    ServiceConfig,
    ServiceDependencyConfig,
    Settings,
    load_settings,
)


class MainApiProcessConfig(ServiceConfig):
    service_name: str = "kbot-main-api"
    service_port: int = 18099
    allowed_origins: list[str] = Field(default_factory=list)


class MainApiSettings(Settings):
    api: MainApiProcessConfig = Field(default_factory=MainApiProcessConfig)
    knowledge_core: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18090",
            audience="kbot-knowledge-core-api",
        )
    )


@lru_cache(maxsize=1)
def get_main_api_settings() -> MainApiSettings:
    return load_settings(MainApiSettings, service="main_api")
