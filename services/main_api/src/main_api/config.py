"""Main API 服务专属配置。"""

from functools import lru_cache

from pydantic import BaseModel, Field

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
    test_auth_bypass_enabled: bool = False
    docs_enabled: bool = False
    sse_poll_interval_seconds: float = Field(default=0.5, ge=0.1, le=10)
    sse_heartbeat_seconds: float = Field(default=15, ge=1, le=60)
    sse_batch_size: int = Field(default=200, ge=1, le=500)


class MainApiIntegrationConfig(BaseModel):
    monitoring_max_webhook_bytes: int = Field(
        default=1024 * 1024, ge=1024, le=20 * 1024 * 1024
    )
    monitoring_requests_per_minute: int = Field(
        default=120, ge=1, le=10000
    )


class MainApiSettings(Settings):
    api: MainApiProcessConfig = Field(default_factory=MainApiProcessConfig)
    knowledge_core: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18090",
            audience="kbot-knowledge-core-api",
        )
    )
    agent_runtime: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18100",
            audience="kbot-agent-runtime-api",
        )
    )
    aiops: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18110",
            audience="kbot-aiops-api",
        )
    )
    model_embedding: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18091",
            audience="kbot-model-embedding",
        )
    )
    model_llm: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18092",
            audience="kbot-model-llm",
        )
    )
    model_visual: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18093",
            audience="kbot-model-visual",
        )
    )
    model_vlm: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18094",
            audience="kbot-model-vlm",
        )
    )
    integrations: MainApiIntegrationConfig = Field(
        default_factory=MainApiIntegrationConfig
    )


@lru_cache(maxsize=1)
def get_main_api_settings() -> MainApiSettings:
    return load_settings(MainApiSettings, service="main_api")
