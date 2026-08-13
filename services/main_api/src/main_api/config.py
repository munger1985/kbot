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
    slack_public_max_webhook_bytes: int = Field(
        default=1024 * 1024, ge=1024, le=20 * 1024 * 1024
    )
    slack_public_requests_per_minute: int = Field(
        default=120, ge=1, le=10000
    )


class DevelopmentLogsConfig(BaseModel):
    """开发日志读取的固定资源上限。"""

    topology_path: str = "resources/topology.toml"
    max_files_per_stream: int = Field(default=8, ge=1, le=64)
    max_bytes_per_file: int = Field(
        default=2 * 1024 * 1024, ge=64 * 1024, le=32 * 1024 * 1024
    )
    max_total_scan_bytes: int = Field(
        default=16 * 1024 * 1024, ge=64 * 1024, le=128 * 1024 * 1024
    )
    max_window_hours: int = Field(default=24 * 31, ge=1, le=24 * 90)
    max_page_size: int = Field(default=500, ge=1, le=2000)
    max_export_events: int = Field(default=2000, ge=1, le=10000)
    max_detail_chars: int = Field(default=65536, ge=1024, le=1024 * 1024)
    max_field_chars: int = Field(default=4096, ge=256, le=65536)


class NotificationConfig(BaseModel):
    """通知投影 Worker 与 SSE 的资源边界。"""

    dispatcher_batch_size: int = Field(default=50, ge=1, le=500)
    dispatcher_lease_seconds: int = Field(default=60, ge=10, le=600)
    dispatcher_max_attempts: int = Field(default=5, ge=1, le=20)
    dispatcher_poll_seconds: float = Field(default=1.0, ge=0.1, le=60)
    sse_poll_interval_seconds: float = Field(default=1.0, ge=0.1, le=10)
    sse_heartbeat_seconds: float = Field(default=15.0, ge=1, le=60)


class MainApiSettings(Settings):
    api: MainApiProcessConfig = Field(default_factory=MainApiProcessConfig)
    knowledge_core: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18090",
            audience="kbot-knowledge-core-api",
        )
    )
    knowledge_retrieval_app: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18150",
            audience="kbot-knowledge-retrieval-app-api",
        )
    )
    km_asset_app: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18160",
            audience="kbot-km-asset-app-api",
        )
    )
    agent_runtime: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18100",
            audience="kbot-agent-runtime-api",
        )
    )
    data_query: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18140",
            audience="kbot-data-query-api",
            timeout_seconds=120,
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
    development_logs: DevelopmentLogsConfig = Field(
        default_factory=DevelopmentLogsConfig
    )
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)


@lru_cache(maxsize=1)
def get_main_api_settings() -> MainApiSettings:
    return load_settings(MainApiSettings, service="main_api")
