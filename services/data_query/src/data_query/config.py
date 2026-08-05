"""Data Query API 与 Worker 配置。"""

from functools import lru_cache

from pydantic import Field

from platform_core.config import ServiceConfig, ServiceDependencyConfig, Settings, load_settings


class DataQueryApiConfig(ServiceConfig):
    service_name: str = "kbot-data-query-api"
    service_port: int = 18140


class DataQueryWorkerConfig(ServiceConfig):
    service_name: str = "kbot-data-query-worker"
    service_port: int = 18141
    worker_id: str = Field(
        default="data-query-worker-local", min_length=1, max_length=256
    )
    concurrency: int = Field(default=2, ge=1, le=64)
    claim_interval_seconds: float = Field(default=2.0, ge=0.1, le=60)
    lease_seconds: int = Field(default=120, ge=30, le=3600)
    heartbeat_seconds: int = Field(default=30, ge=1, le=600)
    result_availability_hours: int = Field(default=24, ge=1, le=168)
    result_expiry_sweep_interval_seconds: float = Field(
        default=3600.0, ge=60.0, le=86400.0
    )
    result_expiry_batch_size: int = Field(default=100, ge=1, le=1000)


class DataQuerySettings(Settings):
    api: DataQueryApiConfig = Field(default_factory=DataQueryApiConfig)
    worker: DataQueryWorkerConfig = Field(default_factory=DataQueryWorkerConfig)
    llm: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18092",
            audience="kbot-model-llm",
            timeout_seconds=300,
        )
    )

@lru_cache(maxsize=1)
def get_data_query_settings() -> DataQuerySettings:
    return load_settings(DataQuerySettings, service="data_query")
