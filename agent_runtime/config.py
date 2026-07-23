"""Agent Runtime API 与 Worker 配置。"""

from functools import lru_cache

from pydantic import BaseModel, Field

from platform_core.config import ServiceConfig, Settings, load_settings


class AgentRuntimeApiConfig(ServiceConfig):
    service_name: str = "kbot-agent-runtime-api"
    service_port: int = 18100


class AgentRuntimeWorkerConfig(BaseModel):
    service_name: str = "kbot-agent-runtime-worker"
    worker_id: str = Field(
        default="agent-runtime-worker-local", min_length=1, max_length=256
    )
    poll_interval_seconds: float = Field(default=1.0, ge=0.1, le=60)
    lease_seconds: int = Field(default=120, ge=15, le=3600)
    max_tasks_per_run: int = Field(default=16, ge=1, le=128)
    max_parallel_tasks: int = Field(default=4, ge=1, le=32)
    max_total_retries: int = Field(default=16, ge=0, le=128)
    max_task_timeout_seconds: int = Field(default=600, ge=1, le=3600)


class AgentRuntimeSettings(Settings):
    api: AgentRuntimeApiConfig = Field(default_factory=AgentRuntimeApiConfig)
    worker: AgentRuntimeWorkerConfig = Field(
        default_factory=AgentRuntimeWorkerConfig
    )


@lru_cache(maxsize=1)
def get_agent_runtime_settings() -> AgentRuntimeSettings:
    return load_settings(AgentRuntimeSettings, service="agent_runtime")
