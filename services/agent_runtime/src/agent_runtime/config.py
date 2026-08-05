"""Agent Runtime API 与 Worker 配置。"""

from functools import lru_cache
import os

from pydantic import BaseModel, Field

from platform_core.config import (
    ServiceConfig,
    ServiceDependencyConfig,
    Settings,
    load_settings,
)


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
    memory_lease_seconds: int = Field(default=300, ge=60, le=3600)
    retention_poll_interval_seconds: float = Field(
        default=60, ge=10, le=3600
    )
    max_tasks_per_run: int = Field(default=16, ge=1, le=128)
    max_parallel_tasks: int = Field(default=4, ge=1, le=32)
    max_total_retries: int = Field(default=16, ge=0, le=128)
    max_task_timeout_seconds: int = Field(default=600, ge=1, le=3600)


class ConversationAttachmentConfig(BaseModel):
    local_storage_path: str = "./agent_runtime_storage"


class MCPDataConfig(BaseModel):
    """外部 SelectAI/AIReport 问数服务配置。"""

    api_endpoint: str = (
        "http://127.0.0.1:10090/aireport/chat/with_selectai_api"
    )
    profiles_endpoint: str = (
        "http://127.0.0.1:10090/aireport/admin/list_profiles"
    )
    api_key_env: str = "KBOT_MCP_DATA_API_KEY"
    timeout: int = Field(default=120, ge=10, le=600)
    max_rows: int = Field(default=1000, ge=1, le=10000)
    max_response_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1024,
        le=100 * 1024 * 1024,
    )

    def require_api_key(self) -> str:
        value = os.getenv(self.api_key_env)
        if not value:
            raise RuntimeError(
                f"问数 API Key 环境变量 {self.api_key_env} 未设置"
            )
        return value


class AgentRuntimeSettings(Settings):
    api: AgentRuntimeApiConfig = Field(default_factory=AgentRuntimeApiConfig)
    worker: AgentRuntimeWorkerConfig = Field(
        default_factory=AgentRuntimeWorkerConfig
    )
    attachments: ConversationAttachmentConfig = Field(
        default_factory=ConversationAttachmentConfig
    )
    ask_data_api: MCPDataConfig = Field(default_factory=MCPDataConfig)
    knowledge_core: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18090",
            audience="kbot-knowledge-core-api",
            timeout_seconds=120,
        )
    )
    aiops: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18110",
            audience="kbot-aiops-api",
            timeout_seconds=120,
        )
    )
    data_query: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18140",
            audience="kbot-data-query-api",
            timeout_seconds=300,
        )
    )
    llm: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18092",
            audience="kbot-model-llm",
            timeout_seconds=300,
        )
    )
    embedding: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18091",
            audience="kbot-model-embedding",
            timeout_seconds=300,
        )
    )
    vlm: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18094",
            audience="kbot-model-vlm",
            timeout_seconds=300,
        )
    )


@lru_cache(maxsize=1)
def get_agent_runtime_settings() -> AgentRuntimeSettings:
    return load_settings(AgentRuntimeSettings, service="agent_runtime")
