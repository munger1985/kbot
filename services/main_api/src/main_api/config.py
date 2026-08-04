"""Main API 服务专属配置。"""

from functools import lru_cache

import os
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

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


class SlackWorkerProcessConfig(ServiceConfig):
    service_name: str = "kbot-main-api-slack-worker"
    service_port: int = 18120
    worker_id: str = Field(
        default="main-api-slack-worker-1", min_length=1, max_length=128
    )


class SlackWorkspaceConfig(BaseModel):
    workspace_id: str = Field(min_length=1, max_length=64)
    domain_id: int = Field(ge=1)
    agent_id: UUID
    security_level: int = Field(default=0, ge=0, le=999)
    signing_secret_env: str = Field(
        default="KBOT_SLACK_SIGNING_SECRET",
        min_length=1,
        max_length=128,
    )
    bot_token_env: str = Field(
        default="KBOT_SLACK_BOT_TOKEN",
        min_length=1,
        max_length=128,
    )

    def require_signing_secret(self) -> str:
        value = os.getenv(self.signing_secret_env)
        if not value:
            raise RuntimeError(
                f"Slack Signing Secret 环境变量 {self.signing_secret_env} 未设置"
            )
        return value

    def require_bot_token(self) -> str:
        value = os.getenv(self.bot_token_env)
        if not value:
            raise RuntimeError(
                f"Slack Bot Token 环境变量 {self.bot_token_env} 未设置"
            )
        return value


class SlackExternalCallbackConfig(BaseModel):
    enabled: bool = False
    url: str = Field(default="", max_length=2048)
    timeout_seconds: int = Field(default=10, ge=1, le=120)

    @model_validator(mode="after")
    def validate_enabled_url(self) -> "SlackExternalCallbackConfig":
        if self.enabled and not self.url.strip():
            raise ValueError("Slack external callback 启用时必须配置 url")
        if self.url and not self.url.startswith(("https://", "http://")):
            raise ValueError("Slack external callback url 必须使用 http 或 https")
        return self


class SlackDebugConfig(BaseModel):
    callback_payload_log_enabled: bool = False
    slack_reply_dump_enabled: bool = False
    slack_reply_dump_dir: str = Field(
        default="/tmp/slackmess", min_length=1, max_length=1024
    )

    @model_validator(mode="after")
    def validate_dump_dir(self) -> "SlackDebugConfig":
        if self.slack_reply_dump_enabled and not os.path.isabs(
            self.slack_reply_dump_dir
        ):
            raise ValueError("Slack 回复调试目录必须使用绝对路径")
        return self


class SlackIntegrationConfig(BaseModel):
    enabled: bool = False
    max_webhook_bytes: int = Field(
        default=1024 * 1024, ge=1024, le=20 * 1024 * 1024
    )
    requests_per_minute: int = Field(default=120, ge=1, le=10000)
    outbox_poll_interval_seconds: float = Field(
        default=1.0, ge=0.1, le=60
    )
    lease_seconds: int = Field(default=60, ge=15, le=600)
    max_delivery_attempts: int = Field(default=8, ge=1, le=100)
    workspaces: list[SlackWorkspaceConfig] = Field(default_factory=list)
    external_callback: SlackExternalCallbackConfig = Field(
        default_factory=SlackExternalCallbackConfig
    )
    debug: SlackDebugConfig = Field(default_factory=SlackDebugConfig)

    @model_validator(mode="after")
    def validate_workspaces(self) -> "SlackIntegrationConfig":
        ids = [item.workspace_id for item in self.workspaces]
        if len(ids) != len(set(ids)):
            raise ValueError("Slack workspace_id 不能重复")
        if self.enabled and not self.workspaces:
            raise ValueError("Slack 启用时至少需要配置一个 workspace")
        return self

    def workspace(self, workspace_id: str) -> SlackWorkspaceConfig | None:
        return next(
            (
                item
                for item in self.workspaces
                if item.workspace_id == workspace_id
            ),
            None,
        )


class MainApiIntegrationConfig(BaseModel):
    monitoring_max_webhook_bytes: int = Field(
        default=1024 * 1024, ge=1024, le=20 * 1024 * 1024
    )
    monitoring_requests_per_minute: int = Field(
        default=120, ge=1, le=10000
    )
    slack: SlackIntegrationConfig = Field(
        default_factory=SlackIntegrationConfig
    )


class MainApiSettings(Settings):
    api: MainApiProcessConfig = Field(default_factory=MainApiProcessConfig)
    slack_worker: SlackWorkerProcessConfig = Field(
        default_factory=SlackWorkerProcessConfig
    )
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
