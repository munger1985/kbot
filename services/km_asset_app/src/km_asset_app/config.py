"""KM Asset App 配置。"""

import os
from functools import lru_cache
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from platform_core.config import (
    ServiceConfig,
    ServiceDependencyConfig,
    Settings,
    load_settings,
)


class KmAssetApiConfig(ServiceConfig):
    service_name: str = "kbot-km-asset-app-api"
    service_port: int = 18160


class KmAssetWorkerConfig(ServiceConfig):
    service_name: str = "kbot-km-asset-app-worker"
    service_port: int = 18161
    poll_interval_seconds: int = 5
    lease_seconds: int = 120


class SlackWorkerConfig(BaseModel):
    service_name: str = "kbot-km-asset-app-slack-worker"
    worker_id: str = Field(
        default="km-asset-slack-worker-1",
        min_length=1,
        max_length=128,
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
        default="/tmp/slackmess",
        min_length=1,
        max_length=1024,
    )

    @model_validator(mode="after")
    def validate_dump_dir(self) -> "SlackDebugConfig":
        if self.slack_reply_dump_enabled and not os.path.isabs(
            self.slack_reply_dump_dir
        ):
            raise ValueError("Slack 回复调试目录必须使用绝对路径")
        return self


class SlackReplyConfig(BaseModel):
    assistant_name: str = Field(
        default="Asset问答助手",
        min_length=1,
        max_length=80,
    )
    max_references: int = Field(default=5, ge=0, le=10)
    show_warnings: bool = True
    show_query_result_summary: bool = True
    show_visualization_notice: bool = True


class SlackIntegrationConfig(BaseModel):
    enabled: bool = False
    max_webhook_bytes: int = Field(
        default=1024 * 1024,
        ge=1024,
        le=20 * 1024 * 1024,
    )
    requests_per_minute: int = Field(default=120, ge=1, le=10000)
    outbox_poll_interval_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60,
    )
    lease_seconds: int = Field(default=60, ge=15, le=600)
    max_delivery_attempts: int = Field(default=8, ge=1, le=100)
    workspaces: list[SlackWorkspaceConfig] = Field(default_factory=list)
    external_callback: SlackExternalCallbackConfig = Field(
        default_factory=SlackExternalCallbackConfig
    )
    debug: SlackDebugConfig = Field(default_factory=SlackDebugConfig)
    reply: SlackReplyConfig = Field(default_factory=SlackReplyConfig)

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


class KmAssetIntegrationConfig(BaseModel):
    slack: SlackIntegrationConfig = Field(
        default_factory=SlackIntegrationConfig
    )


class KmAssetAppSettings(Settings):
    api: KmAssetApiConfig = Field(default_factory=KmAssetApiConfig)
    worker: KmAssetWorkerConfig = Field(default_factory=KmAssetWorkerConfig)
    slack_worker: SlackWorkerConfig = Field(default_factory=SlackWorkerConfig)
    integrations: KmAssetIntegrationConfig = Field(
        default_factory=KmAssetIntegrationConfig
    )
    knowledge_core: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18090",
            audience="kbot-knowledge-core-api",
            timeout_seconds=300,
        )
    )
    data_query: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18140",
            audience="kbot-data-query-api",
            timeout_seconds=300,
        )
    )
    agent_runtime: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18100",
            audience="kbot-agent-runtime-api",
            timeout_seconds=120,
        )
    )


@lru_cache(maxsize=1)
def get_km_asset_settings() -> KmAssetAppSettings:
    return load_settings(KmAssetAppSettings, service="km_asset_app")
