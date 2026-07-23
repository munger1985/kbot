"""AIOps API、Worker、Scheduler 与 DB Executor 的组合配置。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from platform_core.config import (
    ServiceConfig,
    ServiceDependencyConfig,
    Settings,
    load_settings,
)


class AIOpsApiConfig(ServiceConfig):
    service_name: str = "kbot-aiops-api"
    service_port: int = 18110


class AIOpsRuntimeConfig(BaseModel):
    system_aiops_agent_id: UUID = UUID(
        "019c03b5-4b88-7ab2-8c19-7b6ea34f2a11"
    )

    @model_validator(mode="after")
    def validate_agent_id(self) -> "AIOpsRuntimeConfig":
        if self.system_aiops_agent_id.version != 7:
            raise ValueError("system_aiops_agent_id 必须是 UUIDv7")
        return self


class AIOpsWorkerConfig(ServiceConfig):
    service_name: str = "kbot-aiops-worker"
    service_port: int = 18112
    worker_id: str = Field(
        default="aiops-worker-local", min_length=1, max_length=256
    )
    concurrency: int = Field(default=4, ge=1, le=64)
    claim_interval_seconds: float = Field(default=2.0, ge=0.1, le=60)
    lease_seconds: int = Field(default=120, ge=30, le=3600)
    heartbeat_seconds: int = Field(default=30, ge=1, le=600)


class AIOpsSchedulerConfig(ServiceConfig):
    service_name: str = "kbot-aiops-scheduler"
    service_port: int = 18113
    scheduler_id: str = Field(
        default="aiops-scheduler-local", min_length=1, max_length=256
    )
    scan_interval_seconds: float = Field(default=30, ge=1, le=3600)
    lease_seconds: int = Field(default=120, ge=30, le=3600)


class AIOpsExecutorConfig(ServiceConfig):
    service_name: str = "kbot-aiops-db-executor"
    service_port: int = 18111
    executor_instance_id: str = Field(
        default="aiops-executor-local", min_length=1, max_length=256
    )
    mutation_enabled: bool = False
    readonly_concurrency: int = Field(default=8, ge=1, le=64)
    mutation_concurrency: int = Field(default=1, ge=1, le=1)
    statement_timeout_seconds: int = Field(default=60, ge=1, le=3600)
    max_result_rows: int = Field(default=5000, ge=1, le=100000)
    max_result_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1024,
        le=100 * 1024 * 1024,
    )


class AIOpsDependencyEndpoints(BaseModel):
    agent_runtime: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18100",
            audience="kbot-agent-runtime-api",
            timeout_seconds=120,
        )
    )
    model_serving: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18092",
            audience="kbot-model-llm",
            timeout_seconds=300,
        )
    )
    knowledge_core: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18090",
            audience="kbot-knowledge-core-api",
            timeout_seconds=120,
        )
    )
    aiops_api: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18110",
            audience="kbot-aiops-api",
            timeout_seconds=120,
        )
    )
    db_executor: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18111",
            audience="kbot-aiops-db-executor",
            timeout_seconds=120,
        )
    )


class SecretStoreConfig(BaseModel):
    provider: Literal["environment", "vault", "secret_manager"] = "environment"
    allowed_schemes: tuple[str, ...] = (
        "env",
        "vault",
        "secret-manager",
    )


class AIOpsLimitsConfig(BaseModel):
    max_input_chars: int = Field(default=32000, ge=1, le=200000)
    max_artifact_bytes: int = Field(
        default=20 * 1024 * 1024,
        ge=1024,
        le=200 * 1024 * 1024,
    )
    max_tasks_per_run: int = Field(default=64, ge=1, le=512)
    run_timeout_seconds: int = Field(default=3600, ge=60, le=86400)
    max_targets_per_inspection_fire: int = Field(default=100, ge=1, le=1000)


class AIOpsMonitoringConfig(BaseModel):
    catalog_path: str | None = None
    default_window_seconds: int = Field(default=3600, ge=60, le=604800)
    provider_timeout_seconds: float = Field(default=30, ge=1, le=300)
    webhook_replay_seconds: int = Field(default=300, ge=30, le=3600)
    max_webhook_bytes: int = Field(
        default=1024 * 1024, ge=1024, le=20 * 1024 * 1024
    )
    max_response_bytes: int = Field(
        default=5 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024
    )
    payload_store_root: str = "/tmp/kbot-aiops-monitor-payloads"


class InspectionTemplateRegistration(BaseModel):
    template_id: str = Field(min_length=1, max_length=128)
    template_version: str = Field(min_length=1, max_length=64)
    schedule_resolver_version: str = Field(min_length=1, max_length=64)
    allowed_override_keys: tuple[str, ...] = ()


class AIOpsManagementConfig(BaseModel):
    cursor_secret_env: str = "KBOT_AIOPS_CURSOR_SECRET"
    webhook_key_secret_env: str = "KBOT_AIOPS_WEBHOOK_KEY_SECRET"
    cursor_ttl_seconds: int = Field(default=900, ge=60, le=86400)
    webhook_key_overlap_seconds: int = Field(
        default=3600, ge=0, le=86400
    )
    agent_execution_enabled: bool = False
    inspection_templates: tuple[InspectionTemplateRegistration, ...] = (
        InspectionTemplateRegistration(
            template_id="database_daily",
            template_version="1.0.0",
            schedule_resolver_version="1.0.0",
            allowed_override_keys=(
                "thresholds",
                "window",
                "optional_checks",
            ),
        ),
    )


class AIOpsSettings(Settings):
    api: AIOpsApiConfig = Field(default_factory=AIOpsApiConfig)
    runtime: AIOpsRuntimeConfig = Field(default_factory=AIOpsRuntimeConfig)
    worker: AIOpsWorkerConfig = Field(default_factory=AIOpsWorkerConfig)
    scheduler: AIOpsSchedulerConfig = Field(
        default_factory=AIOpsSchedulerConfig
    )
    executor: AIOpsExecutorConfig = Field(default_factory=AIOpsExecutorConfig)
    clients: AIOpsDependencyEndpoints = Field(
        default_factory=AIOpsDependencyEndpoints
    )
    secret_store: SecretStoreConfig = Field(default_factory=SecretStoreConfig)
    limits: AIOpsLimitsConfig = Field(default_factory=AIOpsLimitsConfig)
    monitoring: AIOpsMonitoringConfig = Field(
        default_factory=AIOpsMonitoringConfig
    )
    management: AIOpsManagementConfig = Field(
        default_factory=AIOpsManagementConfig
    )

    @model_validator(mode="after")
    def validate_aiops_safety(self) -> "AIOpsSettings":
        if self.worker.heartbeat_seconds * 2 >= self.worker.lease_seconds:
            raise ValueError("AIOps Worker heartbeat 必须小于 lease 的一半")
        if self.api.service_name == self.executor.service_name:
            raise ValueError("AIOps API 与 DB Executor 必须使用不同服务身份")
        if self.is_production() and self.secret_store.provider == "environment":
            raise ValueError("生产环境禁止使用 environment Secret Provider")
        if self.is_production() and not self.management.cursor_secret_env:
            raise ValueError("生产环境必须配置 AIOps Cursor Secret 环境变量名")
        if self.is_production() and Path(
            self.monitoring.payload_store_root
        ).resolve().is_relative_to(Path("/tmp")):
            raise ValueError("生产环境监控正文存储不能位于 /tmp")
        for dependency in (
            self.clients.agent_runtime,
            self.clients.model_serving,
            self.clients.knowledge_core,
            self.clients.aiops_api,
            self.clients.db_executor,
        ):
            parsed = urlparse(dependency.base_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    f"AIOps 依赖地址不安全或格式无效：{dependency.base_url}"
                )
        return self


@lru_cache(maxsize=1)
def get_aiops_settings() -> AIOpsSettings:
    return load_settings(AIOpsSettings, service="aiops_agent")
