"""AIOps 配置资源的公开与内部共享 Wire 契约。"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, HttpUrl, model_validator

from .types import (
    AIOpsContract,
    CursorPage,
    DatabaseType,
    JsonObject,
    PUBLIC_SCHEMA_VERSION,
    SecretRef,
    UUIDv7,
    UtcDatetime,
)


TargetStatus = Literal["ACTIVE", "MAINTENANCE", "DISABLED"]
HealthStatus = Literal["UNKNOWN", "HEALTHY", "DEGRADED", "UNREACHABLE"]
BindingStatus = Literal["ACTIVE", "REVOKED"]
MonitorStatus = Literal["ACTIVE", "DISABLED"]
PolicyStatus = Literal["DRAFT", "ACTIVE", "RETIRED"]
InspectionPlanStatus = Literal["ACTIVE", "PAUSED", "DISABLED"]
InspectionTargetStatus = Literal["ACTIVE", "DISABLED"]


def _validate_monitor_capabilities(capabilities: JsonObject | None) -> None:
    if capabilities is None:
        return
    label = capabilities.get("external_target_label")
    if label is not None and (
        not isinstance(label, str)
        or re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]{0,127}", label)
        is None
    ):
        raise ValueError("external_target_label 不是合法的监控标签名")


class SecretRefStatus(AIOpsContract):
    """只暴露 Secret 引用是否配置及不可逆指纹。"""

    configured: bool
    provider: str | None = None
    fingerprint: str | None = None


class DatabaseCredentialInput(AIOpsContract):
    username: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=4096)


class DatabaseCredentialStatus(AIOpsContract):
    configured: bool
    credential_id: UUIDv7 | None = None
    key_version: str | None = None
    updated_at: UtcDatetime | None = None


class TargetEndpoint(AIOpsContract):
    host: str = Field(
        min_length=1,
        max_length=253,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9.-]*$",
    )
    port: int = Field(ge=1, le=65535)
    service: str | None = Field(default=None, min_length=1, max_length=256)
    database: str | None = Field(default=None, min_length=1, max_length=256)
    tls_enabled: bool = True


class TargetCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str = Field(min_length=1, max_length=256)
    db_type: DatabaseType
    version_code: str | None = Field(default=None, max_length=64)
    environment: Literal["PROD", "STG", "DEV"]
    db_role: Literal["PRIMARY", "STANDBY", "UNKNOWN"] = "UNKNOWN"
    endpoint: TargetEndpoint | None = None
    diagnostic_credential: DatabaseCredentialInput | None = None
    execution_credential: DatabaseCredentialInput | None = None
    security_level: int = Field(default=1, ge=0, le=999)
    capabilities: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_database_endpoint(self) -> "TargetCreate":
        if self.endpoint is None:
            return self
        if self.db_type == DatabaseType.ORACLE:
            if not self.endpoint.service or self.endpoint.database:
                raise ValueError("Oracle Endpoint 必须只设置 service")
        elif not self.endpoint.database or self.endpoint.service:
            raise ValueError("MySQL Endpoint 必须只设置 database")
        return self


class TargetPatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    version_code: str | None = Field(default=None, max_length=64)
    environment: Literal["PROD", "STG", "DEV"] | None = None
    db_role: Literal["PRIMARY", "STANDBY", "UNKNOWN"] | None = None
    endpoint: TargetEndpoint | None = None
    security_level: int | None = Field(default=None, ge=0, le=999)
    capabilities: JsonObject | None = None


class TargetSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    target_id: UUIDv7
    display_name: str
    db_type: DatabaseType
    environment: str
    status: TargetStatus
    health_status: HealthStatus
    row_version: int = Field(ge=1)
    updated_at: UtcDatetime


class TargetDetail(TargetSummary):
    version_code: str | None = None
    db_role: str
    endpoint: TargetEndpoint | None = None
    diagnostic_credential: DatabaseCredentialStatus
    execution_credential: DatabaseCredentialStatus
    security_level: int
    capabilities: JsonObject
    health_version: int = Field(ge=1)
    last_health_check_at: UtcDatetime | None = None
    last_error_code: str | None = None
    created_at: UtcDatetime
    created_by: str
    updated_by: str


class TargetPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[TargetSummary, ...] = ()


class AgentBindingCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    agent_id: UUIDv7
    allow_mutation: bool = False
    policy_id: UUIDv7 | None = None
    allowed_actions: tuple[str, ...] = ()
    change_window: JsonObject | None = None
    max_daily_executions: int | None = Field(default=None, ge=0)


class AgentBindingPatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    allow_mutation: bool | None = None
    policy_id: UUIDv7 | None = None
    allowed_actions: tuple[str, ...] | None = None
    change_window: JsonObject | None = None
    max_daily_executions: int | None = Field(default=None, ge=0)


class AgentBindingView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    binding_id: UUIDv7
    target_id: UUIDv7
    agent_id: UUIDv7
    allow_mutation: bool
    policy_id: UUIDv7 | None = None
    allowed_actions: tuple[str, ...] = ()
    change_window: JsonObject | None = None
    max_daily_executions: int | None = None
    status: BindingStatus
    row_version: int = Field(ge=1)
    created_at: UtcDatetime
    updated_at: UtcDatetime


class MonitorSourceCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str = Field(min_length=1, max_length=256)
    source_type: Literal["PROMETHEUS", "ZABBIX", "OEM"]
    endpoint: HttpUrl
    secret_ref: SecretRef | None = None
    webhook_secret_ref: SecretRef | None = None
    tls_profile_ref: SecretRef | None = None
    capabilities: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_endpoint(self) -> "MonitorSourceCreate":
        if (
            self.endpoint.username
            or self.endpoint.password
            or self.endpoint.query
            or self.endpoint.fragment
        ):
            raise ValueError("Monitor Endpoint 不允许凭证、Query 或 Fragment")
        _validate_monitor_capabilities(self.capabilities)
        return self


class MonitorSourcePatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    endpoint: HttpUrl | None = None
    secret_ref: SecretRef | None = None
    webhook_secret_ref: SecretRef | None = None
    tls_profile_ref: SecretRef | None = None
    capabilities: JsonObject | None = None

    @model_validator(mode="after")
    def validate_endpoint(self) -> "MonitorSourcePatch":
        if self.endpoint is not None and (
            self.endpoint.username
            or self.endpoint.password
            or self.endpoint.query
            or self.endpoint.fragment
        ):
            raise ValueError("Monitor Endpoint 不允许凭证、Query 或 Fragment")
        _validate_monitor_capabilities(self.capabilities)
        return self


class MonitorSourceSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    source_id: UUIDv7
    display_name: str
    source_type: str
    status: MonitorStatus
    health_status: HealthStatus
    health_check_pending: bool
    row_version: int = Field(ge=1)
    updated_at: UtcDatetime


class MonitorSourceDetail(MonitorSourceSummary):
    endpoint: str
    secret: SecretRefStatus
    webhook_secret: SecretRefStatus
    tls_profile: SecretRefStatus
    capabilities: JsonObject
    webhook_configured: bool
    health_version: int = Field(ge=1)
    last_health_check_at: UtcDatetime | None = None
    last_error_code: str | None = None
    created_at: UtcDatetime
    created_by: str
    updated_by: str


class MonitorSourcePage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[MonitorSourceSummary, ...] = ()


class MonitorBindingCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    source_id: UUIDv7
    external_target_key: str = Field(min_length=1, max_length=256)
    role: Literal["PRIMARY", "SUPPLEMENTARY"] = "PRIMARY"
    priority: int = Field(default=100, ge=0)
    metric_scope: JsonObject | None = None
    mapping_overrides: JsonObject | None = None


class MonitorBindingPatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    external_target_key: str | None = Field(
        default=None, min_length=1, max_length=256
    )
    role: Literal["PRIMARY", "SUPPLEMENTARY"] | None = None
    priority: int | None = Field(default=None, ge=0)
    metric_scope: JsonObject | None = None
    mapping_overrides: JsonObject | None = None


class MonitorBindingView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    binding_id: UUIDv7
    target_id: UUIDv7
    source_id: UUIDv7
    external_target_key: str
    role: str
    priority: int
    metric_scope: JsonObject | None = None
    mapping_overrides: JsonObject | None = None
    status: MonitorStatus
    health_status: HealthStatus
    row_version: int = Field(ge=1)
    created_at: UtcDatetime
    updated_at: UtcDatetime


class PolicyCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    policy_key: str = Field(pattern=r"^[a-z][a-z0-9._-]{0,127}$")
    display_name: str = Field(min_length=1, max_length=256)
    rules: JsonObject


class PolicySummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    policy_id: UUIDv7
    policy_key: str
    version_no: int = Field(ge=1)
    display_name: str
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: PolicyStatus
    row_version: int = Field(ge=1)
    updated_at: UtcDatetime


class PolicyDetail(PolicySummary):
    rules: JsonObject
    effective_at: UtcDatetime | None = None
    retired_at: UtcDatetime | None = None
    created_at: UtcDatetime
    created_by: str
    updated_by: str


class PolicyPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[PolicySummary, ...] = ()


class InspectionPlanCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str = Field(min_length=1, max_length=256)
    schedule_type: Literal["DAILY", "WEEKLY", "CRON"]
    cron_expression: str = Field(min_length=9, max_length=256)
    timezone: str = Field(min_length=1, max_length=64)
    template_id: str = Field(min_length=1, max_length=128)
    template_version: str = Field(min_length=1, max_length=64)
    timeout_seconds: int = Field(ge=1, le=86400)
    overlap_policy: Literal["SKIP", "QUEUE"] = "SKIP"
    misfire_policy: Literal["SKIP", "LATEST_ONLY"] = "LATEST_ONLY"
    schedule_resolver_version: str = Field(min_length=1, max_length=64)


class InspectionPlanPatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    cron_expression: str | None = Field(default=None, min_length=9, max_length=256)
    timezone: str | None = Field(default=None, min_length=1, max_length=64)
    template_id: str | None = Field(default=None, min_length=1, max_length=128)
    template_version: str | None = Field(default=None, min_length=1, max_length=64)
    timeout_seconds: int | None = Field(default=None, ge=1, le=86400)
    overlap_policy: Literal["SKIP", "QUEUE"] | None = None
    misfire_policy: Literal["SKIP", "LATEST_ONLY"] | None = None
    schedule_resolver_version: str | None = Field(
        default=None, min_length=1, max_length=64
    )


class InspectionPlanSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    plan_id: UUIDv7
    display_name: str
    schedule_type: str
    timezone: str
    status: InspectionPlanStatus
    next_run_at: UtcDatetime | None = None
    row_version: int = Field(ge=1)
    updated_at: UtcDatetime


class InspectionPlanDetail(InspectionPlanSummary):
    cron_expression: str
    template_id: str
    template_version: str
    timeout_seconds: int
    overlap_policy: str
    misfire_policy: str
    schedule_resolver_version: str
    active_target_count: int = Field(ge=0)
    created_at: UtcDatetime
    created_by: str
    updated_by: str


class InspectionPlanPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[InspectionPlanSummary, ...] = ()


class InspectionTargetCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    target_id: UUIDv7
    template_overrides: JsonObject | None = None


class InspectionTargetPatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    template_overrides: JsonObject | None = None
    status: InspectionTargetStatus | None = None


class InspectionTargetView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    plan_target_id: UUIDv7
    plan_id: UUIDv7
    target_id: UUIDv7
    template_overrides: JsonObject | None = None
    status: InspectionTargetStatus
    created_at: UtcDatetime
    updated_at: UtcDatetime


class HealthCheckReceipt(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    source_id: UUIDv7
    request_id: UUIDv7
    accepted_at: UtcDatetime
    config_row_version: int = Field(ge=1)
    health_version: int = Field(ge=1)


class WebhookKeyRotation(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    source_id: UUIDv7
    webhook_key: str = Field(min_length=32, max_length=256)
    previous_key_expires_at: UtcDatetime | None = None
    created_at: UtcDatetime


class ConfigListQuery(AIOpsContract):
    """内部 Client 对列表过滤条件的规范表达。"""

    status: str | None = Field(default=None, max_length=32)
    cursor: str | None = Field(default=None, max_length=2048)
    limit: int = Field(default=50, ge=1, le=200)


class ConfigCommandReceipt(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    resource_id: UUIDv7
    status: str
    row_version: int = Field(ge=1)
    accepted_at: UtcDatetime
