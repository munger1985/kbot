"""Provider 无关的实时监控跨服务合同。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


MonitoringReadiness = Literal[
    "READY",
    "DISABLED",
    "DISCONNECTED",
    "CAPABILITY_MISSING",
    "NO_MAPPED_INSTANCE",
]
DiagnosticReadiness = Literal["READY", "PARTIAL", "UNAVAILABLE"]
MonitoringWindowName = Literal["15m", "1h", "6h", "24h"]
GrafanaDashboardUid = Literal[
    "kbot-database-fleet",
    "kbot-oracle-overview",
    "kbot-oracle-storage",
    "kbot-oracle-alerts",
    "kbot-oracle-alert-log",
    "kbot-mysql-overview",
    "kbot-postgresql-overview",
    "kbot-host-overview",
]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MonitoringGap(_Contract):
    scope: Literal["SOURCE", "INSTANCE", "METRIC"]
    code: str = Field(min_length=1, max_length=64)
    detail: str = Field(min_length=1, max_length=1000)
    source_id: UUID | None = None
    instance_id: UUID | None = None
    metric_code: str | None = None
    retryable: bool = False

    @model_validator(mode="after")
    def validate_scope(self):
        if self.scope in {"INSTANCE", "METRIC"} and self.instance_id is None:
            raise ValueError("实例或指标级 Gap 必须包含 instance_id")
        if self.scope == "METRIC" and not self.metric_code:
            raise ValueError("指标级 Gap 必须包含 metric_code")
        return self


class MonitoringSourceSummary(_Contract):
    source_id: UUID
    display_name: str = Field(min_length=1, max_length=256)
    source_type: Literal["PROMETHEUS", "ZABBIX"]
    monitoring_readiness: MonitoringReadiness
    diagnostic_readiness: DiagnosticReadiness
    capability_gaps: tuple[MonitoringGap, ...] = ()


class MonitoringInstanceSummary(_Contract):
    instance_id: UUID
    display_name: str = Field(min_length=1, max_length=256)
    db_type: Literal["ORACLE", "MYSQL", "POSTGRESQL"]
    status: Literal["AVAILABLE", "UNAVAILABLE", "PARTIAL"]
    monitoring_readiness: MonitoringReadiness
    diagnostic_readiness: DiagnosticReadiness
    capability_gaps: tuple[MonitoringGap, ...] = ()


class MonitoringProfileSummary(_Contract):
    profile_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    version: str = Field(min_length=1, max_length=32)
    display_name: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=1000)
    supported_db_types: tuple[
        Literal["ORACLE", "MYSQL", "POSTGRESQL"], ...
    ]
    metric_codes: tuple[str, ...] = Field(min_length=1, max_length=8)
    grafana_dashboard_uid: GrafanaDashboardUid | None = None


class MonitoringDashboardIntegration(_Contract):
    integration_mode: Literal["LINK", "DISABLED"]
    dashboard_uid: GrafanaDashboardUid | None = None
    launch_url: str | None = Field(default=None, min_length=1, max_length=2048)
    reason_code: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_mode(self):
        if self.integration_mode == "LINK":
            if self.dashboard_uid is None or self.launch_url is None:
                raise ValueError("LINK 模式必须包含固定 Dashboard UID 与受控入口")
        elif self.launch_url is not None:
            raise ValueError("DISABLED 模式不能返回 Grafana 入口")
        return self


class MonitoringWindow(_Contract):
    start: datetime
    end: datetime
    name: MonitoringWindowName | None = None

    @model_validator(mode="after")
    def validate_window(self):
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("监控时间必须包含时区")
        if self.start >= self.end:
            raise ValueError("监控时间窗无效")
        if (self.end - self.start).total_seconds() > 86_400:
            raise ValueError("监控时间窗不能超过 24 小时")
        return self


class MonitoringPoint(_Contract):
    observed_at: datetime
    value: float | str | bool | None
    quality: Literal["GOOD", "INVALID", "STALE", "ESTIMATED"] = "GOOD"


class MonitoringSeries(_Contract):
    source_id: UUID
    source_display_name: str = Field(min_length=1, max_length=256)
    source_type: Literal["PROMETHEUS", "ZABBIX"]
    instance_id: UUID
    instance_display_name: str = Field(min_length=1, max_length=256)
    series_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    dimensions: dict[str, str] = Field(default_factory=dict)
    points: tuple[MonitoringPoint, ...] = Field(default=(), max_length=240)
    coverage_ratio: float = Field(ge=0, le=1)


class MonitoringPanel(_Contract):
    panel_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,127}$")
    title: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=1000)
    visualization: Literal["STAT", "STATE_TIMELINE", "TIME_SERIES", "GAUGE"]
    metric_code: str
    unit: str = Field(min_length=1, max_length=32)
    value_kind: Literal["GAUGE", "COUNTER", "STATE"]
    series: tuple[MonitoringSeries, ...] = ()
    summary: dict[str, float | int | str | None] = Field(default_factory=dict)
    quality: Literal["GOOD", "PARTIAL", "NO_DATA"]


class MonitoringView(_Contract):
    schema_version: Literal["aiops.public.v1"] = "aiops.public.v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    refresh_after_seconds: int = Field(default=30, ge=15, le=300)
    source: MonitoringSourceSummary
    compare_source: MonitoringSourceSummary | None = None
    dashboard: MonitoringDashboardIntegration | None = None
    profile: MonitoringProfileSummary
    window: MonitoringWindow
    instances: tuple[MonitoringInstanceSummary, ...] = Field(
        min_length=1, max_length=12
    )
    panels: tuple[MonitoringPanel, ...] = Field(default=(), max_length=8)
    gaps: tuple[MonitoringGap, ...] = ()
    partial: bool


class MonitoringViewRequest(_Contract):
    profile_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    instance_ids: tuple[UUID, ...] = Field(min_length=1, max_length=12)
    window: MonitoringWindowName

    @model_validator(mode="after")
    def validate_instances(self):
        if len(set(self.instance_ids)) != len(self.instance_ids):
            raise ValueError("instance_ids 不能重复")
        return self
