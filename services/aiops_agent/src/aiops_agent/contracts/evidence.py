"""指标、事件与日志诊断证据的类型化契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aiops_agent.domain.evidence.events import (
    SignalEventStatus,
    SignalSeverity,
)
from platform_core.contracts.aiops.types import UtcDatetime


class NormalizedSignalEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_event_key: str = Field(min_length=1, max_length=256)
    source_locator_key: str = Field(min_length=1, max_length=256)
    event_type: str = Field(min_length=1, max_length=64)
    event_status: SignalEventStatus
    severity: SignalSeverity
    occurred_at: UtcDatetime
    fingerprint_basis: str = Field(min_length=1, max_length=512)
    summary: str = Field(min_length=1, max_length=1000)
    provider_attributes: dict[str, Any] = Field(default_factory=dict)
    normalizer_version: str = Field(min_length=1, max_length=64)


class NormalizedSignalBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_delivery_id: str | None = Field(default=None, max_length=256)
    events: tuple[NormalizedSignalEvent, ...] = ()
    warnings: tuple[str, ...] = ()


class ProviderMetricDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    template_id: str = Field(min_length=1, max_length=128)
    template_version: str = Field(min_length=1, max_length=64)
    query_template: str | None = Field(default=None, max_length=2000)
    fallback_query_templates: tuple[str, ...] = Field(
        default=(), max_length=4
    )
    exact_item_key: str | None = Field(default=None, max_length=512)
    value_type: Literal[
        "FLOAT", "CHARACTER", "LOG", "UNSIGNED", "TEXT"
    ] | None = None
    unit: str | None = Field(default=None, max_length=32)
    target_type: str | None = Field(default=None, max_length=128)
    metric_name: str | None = Field(default=None, max_length=256)
    required_labels: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_provider_locator(self) -> "ProviderMetricDefinition":
        if any(
            not query.strip() or len(query) > 2000
            for query in self.fallback_query_templates
        ):
            raise ValueError("Provider 回退查询格式无效")
        if not (
            self.query_template
            or self.exact_item_key
            or (self.target_type and self.metric_name)
        ):
            raise ValueError("Provider 指标必须声明受控查询定位")
        return self


class MetricDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_code: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,127}$")
    semantic_role: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,63}$")
    semantic_version: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=1000)
    unit: str = Field(min_length=1, max_length=32)
    value_kind: Literal["GAUGE", "COUNTER", "STATE"]
    expected_dimensions: tuple[str, ...] = ()
    supported_db_types: tuple[str, ...] = ("ORACLE", "MYSQL")
    allowed_aggregations: tuple[str, ...] = ("AVG", "MAX", "LAST")
    default_window_seconds: int = Field(ge=60, le=5_184_000)
    min_step_seconds: int = Field(ge=1, le=3600)
    max_points: int = Field(ge=2, le=100000)
    max_series: int = Field(ge=1, le=10000)
    providers: dict[
        Literal["PROMETHEUS", "ZABBIX", "OEM"],
        ProviderMetricDefinition,
    ]


class MetricCatalogDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    catalog_version: str = Field(min_length=1, max_length=64)
    metrics: tuple[MetricDefinition, ...]

    @model_validator(mode="after")
    def validate_unique_codes(self) -> "MetricCatalogDocument":
        codes = [item.metric_code for item in self.metrics]
        if len(codes) != len(set(codes)):
            raise ValueError("Metric Catalog 中 metric_code 不能重复")
        return self


class MetricPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observed_at: UtcDatetime
    value: float | str | bool | None
    quality: Literal["GOOD", "INVALID", "STALE", "ESTIMATED"] = "GOOD"


class MetricSeries(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dimensions: dict[str, str] = Field(default_factory=dict)
    points: tuple[MetricPoint, ...] = ()


class ObservationGap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_code: str | None = None
    source_id: str
    binding_id: str
    code: str
    detail: str
    retryable: bool = False


class LogEvidenceEntry(BaseModel):
    """经过限长和来源标记的单条日志证据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    observed_at: UtcDatetime
    line: str = Field(min_length=1, max_length=4000)
    labels: dict[str, str] = Field(default_factory=dict)
    structured_fields: dict[str, str] = Field(default_factory=dict)
    entry_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


class LogEvidenceSet(BaseModel):
    """可直接固化为不可变 Artifact 的日志查询结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["LOG_EVIDENCE_SET.v1"] = "LOG_EVIDENCE_SET.v1"
    target_id: str
    binding_id: str
    source_id: str
    window_start: UtcDatetime
    window_end: UtcDatetime
    entries: tuple[LogEvidenceEntry, ...] = ()
    gaps: tuple[ObservationGap, ...] = ()
    collected_at: UtcDatetime
    truncated: bool = False
    query_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    provenance: dict[str, Any] = Field(default_factory=dict)


class MetricObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_code: str
    semantic_role: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,63}$")
    semantic_version: str
    unit: str
    value_kind: str
    window_start: UtcDatetime
    window_end: UtcDatetime
    requested_step_seconds: int = Field(ge=1)
    effective_step_seconds: int = Field(ge=1)
    source_id: str
    source_type: str
    source_version: int = Field(ge=1)
    target_id: str
    binding_id: str
    external_target_fingerprint: str
    series: tuple[MetricSeries, ...] = ()
    summary: dict[str, float | int | str | None] = Field(
        default_factory=dict
    )
    expected_points: int = Field(ge=0)
    actual_points: int = Field(ge=0)
    coverage_ratio: float = Field(ge=0, le=1)
    truncated: bool = False
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)


class DiscoveredMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=256)
    label_names: tuple[str, ...] = Field(default=(), max_length=128)
    metric_type: str = "UNKNOWN"
    help_text: str | None = None
    unit: str | None = Field(default=None, max_length=64)
    metadata_state: Literal["VERIFIED", "UNKNOWN", "CONFLICT"] = "UNKNOWN"
    native_value_type: str | None = None
    query_supported: bool = False


class NativeHistoryMetadata(BaseModel):
    """保留日志 Item 的原始语义，不将文本、日志等级当作数值指标。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nanoseconds: int = Field(ge=0, le=999999999)
    log_timestamp_seconds: int | None = Field(default=None, ge=0)
    log_severity: str | None = Field(default=None, max_length=32)
    log_source: str | None = Field(default=None, max_length=1024)
    log_event_id: str | None = Field(default=None, max_length=128)


class NativeMetricSamples(BaseModel):
    """原生历史采样；数字类型不推定为计数器，也不自动生成速率。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str
    target_id: str
    binding_id: str
    source_id: str
    metric_name: str
    native_value_type: str
    sample_kind: Literal["NUMERIC", "TEXT", "LOG"] = "NUMERIC"
    sample_metadata: tuple[NativeHistoryMetadata, ...] = Field(default=(), max_length=1000)
    source_version: str = Field(min_length=1)
    unit: str | None = None
    window_start: UtcDatetime
    window_end: UtcDatetime
    captured_at: UtcDatetime
    points: tuple[tuple[UtcDatetime, str], ...] = Field(max_length=1000)
    truncated: bool = False
    response_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_samples(self):
        if self.sample_metadata and len(self.sample_metadata) != len(self.points):
            raise ValueError("原生历史元数据必须逐条对应采样")
        if self.sample_kind == "LOG" and len(self.sample_metadata) != len(self.points):
            raise ValueError("日志历史必须保留逐条日志元数据")
        if any(not self.window_start <= stamp <= self.window_end or len(value) > 8192 for stamp, value in self.points):
            raise ValueError("原生采样时间或文本长度越界")
        return self


class MetricDiscoveryResult(BaseModel):
    """限定目标与窗口的目录，不是数值采样或全量覆盖证明。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str
    target_id: str
    binding_id: str
    source_id: str
    scope: Literal["DATABASE", "HOST"]
    window_start: UtcDatetime
    window_end: UtcDatetime
    captured_at: UtcDatetime
    entries: tuple[DiscoveredMetric, ...] = Field(default=(), max_length=100)
    truncated: bool = False
    response_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    warnings: tuple[str, ...] = ()


class ObservationSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["OBSERVATION_SET.v1"] = "OBSERVATION_SET.v1"
    target_id: str
    binding_id: str
    source_id: str
    observations: tuple[MetricObservation, ...] = ()
    metric_catalogs: tuple[MetricDiscoveryResult, ...] = Field(default=(), max_length=1)
    native_metrics: tuple[NativeMetricSamples, ...] = Field(default=(), max_length=4)
    active_alerts: tuple[dict[str, Any], ...] = ()
    gaps: tuple[ObservationGap, ...] = ()
    collected_at: UtcDatetime

    @model_validator(mode="after")
    def validate_catalog_scope(self):
        for samples in self.native_metrics:
            if str(samples.target_id) != str(self.target_id) or str(samples.binding_id) != str(self.binding_id) or str(samples.source_id) != str(self.source_id):
                raise ValueError("原生历史与冻结的目标、绑定和来源不一致")
        for catalog in self.metric_catalogs:
            if str(catalog.target_id) != str(self.target_id) or str(catalog.binding_id) != str(self.binding_id) or str(catalog.source_id) != str(self.source_id):
                raise ValueError("指标目录与冻结的目标、绑定和来源不一致")
        return self
