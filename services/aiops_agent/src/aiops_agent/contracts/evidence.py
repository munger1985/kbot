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
    exact_item_key: str | None = Field(default=None, max_length=512)
    target_type: str | None = Field(default=None, max_length=128)
    metric_name: str | None = Field(default=None, max_length=256)
    required_labels: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_provider_locator(self) -> "ProviderMetricDefinition":
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
    semantic_version: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=1000)
    unit: str = Field(min_length=1, max_length=32)
    value_kind: Literal["GAUGE", "COUNTER", "STATE"]
    expected_dimensions: tuple[str, ...] = ()
    supported_db_types: tuple[str, ...] = ("ORACLE", "MYSQL")
    allowed_aggregations: tuple[str, ...] = ("AVG", "MAX", "LAST")
    default_window_seconds: int = Field(ge=60, le=604800)
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


class ObservationSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["OBSERVATION_SET.v1"] = "OBSERVATION_SET.v1"
    target_id: str
    binding_id: str
    source_id: str
    observations: tuple[MetricObservation, ...] = ()
    active_alerts: tuple[dict[str, Any], ...] = ()
    gaps: tuple[ObservationGap, ...] = ()
    collected_at: UtcDatetime
