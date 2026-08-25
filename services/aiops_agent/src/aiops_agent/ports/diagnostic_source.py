"""诊断源 Adapter 的分能力端口与调用契约。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aiops_agent.contracts.evidence import (
    MetricDefinition,
    MetricObservation,
    LogEvidenceSet,
    NormalizedSignalBatch,
    ObservationGap,
)
from platform_core.contracts.aiops.types import UtcDatetime


CAPABILITY_HEALTH_CHECK = "health.check"
CAPABILITY_EVENT_RECEIVE = "event.receive"
CAPABILITY_EVENT_QUERY = "event.query"
CAPABILITY_METRIC_QUERY_RANGE = "metric.query_range"
CAPABILITY_LOG_QUERY = "log.query"
CAPABILITY_DATABASE_QUERY_LIVE = "database.query_live"
CAPABILITY_DATABASE_QUERY_HISTORY = "database.query_history"
CAPABILITY_HOST_INSPECT = "host.inspect"
CAPABILITY_TOPOLOGY_RESOLVE = "topology.resolve"
CAPABILITY_CHANGE_QUERY = "change.query"
CAPABILITY_WORKLOAD_QUERY = "workload.query"
CAPABILITY_ACTION_EXECUTE = "action.execute"


@dataclass(frozen=True)
class DiagnosticSourceAdapterDescriptor:
    """对应用例层公开的稳定 Adapter 元数据。"""

    adapter_id: str
    adapter_version: str
    source_types: frozenset[str]
    capabilities: frozenset[str]


class DiagnosticSourceAdapterCatalogPort(Protocol):
    """诊断源配置校验只依赖目录，不依赖具体 Adapter 实现。"""

    def describe(
        self, *, adapter_id: str, adapter_version: str
    ) -> DiagnosticSourceAdapterDescriptor: ...


class DiagnosticSourceContext(BaseModel):
    """冻结一次 Adapter 调用所需的配置和凭据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    source_type: str
    adapter_id: str
    adapter_version: str
    config_version: int = Field(ge=1)
    endpoint: str | None = None
    credentials: dict[str, str] = Field(default_factory=dict)
    declared_capabilities: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class SourceHealthRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str


class SourceHealthResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    healthy: bool
    error_code: str | None = None
    adapter_id: str
    adapter_version: str
    discovered_capabilities: tuple[str, ...] = ()


class MetricsEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_id: str
    binding_id: str
    source_locator_key: str
    metric_definitions: tuple[MetricDefinition, ...]
    window_start: UtcDatetime
    window_end: UtcDatetime
    requested_step_seconds: int = Field(ge=1)
    max_response_bytes: int = Field(ge=1024)
    trace_id: str


class MetricsEvidenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observations: tuple[MetricObservation, ...] = ()
    gaps: tuple[ObservationGap, ...] = ()


class EventEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_id: str
    binding_id: str
    source_locator_key: str
    window_start: UtcDatetime
    window_end: UtcDatetime
    max_events: int = Field(ge=1, le=10000)
    trace_id: str


class EventEvidenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    events: tuple[dict[str, Any], ...] = ()
    gaps: tuple[ObservationGap, ...] = ()


class LogSourceLocator(BaseModel):
    """跨日志 Adapter 通用的精确标签定位契约。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    labels: dict[str, str]

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, labels: dict[str, str]) -> dict[str, str]:
        if not labels or len(labels) > 16:
            raise ValueError("日志定位必须包含 1 到 16 个精确标签")
        for name, value in labels.items():
            if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", name):
                raise ValueError("日志标签名称格式无效")
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ValueError("日志标签值必须是 1 到 256 字符的字符串")
        return labels


class LogEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_id: str
    binding_id: str
    source_locator_key: str
    selector_labels: dict[str, str]
    window_start: UtcDatetime
    window_end: UtcDatetime
    max_entries: int = Field(ge=1, le=5000)
    max_response_bytes: int = Field(ge=1024, le=20 * 1024 * 1024)
    trace_id: str


class SignalWebhookRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    headers: dict[str, str]
    body: bytes
    received_at: UtcDatetime


@runtime_checkable
class HealthCheckPort(Protocol):
    async def health_check(
        self, request: SourceHealthRequest
    ) -> SourceHealthResult: ...


@runtime_checkable
class EventSourcePort(Protocol):
    async def verify_and_normalize_webhook(
        self, request: SignalWebhookRequest
    ) -> NormalizedSignalBatch: ...


@runtime_checkable
class EventEvidencePort(Protocol):
    async def query_events(
        self, request: EventEvidenceRequest
    ) -> EventEvidenceResult: ...


@runtime_checkable
class MetricsEvidencePort(Protocol):
    async def query_metrics(
        self, request: MetricsEvidenceRequest
    ) -> MetricsEvidenceResult: ...


@runtime_checkable
class LogEvidencePort(Protocol):
    async def query_logs(
        self, request: LogEvidenceRequest
    ) -> LogEvidenceSet: ...


@runtime_checkable
class DatabaseEvidencePort(Protocol):
    async def query_database(self, request: BaseModel) -> BaseModel: ...


@runtime_checkable
class HostEvidencePort(Protocol):
    async def inspect_host(self, request: BaseModel) -> BaseModel: ...


@runtime_checkable
class TopologyEvidencePort(Protocol):
    async def resolve_topology(self, request: BaseModel) -> BaseModel: ...


@runtime_checkable
class ChangeEvidencePort(Protocol):
    async def query_changes(self, request: BaseModel) -> BaseModel: ...


@runtime_checkable
class SourceActionPort(Protocol):
    async def execute_action(self, request: BaseModel) -> BaseModel: ...
