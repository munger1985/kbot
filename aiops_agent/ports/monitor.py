"""监控 Provider 与应用层之间的窄接口。"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from aiops_agent.contracts.monitoring import (
    MetricDefinition,
    MetricObservation,
    NormalizedWebhookBatch,
    ObservationGap,
)
from platform_core.contracts.aiops.types import UtcDatetime


class MonitorProviderContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    source_type: str
    source_version: int = Field(ge=1)
    endpoint: str
    credentials: dict[str, str] = Field(default_factory=dict)
    capabilities: dict[str, Any] = Field(default_factory=dict)


class MonitorHealthRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str


class MonitorHealthResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    healthy: bool
    error_code: str | None = None
    adapter_version: str


class MetricQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_id: str
    binding_id: str
    external_target_key: str
    metric_definitions: tuple[MetricDefinition, ...]
    window_start: UtcDatetime
    window_end: UtcDatetime
    requested_step_seconds: int = Field(ge=1)
    max_response_bytes: int = Field(ge=1024)
    trace_id: str


class MetricQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observations: tuple[MetricObservation, ...] = ()
    gaps: tuple[ObservationGap, ...] = ()


class AlertQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_id: str
    binding_id: str
    external_target_key: str
    window_start: UtcDatetime
    window_end: UtcDatetime
    max_alerts: int = Field(ge=1, le=10000)
    trace_id: str


class AlertQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alerts: tuple[dict[str, Any], ...] = ()
    gaps: tuple[ObservationGap, ...] = ()


class RawWebhookRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    headers: dict[str, str]
    body: bytes
    received_at: UtcDatetime


class MonitorPort(Protocol):
    async def health_check(
        self, request: MonitorHealthRequest
    ) -> MonitorHealthResult: ...

    async def query_metrics(
        self, request: MetricQueryRequest
    ) -> MetricQueryResult: ...

    async def query_alerts(
        self, request: AlertQueryRequest
    ) -> AlertQueryResult: ...

    async def verify_and_parse_webhook(
        self, request: RawWebhookRequest
    ) -> NormalizedWebhookBatch: ...
